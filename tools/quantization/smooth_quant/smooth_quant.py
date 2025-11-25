import inspect
import functools
from tqdm import tqdm
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scale import apply_scale

def get_named_linears(module: nn.Module) -> dict[str, nn.Linear]:
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_op_name(root_module: nn.Module, op: nn.Module) -> str:
    for name, submodule in root_module.named_modules():
        if submodule is op:
            return name  # type: ignore
    raise ValueError(f"Cannot find op {op} in module {root_module}")

def get_nested_attr_from_module(obj: nn.Module, attr_path: str) -> Any:
    """
    Retrieves the value of a nested attribute based on a given attribute path string.

    Parameters:
    - obj: The starting object.
    - attr_path: The string representing the attribute path, such as "model.decoder.layers".

    Returns:
    - The value of the nested attribute.
    """

    return functools.reduce(getattr, attr_path.split("."), obj)

def get_layers_for_scaling(
    module: nn.Module, input_feat: dict[str, Any], module_kwargs: dict[str, Any], scaling_layers: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    def get_dense_layers(
        module: nn.Module,
        input_feat: dict[str, Any],
        module_kwargs: dict[str, Any],
        layer: dict[str, Any],
        layers: list[dict[str, Any]],
        has_kwargs: bool,
    ) -> bool:
        if layer["inp"] in input_feat:  # hooked inputs
            linear_layers = []
            for i in range(len(layer["layers"])):
                linear_layers.append(get_nested_attr_from_module(module, layer["layers"][i]))

            layer_dict = dict(
                prev_op=get_nested_attr_from_module(module, layer["prev_op"]),
                layers=linear_layers,
                inp=input_feat[layer["inp"]],
            )

            if "module2inspect" in layer and layer["module2inspect"] is not None:
                if layer["module2inspect"] == "":
                    layer_dict["module2inspect"] = module
                else:
                    layer_dict["module2inspect"] = get_nested_attr_from_module(module, layer["module2inspect"])
            if has_kwargs:
                layer_dict["kwargs"] = module_kwargs
                has_kwargs = False

            layers.append(layer_dict)

        return has_kwargs

    layers: list[dict[str, Any]] = []
    has_kwargs = True  # For first layer from module, input kwargs.

    for layer in scaling_layers:
        _ = get_nested_attr_from_module(module, layer["layers"][0])  # check is_moe
        has_kwargs = get_dense_layers(module, input_feat, module_kwargs, layer, layers, has_kwargs)

    return layers

def cache_model_inps(
    model: nn.Module, modules: nn.ModuleList, samples: DataLoader[torch.Tensor]
) -> tuple[nn.ModuleList, dict[str, Any], list[torch.Tensor]]:
    inps: list[torch.Tensor] = []
    layer_args: list[Union[torch.Tensor, None]] = []
    layer_kwargs: dict[str, Any] = {}

    # get input and kwargs to layer 0
    # with_kwargs is only supported in PyTorch 2.0
    # use this Catcher hack for now
    class Catcher(nn.Module):
        def __init__(
            self,
            module: nn.Module,
            inps: list[torch.Tensor],
            layer_args: list[Union[torch.Tensor, None]],
            layer_kwargs: dict[str, Any],
        ) -> None:
            super().__init__()
            self.module = module
            self.inps = inps
            self.layer_args = layer_args
            self.layer_kwargs = layer_kwargs

        # in case need module's attribute is explicitly needed
        def __getattr__(self, name: str) -> Any:
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, *args: torch.Tensor, **kwargs: Any) -> None:
            # assume first input to forward is hidden states
            if len(args) > 0:
                hidden_states = args[0]
                if len(self.layer_args) == 0:
                    self.layer_args.extend(
                        args[1:]
                    )  # For attention_mask and rotary_pos_emb, the value of the new input is always same, so it is kept once
            else:
                first_key = list(kwargs.keys())[0]
                hidden_states = kwargs.pop(first_key)

            self.inps.append(hidden_states)
            self.layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs

    cur_layer_device = torch.device("cpu")
    required_kwargs = inspect.signature(modules[0].forward).parameters
    modules[0] = Catcher(modules[0], inps, layer_args, layer_kwargs)
    for sample in samples:
        if isinstance(sample, torch.Tensor):
            try:
                model(sample.to(cur_layer_device), use_cache=False)
            except ValueError:  # work with early exit
                pass
        else:
            try:
                model(**{key: val.to(cur_layer_device) for key, val in sample.items()})
            except ValueError:  # work with early exit
                pass
    del samples
    modules[0] = modules[0].module  # restore

    arg_idx = 0

    for k, v in required_kwargs.items():
        if k == "hidden_states" or k in layer_kwargs or v.kind == v.VAR_KEYWORD:
            # `layer_args` here holds the positional arguments from position one, so
            # `arg_idx` is not incremented here.
            continue
        elif arg_idx < len(layer_args):  # pragma: no cover
            layer_kwargs[k] = layer_args[arg_idx]
            arg_idx += 1
        else:
            break

    return modules, layer_kwargs, inps

class SmoothQuantProcessor():
    def __init__(self, model: nn.Module, data_loader: DataLoader[torch.Tensor]) -> None:
        self.model = model
        self.device = model.device
        self.data_loader = data_loader
        self.alpha = 1
        self.clamp_min = 0.001
        self.model_decoder_layers = "model.layers"
        self.scaling_layers = [{
                                "prev_op": "input_layernorm",
                                "layers": [
                                    "self_attn.q_proj",
                                    "self_attn.k_proj",
                                    "self_attn.v_proj"
                                ],
                                "inp": "self_attn.q_proj",
                                "module2inspect": "self_attn"
                                },
                                {
                                "prev_op": "self_attn.v_proj",
                                "layers": [
                                    "self_attn.o_proj"
                                ],
                                "inp": "self_attn.o_proj"
                                },
                                {
                                "prev_op": "post_attention_layernorm",
                                "layers": [
                                    "mlp.gate_proj",
                                    "mlp.up_proj"
                                ],
                                "inp": "mlp.gate_proj",
                                "module2inspect": "mlp"
                                },
                                {
                                "prev_op": "mlp.up_proj",
                                "layers": [
                                    "mlp.down_proj"
                                ],
                                "inp": "mlp.down_proj"
                                }]
        self.modules, self.module_kwargs, self.inps = self.init_quant()
        self.num_attention_heads, self.num_key_value_heads = model.config.num_attention_heads, model.config.num_key_value_heads

    def apply(self) -> None:
        for i in tqdm(range(len(self.modules)), desc="Applying SmoothQuant"):
            named_linears = get_named_linears(self.modules[i])
            input_feat, act_scales = self._get_act_scale_and_input_feat(self.modules[i], named_linears)   
            module_config: list[dict[str, Any]] = get_layers_for_scaling(self.modules[i], input_feat, self.module_kwargs, self.scaling_layers)
            scales_list = []
            for layer in module_config:
                scales = self._search_best_scale(
                    self.modules[i], act_scales, **layer
                )  # scales: (pre_layer, layer, best_scales)
                if scales is not None:
                    print(
                        f"SmoothQuant for layer {i}: {scales[1]}, scales_max={scales[2].max().item()}, scales_min={scales[2].min().item()}"
                    )
                    scales_list.append(scales)
                else:
                    print(f"SmoothQuant is skipped for layer {i}: {scales[1]}!")

            apply_scale(
                self.modules[i],
                scales_list,
                input_feat_dict=None,
                device=self.device,
                num_attention_heads=self.num_attention_heads,
                num_key_value_heads=self.num_key_value_heads,
            )


    @torch.no_grad()
    def _search_best_scale(
        self,
        module: nn.Module,
        act_scales: dict[str, torch.Tensor],
        prev_op: nn.Module,
        layers: list[nn.Linear],
        inp: torch.Tensor,
        module2inspect: nn.Module | None = None,
        kwargs: dict[str, Any] = {},
    ) -> tuple[str, tuple[str, ...], torch.Tensor]:
        for fc in layers:
            assert isinstance(fc, nn.Linear)
            assert fc.in_features == act_scales[get_op_name(module, fc)].numel()

        # [STEP 1]: Compute maximum of weight
        device, dtype = layers[0].weight.device, layers[0].weight.dtype
        layer_act_scales = act_scales[get_op_name(module, layers[0])].to(device).to(dtype)
        weight_scales = torch.cat([fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in layers], dim=0)
        weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)
        # [STEP 2]: Balance quant error between weight and act
        best_scales = (layer_act_scales.pow(self.alpha) / weight_scales.pow(1 - self.alpha)).clamp(min=self.clamp_min)
        assert torch.isnan(best_scales).sum() == 0, best_scales
        return (get_op_name(module, prev_op), tuple([get_op_name(module, m) for m in layers]), best_scales)
    

    @torch.no_grad()
    def _get_act_scale_and_input_feat(
        self, layer: nn.ModuleList, named_linears: dict[str, nn.Linear]
    ) -> tuple[dict[str, list[torch.Tensor]], dict[str, torch.Tensor]]:
        act_scales: dict[str, torch.Tensor] = {}
        num_batches = len(self.inps)
        layer_inputs = [inp for inp in self.inps]
        cur_layer_device = torch.device("cpu")
        layer_outputs = []

        def stat_tensor(name: str, tensor: torch.Tensor) -> None:
            hidden_dim = tensor.shape[-1]
            tensor = tensor.view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(tensor, dim=0)[0].float()
            if name in act_scales:
                act_scales[name] = torch.max(act_scales[name], comming_max)
            else:
                act_scales[name] = comming_max

        # collect act scale
        def cache_input_hook(
            m: nn.Module,
            x: tuple[torch.Tensor, ...],
            y: torch.Tensor,
            name: str,
            feat_dict: dict[str, list[torch.Tensor]],
        ) -> None:
            x = x[0]
            x = x.detach()
            feat_dict[name] = []
            stat_tensor(name, x)

        input_feat: dict[str, list[torch.Tensor]] = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )

        for j in range(num_batches):
            layer_input = layer_inputs[j]
            output = layer(layer_input, **self.module_kwargs)
            torch.onnx.export(layer, (layer_input, self.module_kwargs), "llama.onnx", dynamo=True, opset_version=23)
            raise
            if isinstance(output, tuple):
                layer_output = output[0]
            elif isinstance(output, torch.Tensor):
                layer_output = output
            else:
                raise ValueError(f"Unexpected output type: {type(output)}")

            layer_outputs.append(layer_output)

        # get output as next layer's input
        self.inps = layer_outputs
        for h in handles:
            h.remove()

        return input_feat, act_scales

    def init_quant(self) -> tuple[nn.ModuleList, dict[str, Any], list[torch.Tensor]]:
        assert self.model_decoder_layers is not None
        modules = self.model.model.layers
        modules, layer_kwargs, inputs = cache_model_inps(self.model, modules, self.data_loader)

        return modules, layer_kwargs, inputs

if __name__ == "__main__":
    from visualize_outlier import Visualizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = "/data/llm/Meta-Llama-3-8B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype="auto", attn_implementation="eager"
    )
    model.config.use_cache = False
    model.eval()

    inputs = tokenizer(
        ["In 2006 Boulter starred in the play Citizenship written by Mark Ravenhill . The play was part of a series which featured different playwrights , titled Burn / Chatroom / Citizenship . In a 2006 interview , fellow actor Ben Whishaw identified Boulter as one of his favorite co @-@ stars : \" I worked with Indira Boulter on a film called The Unloved , and she 's quite extraordinary . She 's very brave and very clever as well . \" "],
        return_tensors="pt",
        truncation=True,
    )

    data_loader = [inputs]

    visualizer = Visualizer(model, inputs, "./float")
    activations = visualizer._get_activations()
    visualizer._visualize_outliers(activations)    

    processor = SmoothQuantProcessor(model, data_loader)
    processor.apply()
    
    visualizer_smooth = Visualizer(processor.model, inputs, "./smooth")

    activations_smooth = visualizer_smooth._get_activations()
    visualizer_smooth._visualize_outliers(activations_smooth)
