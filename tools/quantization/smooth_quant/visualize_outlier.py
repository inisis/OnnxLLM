import os
import functools
from collections import defaultdict

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Visualizer(object):
    def __init__(self, model:nn.Module, data_loader: DataLoader[torch.Tensor], save_folder: str = "."):
        self.model = model
        self.data_loader = data_loader
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
    
    def _get_activations(self) -> torch.Tensor:
        activations: dict[str, list[torch.Tensor]] = defaultdict(dict)

        def hook(module, input, output, name):
            activations[name]["activation"] = input[0].detach().cpu()
            activations[name]["weight"] = module.weight.detach().cpu()

        handles = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and "q_proj" in name:
                handles.append(module.register_forward_hook(functools.partial(hook, name=name)))

        # Forward pass
        self.model(self.data_loader["input_ids"], past_key_values=None, use_cache=False)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return activations

    def plot_3d(self, act, weight, name):
        """
        Plot both activation and weight surfaces side-by-side.
        """
        fig = plt.figure(figsize=(14, 6))

        # -------- Activation --------
        act_abs = np.abs(act)
        amax, amin = act_abs.max(), act_abs.min()
        tokens = np.arange(act.shape[0])
        channels = np.arange(act.shape[1])
        T, C = np.meshgrid(tokens, channels)
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(C, T, act_abs.reshape(act.shape[1], -1),
                        cmap='coolwarm', linewidth=0, antialiased=False)
        ax1.set_title(
                f"Activation\nmax={amax:.4f}, min={amin:.4f}",
                fontsize=12
            )

        # -------- Weight --------
        weight_abs = np.abs(weight)
        wmax, wmin = weight_abs.max(), weight_abs.min()        
        out_ch = np.arange(weight.shape[0])
        in_ch = np.arange(weight.shape[1])
        T2, C2 = np.meshgrid(out_ch, in_ch)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(C2, T2, weight_abs.t(), cmap='coolwarm',
                        linewidth=0, antialiased=False)
        ax2.set_title(
            f"Weight (Abs)\nmax={wmax:.4f}, min={wmin:.4f}",
            fontsize=12
        )

        plt.tight_layout()
        plt.savefig(f"{self.save_folder}/layer_{name}_act_and_weight.png",
                    format='png', dpi=300)
        plt.close()

    def _visualize_outliers(self, data: torch.Tensor):
        for name, data in data.items():
            self.plot_3d(data["activation"][0].float().abs().cpu().numpy(), data["weight"].float(), name)


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = "/data/llm/Meta-Llama-3-8B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype="auto"
    )
    print(model)
    inputs = tokenizer(
        ["In 2006 Boulter starred in the play Citizenship written by Mark Ravenhill . The play was part of a series which featured different playwrights , titled Burn / Chatroom / Citizenship . In a 2006 interview , fellow actor Ben Whishaw identified Boulter as one of his favorite co @-@ stars : \" I worked with Indira Boulter on a film called The Unloved , and she 's quite extraordinary . She 's very brave and very clever as well . \" "],
        return_tensors="pt",
        truncation=True,
    )

    model.eval()    

    print(inputs)
    visualizer = Visualizer(model, inputs)

    activations = visualizer._get_activations()
    visualizer._visualize_outliers(activations)
    