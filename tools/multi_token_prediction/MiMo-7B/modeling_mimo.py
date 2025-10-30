from typing import Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen2.modeling_qwen2 import (Qwen2Attention,
                                                      Qwen2ForCausalLM,
                                                      Qwen2MLP, Qwen2Model,
                                                      Qwen2RMSNorm)
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache

from .configuration_mimo import MiMoConfig


class MiMoMTPLayers(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.token_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.final_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx=0)
        self.mlp = Qwen2MLP(config)

    def forward(self, input_embeds,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values: Optional[Cache]=None,
                    output_attentions: Optional[bool]=False,
                    use_cache: Optional[bool]=False,
                    position_embedding: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                    cache_position=None,
                    **kwargs):

        input_embeds = self.token_layernorm(input_embeds)
        previous_hidden_states = self.hidden_layernorm(hidden_states)
        hidden_states = self.input_proj(torch.cat([previous_hidden_states, input_embeds], dim=-1))
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + input_embeds.shape[1], device=input_embeds.device
            )

        position_ids = cache_position.unsqueeze(0)
        hidden_states, _ = self.self_attn(hidden_states,
                                       attention_mask=attention_mask,
                                       position_ids=position_ids,
                                       past_key_value=past_key_values,
                                       output_attentions=output_attentions,
                                       use_cache=use_cache,
                                       cache_position=cache_position,
                                       position_embedding=position_embedding,
                                       **kwargs)

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layernorm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class MiMoModel(Qwen2Model):
    config_class = MiMoConfig

    def __init__(self, config: MiMoConfig):
        super().__init__(config)
        self.mtp_layers = nn.ModuleList([MiMoMTPLayers(config) for _ in range(config.num_nextn_predict_layers)])


class MiMoForCausalLM(Qwen2ForCausalLM):
    config_class = MiMoConfig
    def __init__(self, config: MiMoConfig):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = MiMoModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
