from transformers import PretrainedConfig
from typing import Tuple
import re
import torch
from transformers import PretrainedConfig


def parse_expert_dtype(config: PretrainedConfig) -> int:
    dtype = config.torch_dtype
    if dtype == torch.bfloat16:
        dtype = 0
    elif dtype == torch.float32:
        dtype = 1
    elif dtype == torch.float16:
        dtype = 2
    else:
        assert False, "Unknown dtype %s" % dtype

    return dtype


def parse_moe_param(config: PretrainedConfig) -> Tuple[int, int, int]:
    arch = config.architectures[0].lower()
    if "qwen" in arch:
        num_encoder_layers = 0
        num_decoder_layers = config.num_hidden_layers
        num_layers = config.num_hidden_layers
        num_experts = config.num_experts
        embed_dim = config.hidden_size
        top_k = config.num_experts_per_tok
    else:
        raise RuntimeError(f"Unsupported architecture {arch}")

    return num_layers, num_experts, num_encoder_layers, embed_dim, top_k


def parse_expert_id(param_name: str, config: PretrainedConfig) -> Tuple[int, int]:
    arch = config.architectures[0].lower()
    _, _, num_encoder_layers, _, _ = parse_moe_param(config)

    if "qwen" in arch:
        encoder_sparse_step = None
        decoder_sparse_step = 1
        layer_type = "decoder"

        # example "model.layers.0.mlp.experts.0.down_proj.weight"
        result = re.findall(
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.", param_name
        )
        if result:
            layer_id, expert_id = result[0]
            layer_id = int(layer_id)
            expert_id = int(expert_id)
    else:
        raise ValueError(f"{arch} not supported")

    if result:
        if layer_type == "decoder":
            layer_id = layer_id // decoder_sparse_step + num_encoder_layers
        elif layer_type == "encoder":
            layer_id = layer_id // encoder_sparse_step
        else:
            raise ValueError(f"Unsupported layer type {layer_type}")

        return layer_id, expert_id

    return None, None
