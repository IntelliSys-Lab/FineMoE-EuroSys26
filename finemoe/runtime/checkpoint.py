"""Prepared checkpoints with expert grouping based on MoE-Infinity's moe-store."""

import json
import os
from pathlib import Path
import tempfile

from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from transformers import AutoConfig, Qwen3_5MoeForCausalLM, GenerationConfig
from transformers.initialization import no_init_weights


def save_prepared(model, engine, path):
    """Atomically export model weights."""
    if engine._active or engine._closed:
        raise RuntimeError("export requires an open model outside a request")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors, seen = {}, set()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        # Safetensors requires separate storage for tied tensors.
        if value.data_ptr() in seen:
            value = value.clone()
        seen.add(value.data_ptr())
        tensors[name] = value
    sparse_id = 0
    for layer_id, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, "layer_id"):
            continue
        rows = [engine.cache.host_storage[sparse_id, e] for e in range(engine.num_experts)]
        first = rows[0]
        size = first.numel()
        adjacent = all(row.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
                       and row.data_ptr() == first.data_ptr() + e * size * first.element_size()
                       for e, row in enumerate(rows))
        tensors[f"model.layers.{layer_id}.mlp.experts"] = (
            first.as_strided((len(rows), size), (size, 1)) if adjacent else torch.stack(rows))
        sparse_id += 1
    config = model.config.to_dict()
    config["dtype"] = str(next(iter(engine.cache.host_weights.values()))[0].dtype).removeprefix("torch.")
    metadata = dict(config=json.dumps(config),
                    generation_config=json.dumps(model.generation_config.to_dict()))
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        save_file(tensors, temporary, metadata=metadata)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_prepared(path):
    """Load and validate a prepared checkpoint."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        config_dict = json.loads(metadata["config"])
        if config_dict.get("model_type") != "qwen3_5_moe_text" or config_dict.get("quantization_config"):
            raise ValueError("prepared checkpoints require an unquantized Qwen3.5-MoE text model")
        config = AutoConfig.for_model(config_dict.pop("model_type"), **config_dict)
        if config.dtype != torch.bfloat16:
            raise ValueError("prepared checkpoints require BF16 weights")
        # Keep non-parameter buffers initialized.
        with no_init_weights(), init_empty_weights():
            model = Qwen3_5MoeForCausalLM._from_config(
                config, dtype=torch.bfloat16, attn_implementation="sdpa",
                experts_implementation="eager")
        expected = model.state_dict()
        groups, expert_names = {}, set()
        for layer_id, layer in enumerate(model.model.layers):
            if not hasattr(layer.mlp, "experts"):
                continue
            prefix = f"model.layers.{layer_id}.mlp.experts"
            members = [f"{prefix}.gate_up_proj", f"{prefix}.down_proj"]
            groups[prefix] = members
            expert_names.update(members)
        required = (expected.keys() - expert_names) | groups.keys()
        if set(handle.keys()) != required:
            raise ValueError("prepared checkpoint must contain a complete set of model weights")
        state = {}
        for name in expected.keys() - expert_names:
            value = handle.get_tensor(name)
            if value.shape != expected[name].shape or value.dtype != torch.bfloat16:
                raise ValueError(f"invalid prepared-checkpoint tensor: {name}")
            state[name] = value
        for name, members in groups.items():
            shapes = [expected[key].shape for key in members]
            sizes = [shape[1:].numel() for shape in shapes]
            value = handle.get_tensor(name)
            if value.shape != (config.num_experts, sum(sizes)) or value.dtype != torch.bfloat16:
                raise ValueError(f"invalid prepared-checkpoint expert group: {name}")
            for part, shape, key in zip(value.split(sizes, dim=1), shapes, members):
                state[key] = part.view(shape)
        model.load_state_dict(state, strict=True, assign=True)
        model.tie_weights()
        generation = json.loads(metadata["generation_config"])
        # Preserve the saved generation settings.
        generation["_from_model_config"] = False
        model.generation_config = GenerationConfig.from_dict(generation)
    return model
