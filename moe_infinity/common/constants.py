from transformers import (
    PretrainedConfig,
)
from ..models.modeling_qwen.modeling_qwen2_moe import Qwen2MoeForCausalLM

MODEL_MAPPING_NAMES = {
    "qwen": Qwen2MoeForCausalLM,
}

MODEL_MAPPING_TYPES = {
    "qwen": 4,
}


def parse_expert_type(config: PretrainedConfig) -> int:
    architecture = config.architectures[0].lower()
    arch = None
    for supp_arch in MODEL_MAPPING_NAMES:
        if supp_arch in architecture:
            arch = supp_arch
            break
    if arch is None:
        raise RuntimeError(
            f"The `load_checkpoint_and_dispatch` function does not support the architecture {architecture}. "
            f"Please provide a model that is supported by the function. "
            f"Supported architectures are {list(MODEL_MAPPING_NAMES.keys())}."
        )

    return MODEL_MAPPING_TYPES[arch]
