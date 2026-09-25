"""Qwen3.5-MoE loading and inference."""

from contextlib import contextmanager
import torch
from transformers import AutoConfig, Qwen3_5MoeForCausalLM

from finemoe.runtime.config import FineMoEConfig
from finemoe.runtime.model_offload import OffloadEngine


class MoE:
    def __init__(self, model_name_or_path, config=None, **load_kwargs):
        options = FineMoEConfig.load(config)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=False,
                                                  **{k: v for k, v in load_kwargs.items()
                                                     if k in ("revision", "cache_dir", "local_files_only", "token")})
        if model_config.model_type not in ("qwen3_5_moe", "qwen3_5_moe_text"):
            raise ValueError("This backend supports text inference with Qwen3.5-MoE")
        multimodal = model_config.model_type == "qwen3_5_moe"
        model_config = model_config.get_text_config()
        if any(key in load_kwargs for key in
               ("device_map", "trust_remote_code", "_fast_init", "output_loading_info", "experts_implementation",
                "dtype", "torch_dtype", "attn_implementation")):
            raise ValueError("FineMoE controls placement and complete-checkpoint loading")
        self.model, loading = Qwen3_5MoeForCausalLM.from_pretrained(
            model_name_or_path, config=model_config, dtype=torch.bfloat16,
            attn_implementation="sdpa", experts_implementation="eager", device_map="cpu",
            trust_remote_code=False, output_loading_info=True,
            **({"key_mapping": {r"^model\.language_model\.": "model."}} if multimodal else {}),
            **load_kwargs)
        if loading["missing_keys"] or loading["mismatched_keys"]:
            raise ValueError("FineMoE requires a complete checkpoint; "
                             f"missing={loading['missing_keys']}, mismatched={loading['mismatched_keys']}")
        self.engine = OffloadEngine(self.model, options)
        self.options = options

    @classmethod
    def from_model(cls, model, config=None):
        """Attach a loaded BF16 CPU Qwen3.5-MoE text model."""
        instance = cls.__new__(cls)
        instance.options = FineMoEConfig.load(config)
        instance.model = model
        instance.engine = OffloadEngine(model, instance.options)
        return instance

    @classmethod
    def from_prepared(cls, path, config=None):
        """Load a checkpoint exported by save_prepared."""
        from finemoe.runtime.checkpoint import load_prepared
        options = FineMoEConfig.load(config)
        return cls.from_model(load_prepared(path), options)

    def save_prepared(self, path):
        """Save packed weights for subsequent loads."""
        from finemoe.runtime.checkpoint import save_prepared
        save_prepared(self.model, self.engine, path)

    @contextmanager
    def request(self, batch_size):
        """Serialize a request containing one or more cached forward steps."""
        with torch.cuda.device(self.engine.device):
            self.engine.start_request(batch_size)
            try:
                with torch.inference_mode():
                    yield self.model
            finally:
                self.engine.finish_request()

    def _inputs(self, input_ids, kwargs):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [batch, sequence] tensor")
        kwargs = dict(kwargs)
        mask = kwargs.get("attention_mask")
        if mask is not None:
            if mask.ndim != 2 or mask.shape != input_ids.shape or not mask.bool().any(-1).all():
                raise ValueError("attention_mask must match input_ids and contain a token in every row")
            kwargs["attention_mask"] = mask.to(self.engine.device)
        return input_ids.to(self.engine.device), kwargs

    def generate(self, input_ids, **kwargs):
        generation_config = kwargs.get("generation_config", self.model.generation_config)
        for key in ("num_beams", "num_return_sequences"):
            if kwargs.get(key, getattr(generation_config, key, None)) not in (None, 1):
                raise ValueError(f"{key} > 1 is not supported by this offload backend")
        if kwargs.get("use_cache", getattr(generation_config, "use_cache", True)) is False:
            raise ValueError("generation requires use_cache=True")
        if kwargs.get("assistant_model") is not None:
            raise ValueError("assisted generation is not supported")
        input_ids, kwargs = self._inputs(input_ids, kwargs)
        with self.request(input_ids.shape[0]):
            return self.model.generate(input_ids, **kwargs)

    def forward(self, input_ids, **kwargs):
        input_ids, kwargs = self._inputs(input_ids, kwargs)
        with self.request(input_ids.shape[0]):
            return self.model(input_ids, **kwargs)

    __call__ = forward

    def warmup(self):
        self.engine.warmup()

    def close(self):
        self.engine.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
