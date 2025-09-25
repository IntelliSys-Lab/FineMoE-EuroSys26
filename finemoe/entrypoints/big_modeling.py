from typing import Any, Union, Dict
import os
import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights

from accelerate.utils.versions import is_torch_version
from finemoe.common.constants import MODEL_MAPPING_NAMES
from finemoe.runtime import OffloadEngine
from finemoe.utils import get_checkpoint_paths, ArcherConfig
from finemoe.models import apply_rotary_pos_emb
import finemoe


class MoE:
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded and adds the various hooks that will make this model run properly (even if split across devices).

    Args:
        model_name_or_path (`str` or `os.PathLike`): The model to load. It can be:
            - a name of HuggingFace Transformers model
            - a path to a file containing a whole model state dict
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        config (`Dict` or `os.PathLike`): The MoE-Infinity configuration. It can be:
            - a Python dictionary containing the configuration
            - a path to a JSON file containing the configuration
    """

    def __init__(
        self,
        model_name_or_path: Union[str, os.PathLike],
        config: Union[str, os.PathLike, Dict] = None,
    ) -> None:
        # TODO: remove the torch version check once older versions are supported
        if not is_torch_version(">=", "2.0"):
            raise RuntimeError(
                "The `load_checkpoint_and_dispatch` function requires PyTorch >= 2.0. "
                "Please update PyTorch."
            )

        if config is None:
            default_config_path = os.path.join(
                os.path.dirname(__file__), "config.json")
            if not os.path.exists(default_config_path):
                raise RuntimeError(
                    "The `load_checkpoint_and_dispatch` function requires a configuration file. "
                    f"Please provide a configuration file or create a default one at {default_config_path}."
                )
            config = default_config_path

        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        architecture = model_config.architectures[0].lower()

        self.prefetch_distance = config.pop('prefetch_distance')
        self.store_capacity = config.pop('store_capacity')
        self.device = config.pop('device')
        self.eval_batch_size = config.pop('eval_batch_size')
        self.eval_max_length = config.pop('eval_max_length')
        self.eval_mode = config.pop('eval_mode')

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
        self.arch = arch
        model_cls = MODEL_MAPPING_NAMES[arch]
        if os.path.exists(model_name_or_path):
            checkpoint_paths = get_checkpoint_paths(model_name_or_path)
        else:
            checkpoint_paths = None
            # get the checkpoint download path from huggingface hub
            model_path = snapshot_download(
                model_name_or_path,
                cache_dir=os.environ.get("TRANSFORMERS_CACHE", None),
                ignore_patterns=["flax*", "tf*"],
            )
            if model_path is None:
                raise RuntimeError(
                    f"The `snapshot_download` function could not find the checkpoint {model_name_or_path}. "
                    f"Please provide a valid checkpoint."
                )
            checkpoint_paths = get_checkpoint_paths(model_path)

        if isinstance(config, dict):
            engine_config = ArcherConfig.load_from_json(config)
        else:
            engine_config = ArcherConfig.load_from_file(config)

        self.engine = OffloadEngine(
            capacity=self.store_capacity,
            config=model_config,
            prefetch_distance=self.prefetch_distance,
            device=self.device,
            eval_mode=self.eval_mode,
        )
        self.engine.ckpt_files = checkpoint_paths
        # self.engine.save(config.offload_path, checkpoint_paths)
        is_flash_attn_available = False
        try:
            import flash_attn
            is_flash_attn_available = True
        except ImportError:
            print(
                "[WARNING] FlashAttention is not available in the current environment. Using default attention."
            )
            pass
        with self.engine.init(cls=model_cls, ar_config=config):
            self.model = model_cls.from_pretrained(
                model_name_or_path,
                attn_implementation=(
                    "flash_attention_2" if is_flash_attn_available else "eager"
                ),
                is_flash_attn_available=is_flash_attn_available,
                trust_remote_code=True,
            )

    def _configure_hook(self, input_ids: torch.LongTensor):
        if self.arch == "qwen":
            finemoe.models.modeling_qwen.modeling_qwen2_moe.apply_rotary_pos_emb = (
                apply_rotary_pos_emb
            )
        else:
            raise ValueError(f"{self.arch} not supported")

        batch_size = input_ids.shape[0]
        self.seq_id_list = [
            self.engine.expert_tracer.create_entry() for i in range(batch_size)
        ]
        self.model.seq_id_list = self.seq_id_list
        self.model.model.seq_id_list = self.seq_id_list
        for module in self.engine.expert_layer_modules:
            module.seq_id_list = self.seq_id_list

        return self.seq_id_list

    def generate(self, input_ids: torch.LongTensor, **kwargs) -> Any:
        """
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        multinomial sampling, beam-search decoding, and beam-search multinomial sampling.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation. If `past` is used, only `bos_token_id` is used as
                prompt.
            **kwargs: Additional arguments for the generation method. Check the HuggingFace documentation of the model's
                `generate` method for the supported arguments.

        Returns:
            `torch.LongTensor` of shape `(batch_size, sequence_length)`:
                The generated sequences. Sequences shorter than `min_length` are padded with `pad_token_id`.
        """

        seq_id_list = self._configure_hook(input_ids)

        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(input_ids, **kwargs)

        for seq_id in seq_id_list:
            self.engine.expert_tracer.finish_entry(seq_id)

        return output_ids

    def forward(self, input_ids: torch.LongTensor, *args, **kwargs) -> Any:
        """
        Forwards the input through the model.

        Args:
            *args: Additional positional arguments for the model's forward method.
            **kwargs: Additional keyword arguments for the model's forward method.

        Returns:
            Any: The output of the model.
        """

        self._configure_hook(input_ids)

        return self.model(input_ids, *args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        """
        Forwards the input through the model.

        Args:
            *args: Additional positional arguments for the model's forward method.
            **kwargs: Additional keyword arguments for the model's forward method.

        Returns:
            Any: The output of the model.
        """
        return self.forward(*args, **kwargs)
