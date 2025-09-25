# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

import torch.nn.functional as F
import gc
import os
import numpy as np
# from auto_gptq.nn_modules.qlinear.qlinear_cuda import QuantLinear
# from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear as QuantLinearOld

import torch
import functools
import json

from tqdm import tqdm

from finemoe.ops.op_builder.prefetch import PrefetchBuilder
from finemoe.models import (
    SyncQwen2MoeSparseMoeBlock,
    Qwen2MoeMLP,
)
from finemoe.utils import ArcherConfig
from finemoe.utils.arguments import copy_args_to_device, copy_kwargs_to_device

from finemoe.memory import ExpertPrefetcher
import finemoe
from finemoe.utils import (
    parse_moe_param,
    parse_expert_id,
    parse_expert_dtype,
)
from finemoe.common import parse_expert_type
from finemoe.memory import ExpertTracer

from typing import Dict, Type, Union
from transformers import (
    AutoConfig,
)
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
import transformers
from typing import Callable

from safetensors import safe_open

import re

use_jit = False
try:
    import finemoe.ops.prefetch.prefetch_op as prefetch_op
except ImportError:
    print(f"Do not detect pre-installed ops, use JIT mode")
    use_jit = True


# class ArcherException(Exception):
#     pass


class ExpertMapStore():
    def __init__(
        self,
        capacity,
        num_layers,
        num_experts,
        embed_dim,
        prefetch_distance,
        device,
    ):
        self.capacity = capacity
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.embed_dim = embed_dim
        self.prefetch_distance = prefetch_distance
        self.device = torch.device(device)
        self.dtype = torch.float32

        self.store_embed = torch.zeros(
            (capacity, embed_dim), dtype=self.dtype, device=self.device)
        self.store_traj = torch.zeros(
            (capacity, num_layers, num_experts), dtype=self.dtype, device=self.device)

        self.data_size = 0

    def import_store_data(self, state_path):
        self.store_embed = torch.from_numpy(
            np.load(f"{state_path}~embed~{self.capacity}.npy",
                    allow_pickle=False)
        ).to(self.device, dtype=self.dtype, non_blocking=True)

        self.store_traj = torch.from_numpy(
            np.load(f"{state_path}~traj~{self.capacity}.npy",
                    allow_pickle=False)
        ).to(self.device, dtype=self.dtype, non_blocking=True)

        self.data_size = self.store_embed.shape[0]

    def export_store_data(self, state_path):
        np.save(f"{state_path}~embed~{self.capacity}.npy",
                self.store_embed.detach().cpu().numpy(), allow_pickle=False)
        np.save(f"{state_path}~traj~{self.capacity}.npy",
                self.store_traj.detach().cpu().numpy(), allow_pickle=False)

    @torch.inference_mode()
    def _cosine_sim(self, A: torch.Tensor, B: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        A = torch.nn.functional.normalize(A, dim=-1, eps=eps)
        B = torch.nn.functional.normalize(B, dim=-1, eps=eps)
        return A @ B.T

    def _ensure_tensor(self, x, shape_last=None):
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.as_tensor(x)
        if shape_last is not None:
            assert t.shape[-len(shape_last):] == tuple(
                shape_last), f"Expected trailing shape {shape_last}, got {tuple(t.shape)}"
        return t.to(self.device, dtype=self.dtype, non_blocking=True)

    @torch.inference_mode()
    def add(self, embeds, expert_maps):
        embeds = self._ensure_tensor(embeds, shape_last=(self.embed_dim,))
        expert_maps = self._ensure_tensor(
            expert_maps, shape_last=(self.num_layers, self.num_experts))
        B = embeds.size(0)
        if B == 0:
            return

        free = self.capacity - self.data_size
        take = min(free, B)
        if take > 0:
            idx = slice(self.data_size, self.data_size + take)
            self.store_embed[idx] = embeds[:take]
            self.store_traj[idx] = expert_maps[:take]
            self.data_size += take

        rem = B - take
        if rem > 0:
            S_e = F.cosine_similarity(
                embeds[-rem:].unsqueeze(1),
                self.store_embed[:self.data_size].unsqueeze(0),
                dim=-1,
            )
            sims_t = F.cosine_similarity(
                expert_maps[-rem:].reshape(rem, -1).unsqueeze(1),
                self.store_traj[:self.data_size].reshape(
                    self.data_size, -1).unsqueeze(0),
                dim=-1,
            )
            S_t = sims_t
            w = self.prefetch_distance / float(self.num_layers)
            redundant = w * S_e + (1.0 - w) * S_t
            evict_idx = torch.argmax(redundant, dim=1)
            self.store_embed[evict_idx] = embeds[-rem:]
            self.store_traj[evict_idx] = expert_maps[-rem:]

        self.data_size = min(self.capacity, self.data_size)

    @torch.inference_mode()
    def match_embed(self, embeds):
        if self.data_size == 0:
            return None, None
        embeds = self._ensure_tensor(embeds, shape_last=(self.embed_dim,))
        sims = F.cosine_similarity(
            embeds.unsqueeze(1),
            self.store_embed[:self.data_size].unsqueeze(0),
            dim=-1,
        )
        scores, argmax = sims.max(dim=1)
        maps = self.store_traj[:self.data_size][argmax]
        return scores, maps

    @torch.inference_mode()
    def match_traj(self, trajs):
        if self.data_size == 0:
            return None, None
        trajs = self._ensure_tensor(trajs, shape_last=(
            trajs.shape[-2], self.num_experts))
        L_obs = trajs.shape[1]
        B = trajs.shape[0]
        sims = F.cosine_similarity(
            trajs[:, None, :L_obs, :].reshape(B, 1, -1),
            self.store_traj[:self.data_size, :L_obs, :][None,
                                                        :, :, :].reshape(1, self.data_size, -1),
            dim=-1,
        )
        scores, argmax = sims.max(dim=1)
        maps = self.store_traj[:self.data_size][argmax]
        return scores, maps


class ExpertMapMatcher():
    def __init__(
        self,
        expert_tracer,
        expert_map_store,
        expert_prefetcher,
        prefetch_distance,
    ):
        self.expert_tracer = expert_tracer
        self.expert_map_store = expert_map_store
        self.expert_prefetcher = expert_prefetcher

        self.prefetch_distance = prefetch_distance
        self.num_layers = self.expert_map_store.num_layers
        self.num_experts = self.expert_map_store.num_experts
        self.embed_dim = self.expert_map_store.embed_dim
        self.top_k = self.expert_tracer.top_k

        self.device = self.expert_map_store.device

    @torch.inference_mode()
    def _select_by_cumsum(self, probs: torch.Tensor, threshold: torch.Tensor, top_k: int):
        vals, idx = torch.sort(probs, dim=-1, descending=True)
        csum = vals.cumsum(-1)
        th = threshold.view(-1, 1) if threshold.ndim == 1 else threshold
        has_pos = vals.gt(0).any(-1)
        k = (csum <= th).sum(-1)
        k_nonzero = torch.clamp(k, min=top_k, max=vals.size(-1))
        k = torch.where(has_pos, k_nonzero, torch.zeros_like(k))
        ar = torch.arange(vals.size(-1), device=probs.device).unsqueeze(0)
        keep_sorted = ar < k.unsqueeze(-1)
        mask = torch.zeros_like(keep_sorted, dtype=torch.bool)
        mask.scatter_(dim=-1, index=idx, src=keep_sorted)
        return probs * mask

    @torch.inference_mode()
    def _layer_decay_weights(self, layer_start: int, layer_end: int) -> torch.Tensor:
        assert layer_end > layer_start
        w = torch.ones(self.num_layers, device=self.device)
        rng = torch.arange(self.num_layers, device=self.device)
        band = (rng >= layer_start) & (rng < layer_end)
        w[band] = -1 / (layer_end + 1) * (rng[band] - layer_start) + 1
        return w

    @torch.inference_mode()
    def process_expert_map(self, layer_start: int, layer_end: int,
                           score: torch.Tensor, expert_map: torch.Tensor):
        probs = expert_map.clone()
        expert_prob_map = probs
        if layer_start > 0:
            probs[:layer_start, :].zero_()
        if layer_end < self.num_layers:
            probs[layer_end:, :].zero_()
        prefetch_priority_map = self._select_by_cumsum(
            probs, torch.clamp(1 - score, 0, 1), self.top_k)
        decay = self._layer_decay_weights(
            layer_start, layer_end).unsqueeze(-1)
        prefetch_priority_map = prefetch_priority_map * decay
        prefetch_priority_map[layer_start:layer_end] += 1e-6
        return prefetch_priority_map, expert_prob_map

    @torch.inference_mode()
    def embed_prefetch(self, seq_id: int, input_embeds: torch.Tensor):
        seq_len = input_embeds.shape[0]
        scores, maps = self.expert_map_store.match_embed(input_embeds)
        if scores is not None and maps is not None:
            layer_start = 0
            layer_end = self.prefetch_distance

            for i, (s, m) in enumerate(zip(scores, maps)):
                pred_map, prob_map = self.process_expert_map(
                    layer_start, layer_end, s, m)
                iter_id = i if seq_len > 1 else -1
                self.expert_tracer.update_preds(
                    seq_id, iter_id, pred_map, layer_start, layer_end)
                self.expert_prefetcher.prefetch_experts(
                    pred_map, prob_map)

    @torch.inference_mode()
    def traj_prefetch(self, seq_id: int, input_trajs: torch.Tensor):
        seq_len = input_trajs.shape[0]
        num_layers_obs = input_trajs.shape[1]
        layer_start = num_layers_obs + self.prefetch_distance
        if layer_start < self.num_layers:
            layer_end = self.num_layers
            scores, maps = self.expert_map_store.match_traj(
                input_trajs)

            if scores is not None and maps is not None:
                for i, (s, m) in enumerate(zip(scores, maps)):
                    pred_map, prob_map = self.process_expert_map(
                        layer_start, layer_end, s, m)
                    iter_id = i if seq_len > 1 else -1
                    self.expert_tracer.update_preds(
                        seq_id, iter_id, pred_map, layer_start, layer_end)
                    self.expert_prefetcher.prefetch_experts(
                        pred_map, prob_map)


class OffloadEngine(object):
    param_id = 0
    request_id = 0
    # request_id_flag = False

    def __init__(
        self,
        capacity,
        config: PretrainedConfig,
        prefetch_distance,
        device,
        eval_mode,
    ):
        self.offload_exemption = set()
        self.expert_modules = []

        self.model_create_counter = None

        self.ckpt_files = []
        self.config = config

        if capacity is None:
            capacity = 1000

        self.num_layers, self.num_experts, self.num_encoder_layers, self.embed_dim, self.top_k = parse_moe_param(
            self.config)

        self.expert_map_store = ExpertMapStore(
            capacity=capacity,
            num_layers=self.num_layers,
            num_experts=self.num_experts,
            embed_dim=self.embed_dim,
            prefetch_distance=prefetch_distance,
            device=device,
        )

        self.expert_tracer = ExpertTracer(
            capacity, config, self.expert_map_store, eval_mode, device)

        self.quant_method = None

        self.prefetch_distance = prefetch_distance
        self.device = torch.device(device)
        self.eval_mode = eval_mode

        self.moe_layers = []

    def init_expert_map_matcher(self):
        self.expert_map_matcher = ExpertMapMatcher(
            expert_tracer=self.expert_tracer,
            expert_map_store=self.expert_map_store,
            expert_prefetcher=self.expert_prefetcher,
            prefetch_distance=self.prefetch_distance,
        )

    def init(
        self, cls: Type[PreTrainedModel], ar_config: Union[str, Dict, ArcherConfig]
    ):

        self.cls = cls
        self.name_id_map = {}
        self.tensor_id_map = {}
        self.registered_tensors = set()
        self.forward_hooks = []
        self.backward_hooks = []

        self.offload_set = set()

        if isinstance(ar_config, str):
            _archer_config = ArcherConfig.load_from_file(ar_config)
        elif isinstance(ar_config, dict):
            _archer_config = ArcherConfig.load_from_json(ar_config)
        elif isinstance(ar_config, ArcherConfig):
            _archer_config = ar_config
        else:
            raise ValueError(
                "ArcherConfig is not provided. Please provide a path to a config file or a dict."
            )

        # TODO: get trace from trace_path

        self.checkpoint = _archer_config.offload_path

        os.makedirs(self.checkpoint, exist_ok=True)

        self.prefetch_lib = PrefetchBuilder().load() if use_jit else prefetch_op
        self.archer_engine = self.prefetch_lib.prefetch_handle(
            self.checkpoint,
            _archer_config.device_memory_ratio,
        )

        self.archer_config = _archer_config

        self.expert_tracer.offload_engine = self

        return self

    def __enter__(self):

        def do_nothing_decorator(orig_func: Callable) -> Callable:

            @functools.wraps(orig_func)
            def do_nothing(*args, **kwargs):
                pass

            return do_nothing

        def post_init_decorator(orig_post_init: Callable) -> Callable:
            # FIXME: this is a hacky way to get rid of the write to weight in the post_init, need a better way to do this if we need to support model training
            @functools.wraps(orig_post_init)
            def archer_post_init(cls, *args, **kwargs):
                pass

            return archer_post_init

        def torch_index_select_decorator(orig_torch_index_select: Callable):

            @functools.wraps(orig_torch_index_select)
            def archer_torch_index_select(input, dim, index):
                return orig_torch_index_select(input, dim, index.to(input.device)).to(self.device)

            return archer_torch_index_select

        def apply_to_model_decorator(orig_apply_to_model: Callable) -> Callable:

            @functools.wraps(orig_apply_to_model)
            def archer_apply_to_model(cls, fn):
                for name, param in cls.named_parameters(recurse=True):
                    if name not in self.name_id_map:
                        continue
                    param.data = torch.zeros(
                        1, dtype=param.dtype, device=param.device, pin_memory=True
                    )

                for name, buffer in cls.named_buffers(recurse=True):
                    if name not in self.name_id_map:
                        continue
                    buffer.data = torch.zeros(
                        1, dtype=buffer.dtype, device=buffer.device, pin_memory=True
                    )

            return archer_apply_to_model

        def init_decorator(orig_init: Callable) -> Callable:

            @functools.wraps(orig_init)
            def archer_init(cls, config, *args, **kwargs):
                # self.config = config
                pass

            return archer_init

        def param_init_decorator(orig_param_init: Callable) -> Callable:

            @functools.wraps(orig_param_init)
            def archer_param_init(cls, *args, **kwargs):
                orig_param_init(cls, *args, **kwargs)

                cls.param_real_shape = {}
                for name, param in cls.named_parameters(recurse=False):
                    cls.param_real_shape[name] = param.shape
                    param.data = torch.zeros(
                        1, dtype=param.dtype, device=param.device)
                    self.model_create_counter.update(1)

                for name, buf in cls.named_buffers(recurse=False):
                    cls.param_real_shape[name] = buf.shape
                    buf.data = torch.zeros(
                        1, dtype=buf.dtype, device=buf.device)
                    self.model_create_counter.update(1)

            return archer_param_init

        def cast_classifier_decorator(orig_cast_classifier: Callable) -> Callable:

            @functools.wraps(orig_cast_classifier)
            def archer_cast_classifier(cls, *args, **kwargs):
                orig_data_ptr = cls.classifier.weight.data.data_ptr()
                if orig_data_ptr in self.offload_set:
                    self.offload_set.remove(
                        cls.classifier.weight.data.data_ptr())
                    orig_cast_classifier(cls, *args, **kwargs)
                    new_data_ptr = cls.classifier.weight.data.data_ptr()
                    self.offload_set.add(cls.classifier.weight.data.data_ptr())
                    self.archer_engine.update_tensor_map(
                        orig_data_ptr, new_data_ptr)
                else:
                    orig_cast_classifier(cls, *args, **kwargs)
                    self.offload_set.add(cls.classifier.weight.data.data_ptr())

            return archer_cast_classifier

        self.cls._old_init = self.cls.__init__
        self.cls.__init__ = init_decorator(self.cls._old_init)
        torch.nn.modules.module.Module._old_apply = torch.nn.modules.module.Module.apply
        torch.nn.modules.module.Module.apply = apply_to_model_decorator(
            torch.nn.modules.module.Module._old_apply
        )

        torch._old_index_select = torch.index_select
        torch.index_select = torch_index_select_decorator(
            torch._old_index_select)
        torch.Tensor._old_index_select = torch.Tensor.index_select
        torch.Tensor.index_select = torch_index_select_decorator(
            torch.Tensor._old_index_select
        )

        self.cls._old_post_init = self.cls.post_init
        self.cls.post_init = post_init_decorator(self.cls._old_post_init)
        PreTrainedModel._old_post_init = PreTrainedModel.post_init
        PreTrainedModel.post_init = post_init_decorator(
            PreTrainedModel._old_post_init)

        for name, module in torch.nn.modules.__dict__.items():
            if not isinstance(module, type):
                continue
            if not issubclass(module, torch.nn.modules.module.Module):
                continue
            if name in [
                "Module",
                "Sequential",
                "ModuleDict",
                "ModuleList",
                "ParameterList",
                "ParameterDict",
            ]:
                continue
            module._old_init = module.__init__
            module.__init__ = param_init_decorator(module.__init__)

            if hasattr(module, "reset_parameters"):
                module._old_reset_parameters = module.reset_parameters
                module.reset_parameters = do_nothing_decorator(
                    module.reset_parameters)

        finemoe.models.modeling_qwen.modeling_qwen2_moe._old_sparse_mlp = (
            finemoe.models.modeling_qwen.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock
        )
        finemoe.models.modeling_qwen.modeling_qwen2_moe.Qwen2MoeSparseMoeBlock = (
            SyncQwen2MoeSparseMoeBlock
        )

        def from_pretrained_decorator(orig_from_pretrained: Callable) -> Callable:

            @functools.wraps(orig_from_pretrained)
            def archer_from_pretrained(cls, *args, **kwargs):
                # print("Creating model from scratch ...")

                name_id_map_file = os.path.join(
                    self.checkpoint, "name_id_map.json")

                model_name = args[0]
                self.dtype = parse_expert_dtype(self.config)

                self.dtype_cls = self.config.torch_dtype

                if (
                    not self.archer_engine.is_tensor_index_initialized()
                    or not os.path.exists(name_id_map_file)
                ):
                    print("Creating model from scratch ...", flush=True)

                    self.cls.__init__ = self.cls._old_init

                    empty_state_dict = {}
                    self.name_id_map = {}
                    for ckpt in tqdm(
                        self.ckpt_files, desc="Loading checkpoint files", smoothing=0
                    ):
                        state_dict = {}
                        if "safetensors" in ckpt:
                            with safe_open(ckpt, framework="pt", device="cpu") as f:
                                for k in f.keys():
                                    state_dict[k] = f.get_tensor(k)
                        else:
                            state_dict = torch.load(ckpt)

                        # convert all tensors in state_dict to self.dtype
                        for k, v in state_dict.items():
                            state_dict[k] = v.to(self.dtype).to("cpu")

                        self._offload_state_dict(state_dict, empty_state_dict)

                        # print("Loading ckpt file", ckpt, flush=True)

                        del state_dict
                        gc.collect()
                        torch.cuda.empty_cache()

                    with open(name_id_map_file, "w") as f:
                        json.dump(self.name_id_map, f)
                else:
                    print("Loading model from offload_path ...", flush=True)
                    self.cls.__init__ = self.cls._old_init
                    # load the name_id_map
                    with open(name_id_map_file, "r") as f:
                        self.name_id_map = json.load(f)

                # print(self.name_id_map, flush=True)

                # get max tensor id from the name_id_map
                max_tensor_id = max(self.name_id_map.values())
                self.model_create_counter = tqdm(
                    total=max_tensor_id, desc="Model create"
                )

                is_flash_attn_available = kwargs.get(
                    "is_flash_attn_available", False)
                if self.dtype_cls is torch.bfloat16 or self.dtype_cls is torch.float16:
                    model = cls._from_config(
                        self.config,
                        torch_dtype=self.dtype_cls,
                        attn_implementation=(
                            "flash_attention_2" if is_flash_attn_available else "eager"
                        ),
                    )
                else:
                    model = cls._from_config(
                        self.config,
                    )

                base_model_prefix = model.base_model_prefix
                # model = model.to(self.dtype).to("cpu")

                # print("Model created with dtype", self.dtype, flush=True)
                # for name, param in model.named_parameters(recurse=False):
                #     print(name, param.dtype, flush=True)

                # print(self.config, flush=True)

                if hasattr(self.config, "quantization_config"):
                    self.quant_method = self.config.quantization_config["quant_method"]
                    self.config.quantization_config["use_exllama"] = False
                    self.config.quantization_config["disable_exllama"] = True
                    # print("Quantizing model ...", self.quant_method, flush=True)
                    if self.quant_method == "gptq":
                        from optimum.gptq import GPTQQuantizer
                        # print("Quantizing model with GPTQ ...", self.config.quantization_config, flush=True)
                        optimum_quantizer = GPTQQuantizer.from_dict(
                            self.config.quantization_config
                        )

                        model = optimum_quantizer.convert_model(model)

                self.expert_prefetcher = ExpertPrefetcher(
                    self.config, self.device)
                self.expert_prefetcher.set_archer_engine(self.archer_engine)
                self.expert_dispatcher = self.prefetch_lib.expert_dispatcher(
                    self.num_experts,
                    self.num_layers,
                    self.dtype,
                    parse_expert_type(self.config),
                )

                for name, param in model.named_parameters(recurse=True):
                    # remove base_model_prefix from self.name_id_map
                    if name.startswith(base_model_prefix):
                        name_without_prefix = name[(
                            len(base_model_prefix) + 1):]
                        if name_without_prefix in self.name_id_map:
                            self.name_id_map[name] = self.name_id_map[
                                name_without_prefix
                            ]
                            self.name_id_map.pop(name_without_prefix)
                    param.ar_id = self.name_id_map.get(name, None)

                if not "lm_head.weight" in self.name_id_map:
                    print("lm_head.weight not in name_id_map, add it as embed_tokens")
                    self.name_id_map["lm_head.weight"] = 0
                    self.name_id_map["encoder.embed_tokens.weight"] = 0
                    self.name_id_map["decoder.embed_tokens.weight"] = 0

                    model.lm_head.weight.ar_id = 0
                    model.model.encoder.embed_tokens.weight.ar_id = 0
                    model.model.decoder.embed_tokens.weight.ar_id = 0

                self.expert_tensor_map = dict()
                for name, id in self.name_id_map.items():
                    layer_id, expert_id = parse_expert_id(name, self.config)
                    if expert_id is not None:
                        self.expert_tensor_map[(layer_id, expert_id)] = id

                self.expert_prefetcher.expert_tensor_map = self.expert_tensor_map

                self.init_expert_map_matcher()

                model.model.expert_prefetcher = self.expert_prefetcher
                model.model.expert_tracer = self.expert_tracer
                model.model.expert_map_matcher = self.expert_map_matcher
                model.model._device = self.device

                module_idx = 0
                self.expert_layer_modules = []
                for module in model.modules():
                    if isinstance(module, SyncQwen2MoeSparseMoeBlock):
                        # module.archer_prefetch = self.archer_prefetch
                        # module.archer_tracer = self.archer_tracer
                        module.archer_engine = self.archer_engine
                        module.archer_config = self.archer_config
                        # module.expert_dispatcher = self.expert_dispatcher
                        self.expert_modules.append(module)
                        # module.expert_executor = self.expert_executor
                        module.expert_prefetcher = self.expert_prefetcher
                        module.expert_tracer = self.expert_tracer
                        module.expert_map_matcher = self.expert_map_matcher
                        module.expert_tensor_map = self.expert_tensor_map
                        module.prefetch_distance = self.prefetch_distance
                        module.device = self.device

                        self.expert_layer_modules.append(module)

                        module.layer_id = module_idx

                        module_idx += 1

                        self.moe_layers.append(module)
                        module.moe_layers = self.moe_layers

                    if isinstance(module, Qwen2MoeMLP):
                        module.offload_engine = self

                self.setup_archer_hooks(model)
                # print("OffloadEngine init done, rank", dist.get_rank(), flush=True)
                return model

            return archer_from_pretrained

        self.cls._old_from_pretrained = self.cls.from_pretrained
        self.cls.from_pretrained = classmethod(
            from_pretrained_decorator(self.cls.from_pretrained)
        )

        return self

    # clean up initialization hooks
    def __exit__(self, exc_type, exc_value, traceback):

        # GPTQ Override
        # QuantLinear.__init__ = QuantLinear._old_init
        # QuantLinearOld.__init__ = QuantLinearOld._old_init

        self.cls.__init__ = self.cls._old_init
        self.cls.from_pretrained = self.cls._old_from_pretrained
        torch.nn.modules.module.Module.apply = torch.nn.modules.module.Module._old_apply
        torch.index_select = torch._old_index_select
        torch.Tensor.index_select = torch.Tensor._old_index_select

        self.cls.post_init = self.cls._old_post_init
        PreTrainedModel.post_init = PreTrainedModel._old_post_init

        for name, module in torch.nn.modules.__dict__.items():
            if not isinstance(module, type):
                continue
            if not issubclass(module, torch.nn.modules.module.Module):
                continue
            if name in [
                "Module",
                "Sequential",
                "ModuleDict",
                "ModuleList",
                "ParameterList",
                "ParameterDict",
            ]:
                continue
            module.__init__ = module._old_init

            if hasattr(module, "reset_parameters"):
                module.reset_parameters = module._old_reset_parameters

    def get_topology(self, model):
        name_lst = []
        ret_dict = {}

        # print("Getting topology ...", self.name_id_map)

        # for name in model.state_dict().keys():
        for name, _ in model.named_parameters(recurse=True):
            match = re.search(r"\d+", name)
            if name not in self.name_id_map:
                print("param not in self.name_id_map", name)
                continue
            if match:
                if "expert" in name and "shared_expert" not in name:
                    match = re.match(r"(.*experts)", name)
                    assert match, "Not correct expert name!"
                    stored_name = match.group(1)
                    components = name.split(".")
                    # Use negative indexing to get the component between the last third and second dot
                    expert_name = components[-3]
                    if stored_name in name_lst:
                        if expert_name in ret_dict[stored_name]:
                            ret_dict[stored_name][expert_name].append(
                                self.name_id_map[name]
                            )
                        else:
                            ret_dict[stored_name][expert_name] = [
                                self.name_id_map[name]
                            ]
                    else:
                        ret_dict[stored_name] = {
                            expert_name: [self.name_id_map[name]]}
                        name_lst.append(stored_name)

                else:
                    match = re.match(r"(.*\.\d+\.)", name)
                    last_number_position = match.end() - 2
                    stored_name = name[: last_number_position + 1]

                    if stored_name in name_lst:
                        ret_dict[stored_name][0].append(self.name_id_map[name])
                    else:
                        ret_dict[stored_name] = [[self.name_id_map[name]]]
                        name_lst.append(stored_name)

            else:
                components = name.rsplit(".", 1)
                stored_name = components[0]

                if stored_name in name_lst:
                    ret_dict[stored_name][0].append(self.name_id_map[name])
                else:
                    ret_dict[stored_name] = [[self.name_id_map[name]]]
                    name_lst.append(stored_name)

        for name, _ in model.named_buffers(recurse=True):
            match = re.search(r"\d+", name)
            if name not in self.name_id_map:
                # print("buffer not in self.name_id_map", name)
                continue
            if match:
                if "expert" in name and "shared_expert" not in name:
                    match = re.match(r"(.*experts)", name)
                    assert match, "Not correct expert name!"
                    stored_name = match.group(1)
                    components = name.split(".")
                    # Use negative indexing to get the component between the last third and second dot
                    expert_name = components[-3]
                    if stored_name in name_lst:
                        if expert_name in ret_dict[stored_name]:
                            ret_dict[stored_name][expert_name].append(
                                self.name_id_map[name]
                            )
                        else:
                            ret_dict[stored_name][expert_name] = [
                                self.name_id_map[name]
                            ]
                    else:
                        ret_dict[stored_name] = {
                            expert_name: [self.name_id_map[name]]}
                        name_lst.append(stored_name)

                else:
                    matches = [match for match in re.finditer(r"\d", name)]
                    last_number_position = matches[-1].start() if matches else -1
                    stored_name = name[: last_number_position + 1]

                    if stored_name in name_lst:
                        ret_dict[stored_name][0].append(self.name_id_map[name])
                    else:
                        ret_dict[stored_name] = [[self.name_id_map[name]]]
                        name_lst.append(stored_name)
            else:
                components = name.rsplit(".", 1)
                stored_name = components[0]

                if stored_name in name_lst:
                    ret_dict[stored_name][0].append(self.name_id_map[name])
                else:
                    ret_dict[stored_name] = [[self.name_id_map[name]]]
                    name_lst.append(stored_name)

        for i in ret_dict.keys():
            if isinstance(ret_dict[i], dict):
                ret_dict[i] = list(ret_dict[i].values())

        topology = list(ret_dict.items())
        return topology

    def setup_archer_hooks(self, model):
        for name, param in model.named_parameters(recurse=True):
            if name not in self.name_id_map:
                continue
            self.archer_engine.register(param.data, self.name_id_map[name])
            self.offload_set.add(param.data.data_ptr())

            if "shared" in name:
                self.offload_exemption.add(param.data.data_ptr())

        for name, buffer in model.named_buffers(recurse=True):
            if name not in self.name_id_map:
                continue
            self.archer_engine.register(buffer.data, self.name_id_map[name])
            self.offload_set.add(buffer.data.data_ptr())

        topo = self.get_topology(model)
        self.archer_engine.set_topology(topo)

        @torch.no_grad()
        def _pre_forward_input_hook(module, input, kwargs, device, tensors):
            # print("pre_forward_input_hook", device, input, tensors)
            self.archer_engine.fetch_tensors(self.request_id, tensors)
            new_args = copy_args_to_device(device, input)
            new_kwargs = copy_kwargs_to_device(device, kwargs)
            return new_args, new_kwargs

        @torch.no_grad()
        def _post_forward_output_hook(module, input, output, device, tensors):
            if isinstance(output, tuple):
                new_args = copy_args_to_device(device, output)
            elif isinstance(output, dict):
                new_args = copy_kwargs_to_device(device, output)
            else:
                new_args = output.to(device)
            return new_args

        def gen_args_hook(key, input_device_index, output_device_index, tensors):

            keys = key.split(".")
            # print(keys)
            m = model
            for k in keys:
                if k.isdigit():
                    m = m[int(k)]
                else:
                    m = getattr(m, k)

            m.register_forward_pre_hook(
                functools.partial(
                    _pre_forward_input_hook, device=input_device_index, tensors=tensors
                ),
                prepend=True,
                with_kwargs=True,
            )
            if "lm_head" in key:
                m.register_forward_hook(
                    functools.partial(
                        _post_forward_output_hook, device=self.device, tensors=tensors
                    ),
                    prepend=False,
                )

        expert_layer_id = 0
        output_device_index = None
        for key, tensors in topo:
            # print(key, tensors)
            if "shared" in key or "lm_head" in key:
                key = key.split(".")[0]
                output_device_index = 0

            if "expert" in key:
                for expert_idx, expert_tensors in enumerate(tensors):
                    expert_key = (
                        f"{key}.expert_{expert_idx}"
                        if self.config.model_type != "qwen2_moe"
                        else f"{key}.{expert_idx}"
                    )
                    input_device_index = self.archer_engine.get_node_default_device(
                        expert_tensors
                    )
                    gen_args_hook(
                        expert_key,
                        input_device_index,
                        output_device_index,
                        expert_tensors,
                    )

                    self.expert_dispatcher.register_expert(
                        expert_layer_id, expert_idx, expert_tensors
                    )
                expert_layer_id += 1
            else:
                input_device_index = self.archer_engine.get_node_default_device(
                    tensors[0]
                )
                gen_args_hook(key, input_device_index,
                              output_device_index, tensors[0])
                output_device_index = input_device_index

        # @torch.no_grad()
        # def request_id_hook(module, *args):
        #     self.request_id_flag = False
        #     # self.archer_tracer.clear_request_id()
        #     # self.archer_prefetch.clear_request()

        # model.register_forward_hook(request_id_hook)

        # likely one of them should be enough but just to be safe
        self._register_hooks_recursively(model)

    def _generate_param_id(self):
        param_id = self.param_id
        self.param_id += 1
        return param_id

    def _generate_request_id(self):
        request_id = self.request_id
        self.request_id += 1
        return request_id

    def _offload_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        empty_state_dict: Dict[str, torch.Tensor],
    ) -> None:
        param_names = list(state_dict.keys())

        for param_name in param_names:
            self.name_id_map[param_name] = self._generate_param_id()
            if not self.archer_engine.is_tensor_offloaded(self.name_id_map[param_name]):
                self.archer_engine.offload(
                    state_dict[param_name], self.name_id_map[param_name]
                )

        gc.collect()
        torch.cuda.empty_cache()

    def _register_hooks_recursively(self, module, count=[0]):
        my_count = count[0]
        module.id = my_count

        for child in module.children():
            count[0] = count[0] + 1
            self._register_hooks_recursively(child, count=count)

        @torch.no_grad()
        def _pre_forward_module_hook(module, args, kwargs):
            # if self.request_id_flag == False:
            #     self.request_id_flag = True
            #     # print(kwargs, args, type(module))

            #     request_id = self._generate_request_id()
            #     # self.archer_tracer.set_request_id(request_id)
            #     # self.archer_prefetch.set_request(request_id)

            device_list = []

            for name, param in module.named_parameters(recurse=False):
                if not param.data.data_ptr() in self.offload_set:
                    num_devices = torch.cuda.device_count()
                    param.data = param.data.to(f"cuda:{num_devices-1}")
                    continue

                self.offload_set.remove(param.data.data_ptr())
                self.archer_engine.begin(self.request_id, param)
                self.offload_set.add(param.data.data_ptr())

                device_list.append(param.data.device)

            for name, buf in module.named_buffers(recurse=False):

                if not buf.data.data_ptr() in self.offload_set:
                    buf.data = buf.data.to(self.device)
                    continue

                # print("offload buffer", name, buf.data.data_ptr())

                self.offload_set.remove(buf.data_ptr())
                self.archer_engine.begin(self.request_id, buf)
                # buf = buf.to(self.dtype)
                self.offload_set.add(buf.data_ptr())

                device_list.append(buf.data.device)

        @torch.no_grad()
        def _post_forward_module_hook(module, input, output):
            device_list = []
            param_not_offload = set()
            for param in module.parameters(recurse=False):

                if not param.data.data_ptr() in self.offload_set:
                    param_not_offload.add(param.data.data_ptr())
                    continue

                self.offload_set.remove(param.data.data_ptr())
                self.archer_engine.end(self.request_id, param)
                self.offload_set.add(param.data.data_ptr())

                device_list.append(param.data.device)

            for buf in module.buffers(recurse=False):

                if not buf.data_ptr() in self.offload_set:
                    continue

                self.offload_set.remove(buf.data_ptr())
                self.archer_engine.end(self.request_id, buf)
                self.offload_set.add(buf.data_ptr())

                device_list.append(buf.device)

            if param_not_offload:
                if isinstance(output, torch.Tensor):
                    return output.to(torch.device(self.device))

                return copy_args_to_device(torch.device(self.device), *output)

        # Pre forward hook
        self.forward_hooks.append(
            module.register_forward_pre_hook(
                _pre_forward_module_hook, with_kwargs=True)
        )

        # Post forward hook
        self.forward_hooks.append(
            module.register_forward_hook(_post_forward_module_hook)
        )

    # clean runtime hooks
    def clean_up(self):
        finemoe.models.modeling_mixtral.SyncQwen2MoeSparseMoeBlock = (
            finemoe.models.modeling_mixtral._old_sparse_mlp
        )
