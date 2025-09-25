# Copyright (c) TorchMoE.
# SPDX-License-Identifier: Apache-2.0

# TorchMoE Team

import torch
from transformers import PretrainedConfig
from finemoe.utils import parse_moe_param


class ExpertPrefetcher(object):
    cache_file_rd = None

    def __init__(self, config: PretrainedConfig, device):
        # print(config)
        self.num_layers, self.num_experts, self.num_encoder_layers, self.embed_dim, self.top_k = parse_moe_param(
            config)

        self.archer_engine = None
        self.expert_tensor_map = None
        # [L, E] -> tensor_id (int64), -1 if missing
        self._tensor_id_grid = None
        # used only if you pass GPU tensors in
        self.device = torch.device(device)

    def set_archer_engine(self, archer_engine):
        if archer_engine is None:
            raise ValueError(
                "archer_engine must not be None. Call set_archer_engine(...) with a valid engine.")
        global _expert_prefetcher
        _expert_prefetcher = archer_engine
        self.archer_engine = archer_engine

    def set_expert_tensor_map(self, expert_tensor_map: dict):
        self.expert_tensor_map = expert_tensor_map
        self._build_tensor_id_grid()

    def _build_tensor_id_grid(self):
        if self.expert_tensor_map is None:
            raise RuntimeError(
                "expert_tensor_map is not set; set ExpertPrefetcher.expert_tensor_map or call set_expert_tensor_map(...).")
        grid = torch.full((self.num_layers, self.num_experts), -
                          1, dtype=torch.long, device="cpu")
        if self.expert_tensor_map is None:
            raise RuntimeError(
                "expert_tensor_map is not set on ExpertPrefetcher.")
        for (layer_id, expert_id), tid in self.expert_tensor_map.items():
            if 0 <= layer_id < self.num_layers and 0 <= expert_id < self.num_experts:
                grid[layer_id, expert_id] = int(tid)
        self._tensor_id_grid = grid

    @torch.inference_mode()
    def prefetch_experts(self, prefetch_priority_map, expert_prob_map):
        if self.archer_engine is None:
            raise RuntimeError(
                "ExpertPrefetcher.archer_engine is None. Call set_archer_engine(...) before prefetch_experts().")

        if not isinstance(prefetch_priority_map, torch.Tensor):
            prefetch_priority_map = torch.as_tensor(prefetch_priority_map)
        if not isinstance(expert_prob_map, torch.Tensor):
            expert_prob_map = torch.as_tensor(expert_prob_map)

        pp = prefetch_priority_map.detach().to(
            "cpu", dtype=torch.float32, non_blocking=False)
        ep = expert_prob_map.detach().to(
            "cpu", dtype=torch.float32, non_blocking=False)

        if self._tensor_id_grid is None:
            if getattr(self, "expert_tensor_map", None) is None:
                raise RuntimeError(
                    "expert_tensor_map not set; cannot map (layer, expert) to tensor ids.")
            self._build_tensor_id_grid()

        mask = pp > 0
        if not mask.any():
            return

        rows, cols = mask.nonzero(as_tuple=True)
        priors = pp[rows, cols]
        probs = ep[rows, cols]
        tids = self._tensor_id_grid[rows, cols]

        valid = tids >= 0
        if not valid.any():
            return

        tids = tids[valid]
        priors = priors[valid]
        probs = probs[valid]

        order = torch.argsort(priors, descending=True)
        tids = tids[order].tolist()
        probs = probs[order].tolist()

        self.archer_engine.replace_cache_candidates(tids)
        for tid, p in zip(tids, probs):
            gpu_id = self.archer_engine.get_node_default_device([tid])
            self.archer_engine.enqueue_prefetch(tid, gpu_id, float(p))
