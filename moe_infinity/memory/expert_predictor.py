import torch
import torch.nn.functional as F
from transformers import PretrainedConfig
from moe_infinity.utils import parse_moe_param


class ExpertPredictor:
    def __init__(self, config: PretrainedConfig, prefetch_distance) -> None:
        self.num_layers, self.num_experts, self.num_encoder_layers, self.embed_dim, self.top_k = parse_moe_param(
            config)
        self.layer_decay_func = lambda x, l, L: -1 / (L + 1) * (x - l) + 1
        self.prefetch_distance = prefetch_distance
        self.tracer = None  # set by add_tracer

    def add_tracer(self, tracer):
        self.tracer = tracer

    @torch.inference_mode()
    def predict(self, seq_id, layer_idx):
        expert_matrix = self.tracer.get_entry(seq_id).matrix
        expert_matrix = self.tracer.find_most_similar(expert_matrix, layer_idx)

        expert_matrix = expert_matrix.to(
            self.tracer.device, dtype=self.tracer.dtype, copy=False)

        k = min(self.top_k, expert_matrix.size(-1))
        topk_vals, topk_idx = torch.topk(
            expert_matrix, k=k, dim=1)
        result = torch.zeros_like(expert_matrix).scatter_(
            1, topk_idx, topk_vals)

        layer_start = int(layer_idx)
        layer_end = self.num_layers
        if layer_start > 0:
            result[:layer_start].zero_()
        if layer_end < self.num_layers:
            result[layer_end:].zero_()

        rng = torch.arange(self.num_layers, device=result.device)
        band = (rng >= layer_start) & (rng < layer_end)
        decay = torch.ones(self.num_layers, device=result.device)
        decay[band] = -1 / (layer_end+1) * (rng[band]-layer_start) + 1
        result.mul_(decay.view(-1, 1))
        expert_prob_map = torch.zeros_like(result)

        return result, expert_prob_map
