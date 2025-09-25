import uuid
import torch
from transformers import PretrainedConfig
from finemoe.memory.expert_entry import ExpertTraceEntry
from finemoe.utils import parse_moe_param


class ExpertTracer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ExpertTracer, cls).__new__(cls)
        return cls._instance

    def __init__(self, capacity: int, config: PretrainedConfig, expert_map_store, eval_mode, device):
        self.num_layers, self.num_experts, self.num_encoder_layers, self.embed_dim, self.top_k = parse_moe_param(
            config)
        self.capacity = capacity

        self.expert_map_store = expert_map_store
        self.eval_mode = eval_mode
        self.device = torch.device(device)
        self.dtype = torch.float32

        self.trace = {}

        self.trace_collection = torch.zeros(
            (capacity, self.num_layers, self.num_experts),
            dtype=self.dtype, device=self.device
        )
        self.collection_access = torch.zeros(
            (capacity,), dtype=torch.int32, device="cpu")

    def create_entry(self):
        seq_id = uuid.uuid4().hex
        self.trace[seq_id] = ExpertTraceEntry(
            seq_id=seq_id,
            matrix=torch.zeros((self.num_layers, self.num_experts),
                               dtype=self.dtype, device=self.device),
            iters=[],
            num_new_tokens=0,
            num_prefill_tokens=0,
        )
        return seq_id

    @torch.inference_mode()
    def finish_entry(self, seq_id):
        trace_sum = self.trace_collection.abs().sum(dim=(1, 2))
        empty_mask = trace_sum == 0
        if torch.any(empty_mask):
            idx = int(torch.nonzero(empty_mask, as_tuple=False)[0].item())
            self.trace_collection[idx] = self.trace[seq_id].matrix
            self.collection_access[idx] = 1
        else:
            idx = int(torch.argmin(self.collection_access).item())
            self.trace_collection[idx] = self.trace[seq_id].matrix
            self.collection_access[idx] = self.collection_access[idx] + 1

    @torch.inference_mode()
    def update_embed(self, seq_id: int, embeds: torch.Tensor):
        trace_entry = self.trace[seq_id]
        if len(trace_entry.iters) == 0:
            for i in range(embeds.shape[0]):
                trace_entry.iters.append({
                    "stage": "prefill",
                    "embed": embeds[i].detach(),
                    "states": torch.zeros((self.num_layers, self.embed_dim),  dtype=self.dtype, device=self.device),
                    "nodes":  torch.zeros((self.num_layers, self.num_experts), dtype=self.dtype, device=self.device),
                    "probs":  torch.zeros((self.num_layers, self.num_experts), dtype=self.dtype, device=self.device),
                    "preds":  torch.zeros((self.num_layers, self.num_experts), dtype=self.dtype, device=self.device),
                })
        else:
            assert trace_entry.iters[-1]["embed"] is not None, "Something is wrong with decode steps!"
            trace_entry.iters.append({
                "stage": "decode",
                "embed": embeds.detach().squeeze(0),
                "states": torch.zeros((self.num_layers, self.embed_dim),  dtype=self.dtype, device=self.device),
                "nodes":  torch.zeros((self.num_layers, self.num_experts), dtype=self.dtype, device=self.device),
                "probs":  torch.zeros((self.num_layers, self.num_experts), dtype=self.dtype, device=self.device),
                "preds":  torch.zeros((self.num_layers, self.num_experts), dtype=self.dtype, device=self.device),
            })

    @torch.inference_mode()
    def _row_add_bincount(self, row_1d: torch.Tensor, idxs_2d: torch.Tensor):
        device = row_1d.device
        E = row_1d.numel()
        flat = idxs_2d.to(device=device, dtype=torch.long,
                          non_blocking=True).reshape(-1)
        inc = torch.bincount(flat, minlength=E).to(
            dtype=row_1d.dtype, device=device)
        row_1d.add_(inc)

    @torch.inference_mode()
    def _row_add_index_add(self, row_1d: torch.Tensor, idxs_1d: torch.Tensor):
        device = row_1d.device
        idxs_1d = idxs_1d.to(
            device=device, dtype=torch.long, non_blocking=True)
        ones = torch.ones_like(idxs_1d, dtype=row_1d.dtype, device=device)
        row_1d.index_add_(0, idxs_1d, ones)

    @torch.inference_mode()
    def update_entry(self, seq_id, expert_list: torch.Tensor, layer_idx: int,
                     hidden_states: torch.Tensor, expert_probs: torch.Tensor):
        expert_list = torch.as_tensor(expert_list, dtype=torch.long)
        expert_list = expert_list.to(self.device, non_blocking=True)

        hidden_states = torch.as_tensor(hidden_states, dtype=self.dtype)
        hidden_states = hidden_states.to(self.device, non_blocking=True)

        expert_probs = torch.as_tensor(expert_probs, dtype=self.dtype)
        expert_probs = expert_probs.to(self.device, non_blocking=True)

        trace_entry = self.trace[seq_id]
        num_tokens, _ = expert_list.shape

        completed_embeds = []
        completed_maps = []

        row_global = trace_entry.matrix[layer_idx]
        if num_tokens > 1:
            self._row_add_bincount(row_global, expert_list)
        else:
            self._row_add_index_add(row_global, expert_list[0])

        if num_tokens > 1:
            if trace_entry.num_prefill_tokens == 0:
                trace_entry.num_prefill_tokens = num_tokens

            for token_idx in range(num_tokens):
                it = trace_entry.iters[token_idx]
                states = it["states"]
                nodes = it["nodes"]
                probs = it["probs"]

                idxs = expert_list[token_idx]
                self._row_add_index_add(nodes[layer_idx], idxs)
                probs[layer_idx].copy_(expert_probs[token_idx])
                states[layer_idx].copy_(hidden_states[token_idx])

                if layer_idx == self.num_layers - 1:
                    completed_embeds.append(it["embed"])
                    completed_maps.append(probs)
        else:
            assert layer_idx < self.num_layers
            token_idx = trace_entry.num_prefill_tokens + trace_entry.num_new_tokens - 1
            it = trace_entry.iters[token_idx]
            states = it["states"]
            nodes = it["nodes"]
            probs = it["probs"]

            idxs = expert_list[0]
            self._row_add_index_add(nodes[layer_idx], idxs)
            probs[layer_idx].copy_(expert_probs[0])
            states[layer_idx].copy_(hidden_states[0])

            if layer_idx == self.num_layers - 1:
                completed_embeds.append(it["embed"])
                completed_maps.append(probs)

        if layer_idx == self.num_layers - 1:
            trace_entry.num_new_tokens += 1

        if self.eval_mode == "online" and completed_embeds and completed_maps:
            embeds_tensor = torch.stack(completed_embeds, dim=0)
            maps_tensor = torch.stack(completed_maps, dim=0)
            self.expert_map_store.add(
                embeds=embeds_tensor, expert_maps=maps_tensor)

    @torch.inference_mode()
    def update_preds(self, seq_id: int, iter_id: int, expert_preds: torch.Tensor, layer_start: int, layer_end: int):
        expert_preds = torch.as_tensor(
            expert_preds, dtype=self.dtype, device=self.device)
        trace_entry = self.trace[seq_id]
        it = trace_entry.iters[iter_id]
        preds = it["preds"]

        for layer_idx in range(layer_start, layer_end):
            assert layer_idx < self.num_layers, f"Invalid layer_idx {layer_idx}"
            layer_mask = expert_preds[layer_idx] != 0
            preds[layer_idx].zero_()
            preds[layer_idx][layer_mask] = 1

    def get_entry(self, seq_id):
        return self.trace[seq_id]
