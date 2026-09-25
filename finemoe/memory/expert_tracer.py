"""CPU traces recorded during map collection for entropy and heatmap outputs."""

from collections import OrderedDict
import uuid
import torch
from .expert_entry import ExpertTraceEntry


class ExpertTracer:
    def __init__(self, num_layers, num_experts, store, capacity=64):
        self.num_layers, self.num_experts = num_layers, num_experts
        self.expert_map_store = store
        self.capacity = capacity
        self.trace = OrderedDict()
        self.active = []

    def start_request(self, batch_size):
        if batch_size > self.capacity:
            raise ValueError("trace_capacity must be at least the request batch size")
        self.active = []
        for _ in range(batch_size):
            while len(self.trace) >= self.capacity:
                self.trace.popitem(last=False)
            seq_id = uuid.uuid4().hex
            self.trace[seq_id] = ExpertTraceEntry(seq_id, torch.zeros(self.num_layers, self.num_experts))
            self.active.append(seq_id)

    def begin_iteration(self, embeddings, mask, prefill):
        self.current = []
        for seq_id, embed, valid in zip(self.active, embeddings.cpu(), mask.cpu()):
            entry = self.trace[seq_id]
            indices = valid.nonzero().flatten().tolist()
            records = []
            for index in indices:
                record = {"stage": "prefill" if prefill else "decode", "embed": embed[index].clone(),
                          "probs": torch.zeros_like(entry.matrix),
                          "nodes": torch.zeros_like(entry.matrix)}
                entry.iters.append(record)
                records.append((index, record))
            if prefill:
                entry.num_prefill_tokens = len(indices)
            else:
                entry.num_new_tokens += 1
            self.current.append(records)

    def observe(self, layer, probabilities, selected):
        probabilities, selected = probabilities.cpu(), selected.cpu()
        for seq_id, records, probs, experts in zip(self.active, self.current, probabilities, selected):
            entry = self.trace[seq_id]
            for index, record in records:
                record["probs"][layer].copy_(probs[index])
                record["nodes"][layer].scatter_add_(0, experts[index], torch.ones(len(experts[index])))
                entry.matrix[layer].add_(record["nodes"][layer])
