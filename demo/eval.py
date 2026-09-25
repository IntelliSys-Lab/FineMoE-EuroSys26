"""Evaluate the configured prompt batch with TTFT, TPOT, and expert hit rate."""
import json
import random
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from finemoe import MoE
from demo.configs.common.config_common import (
    devices, state_path, result_path, device_memory_ratio, offload_path, eval_batch_size,
)
from demo.configs.models.config_qwen import model_path, model_revision, prefetch_distance, store_capacity
from demo.configs.datasets.config_lmsys import dataset_path, max_length, max_new_tokens


@torch.inference_mode()
def measure(model, input_ids, attention_mask, new_tokens):
    if new_tokens < 2:
        raise ValueError("at least two output tokens are needed to measure decode")
    device = input_ids.device
    torch.cuda.synchronize(device)
    before = model.engine.cache.stats
    before_hits, before_misses = before.hits, before.misses
    start = time.perf_counter()
    with model.request(input_ids.shape[0]) as network:
        output = network(input_ids, attention_mask=attention_mask, past_key_values=DynamicCache(config=network.config),
                         use_cache=True, logits_to_keep=1)
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        past = output.past_key_values
        torch.cuda.synchronize(device)
        first = time.perf_counter()
        for _ in range(new_tokens - 1):
            attention_mask = torch.cat((attention_mask, torch.ones_like(token)), dim=1)
            output = network(token, attention_mask=attention_mask, past_key_values=past,
                             use_cache=True, logits_to_keep=1)
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            past = output.past_key_values
        torch.cuda.synchronize(device)
        end = time.perf_counter()
    after = model.engine.cache.stats
    hits, misses = after.hits - before_hits, after.misses - before_misses
    accesses = hits + misses
    return dict(
        ttft_ms=(first - start) * 1000,
        tpot_ms=(end - first) * 1000 / (new_tokens - 1),
        expert_hit_rate=hits / accesses if accesses else None,
    )


def main():
    torch.set_num_threads(1)
    dataset, name = dataset_path.split("/")[-1], model_path.split("/")[-1]
    records = json.loads((Path(state_path) / f"{dataset}~eval_prompts.json").read_text())
    prompts = random.Random(42).sample([record["prompt"] for record in records], eval_batch_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=model_revision, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    with MoE.from_prepared(Path(offload_path) / name / "prepared.safetensors",
                          dict(devices=devices, device_memory_ratio=device_memory_ratio,
                               prefetch_distance=prefetch_distance,
                               store_capacity=store_capacity)) as model:
        model.engine.expert_map_store.import_store_data(Path(state_path) / f"{name}~{dataset}")
        model.warmup()
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_length).to(model.engine.device)
        result = measure(model, inputs.input_ids, inputs.attention_mask, max_new_tokens)
        print(json.dumps(result), flush=True)
    out = Path(result_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"serving~{name}~{dataset}.json").write_text(json.dumps([result], indent=2) + "\n")


if __name__ == "__main__":
    main()
