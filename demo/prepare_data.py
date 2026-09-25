"""Collect expert maps and traces from the configured dataset sample."""
import json
from pathlib import Path
import pickle
import random
import torch
from transformers import AutoTokenizer
from finemoe import MoE
from demo.configs.common.config_common import (
    devices, state_path, device_memory_ratio, offload_path, eval_sample_size,
)
from demo.configs.models.config_qwen import model_path, model_revision, prefetch_distance, store_capacity
from demo.configs.datasets.config_lmsys import dataset_path, max_length, max_new_tokens


def main():
    torch.set_num_threads(1)
    root = Path(state_path)
    root.mkdir(parents=True, exist_ok=True)
    dataset = dataset_path.split("/")[-1]
    name = model_path.split("/")[-1]
    records = json.loads((root / f"{dataset}~eval_prompts.json").read_text())
    prompts = random.Random(42).sample([record["prompt"] for record in records], eval_sample_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=model_revision)
    prepared = Path(offload_path) / name / "prepared.safetensors"
    options = dict(devices=devices, prefetch_distance=prefetch_distance,
                   store_capacity=store_capacity, device_memory_ratio=device_memory_ratio,
                   collect_maps=True, eval_mode="online", trace_capacity=len(prompts))
    with (MoE.from_prepared(prepared, options) if prepared.exists()
          else MoE(model_path, options, revision=model_revision)) as model:
        if not prepared.exists():
            model.save_prepared(prepared)
        model.warmup()
        for index, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            model.generate(inputs.input_ids, attention_mask=inputs.attention_mask,
                           max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
            print(f"Collected {index + 1}/{len(prompts)}", flush=True)
        if model.engine.predictor.dropped_updates:
            raise RuntimeError("map collection dropped updates; refusing to export an incomplete run")
        trace = {key: dict(matrix=entry.matrix, iters=entry.iters)
                 for key, entry in model.engine.expert_tracer.trace.items()}
        with (root / f"{name}~{dataset}~{len(prompts)}.pkl").open("wb") as handle:
            pickle.dump(trace, handle)
        model.engine.expert_map_store.export_store_data(root / f"{name}~{dataset}")


if __name__ == "__main__":
    main()
