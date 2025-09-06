import warnings
import sys
import os
import json
import transformers
import torch
from moe_infinity import MoE
import polars as pl
import argparse
import random as rd
from configs.common.config_common import (
    offload_path,
    result_path,
    state_path,
    eval_batch_size,
    device_memory_ratio,
    device,
    cache_limit,
)
from configs.models.config_qwen import (
    model_path,
    prefetch_distance,
    expert_topk,
)
from configs.datasets.config_lmsys import (
    dataset_path,
    max_length,
    max_new_tokens,
    min_new_tokens,
)
rd.seed(42)
sys.path.append("../")
warnings.filterwarnings("ignore")


def inference(
    model,
    tokenizer,
    generate_config,
    prompts,
    max_length,
    max_new_tokens,
    min_new_tokens,
):
    inputs = tokenizer(
        prompts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    output_ids = model.generate(
        inputs.input_ids.to(device),
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        attention_mask=inputs.attention_mask,
        do_sample=False,
        **generate_config
    )

    return inputs.input_ids.detach().cpu(), output_ids.detach().cpu()


def main(
    model,
    tokenizer,
    generate_config,
    prompts,
    result_dict,
    moe_name,
    dataset_name,
    max_length,
    max_new_tokens,
    min_new_tokens,
):
    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")

    print(
        f"Running {offload_method}, model: {moe_name}, dataset: {dataset_name}")

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")

    input_ids, output_ids = inference(
        model=model,
        tokenizer=tokenizer,
        generate_config=generate_config,
        prompts=prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )

    timestamps = list(model.engine.timers.values())[0]
    output_len = output_ids.shape[1] - input_ids.shape[1]
    hits = 0
    total = 0
    for _, trace_entry in model.engine.expert_tracer.trace.items():
        for it in trace_entry.iters:
            nodes = it["nodes"].bool()
            preds = it["preds"].bool()
            hits += (nodes & preds).sum().item()
            total += nodes.sum().item()

    result_dict["model"].append(moe_name)
    result_dict["dataset"].append(dataset_name)
    result_dict["ttft"].append(timestamps[1] - timestamps[0])
    result_dict["tpot"].append((timestamps[-1] - timestamps[1]) / output_len)
    result_dict["hit_rate"].append(hits / total)

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")

    print(
        f"Complete {offload_method}, model: {moe_name}, dataset: {dataset_name}")

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offload", type=str, default="finemoe", help="Offload method")

    args = parser.parse_args()

    if args.offload == "deepspeed":
        from configs.baselines.config_deepspeed import (
            offload_method,
        )
        store_capacity = None
    elif args.offload == "moeinf":
        from configs.baselines.config_moeinf import (
            offload_method,
            store_capacity,
        )
    elif args.offload == "mixoff":
        from configs.baselines.config_mixoff import (
            offload_method,
        )
        store_capacity = None
    elif args.offload == "promoe":
        from configs.baselines.config_promoe import (
            offload_method,
        )
        store_capacity = None
    elif args.offload == "finemoe":
        from configs.baselines.config_finemoe import (
            offload_method,
            store_capacity,
        )
    else:
        raise ValueError(f"Offload method {args.offload} not supported")

    eval_mode = "offline"
    result_dict = {
        "model": [],
        "dataset": [],
        "ttft": [],
        "tpot": [],
        "hit_rate": [],
    }

    print(
        f"\nEvaluating... model: {model_path}, dataset: {dataset_path}, offload: {offload_method}\n")

    moe_name = model_path.split("/")[-1]
    dataset_name = dataset_path.split("/")[-1]

    # Load dataset
    with open(f"{state_path}/{dataset_name}~eval_prompts.json", "r") as f:
        prompt_json = json.load(f)
    prompts = [p["prompt"] for p in prompt_json]
    prompts = rd.sample(prompts, eval_batch_size)

    # Initialize MoE
    model = MoE(
        model_path,
        {
            "offload_path": os.path.join(offload_path, moe_name),
            "device_memory_ratio": device_memory_ratio,
            "offload_method": offload_method,
            "prefetch_distance": prefetch_distance,
            "store_capacity": store_capacity,
            "device": device,
            "eval_batch_size": eval_batch_size,
            "eval_max_length": max_length,
            "eval_mode": eval_mode,
        }
    )
    model.engine.archer_engine.set_cache_limit(
        int(cache_limit * 1024 * 1024 * 1024)
    )

    if eval_mode == "offline":
        if offload_method == "FineMoE":
            model.engine.expert_tracer.expert_map_store.import_store_data(
                f"{state_path}/{moe_name}~{dataset_name}"
            )
        elif offload_method == "MoE-Infinity":
            model.engine.expert_tracer.load_trace(
                f"{state_path}/{moe_name}~{dataset_name}"
            )
        else:
            pass

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        device=device,
        clean_up_tokenization_spaces=True,
        trust_remote_code=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    generate_config = {}
    if "mixtral" in moe_name.lower():
        generate_config = {
            "pad_token_id": tokenizer.pad_token_id}
    elif "qwen" in moe_name.lower():
        generate_config = {
            "pad_token_id": tokenizer.pad_token_id}
    elif "phi" in moe_name.lower():
        generate_config = {
            "pad_token_id": tokenizer.pad_token_id}
    else:
        raise ValueError(f"Model {moe_name} not supported")

    main(
        model=model,
        tokenizer=tokenizer,
        generate_config=generate_config,
        prompts=prompts,
        result_dict=result_dict,
        moe_name=moe_name,
        dataset_name=dataset_name,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )

    pl.DataFrame(result_dict).write_csv(
        f"{result_path}/{eval_mode}~{offload_method}~{moe_name}~{dataset_name}.csv"
    )
