import warnings
import sys
import os
import json
import transformers
from finemoe import MoE
import random as rd
from configs.common.config_common import (
    offload_path,
    state_path,
    eval_batch_size,
    device_memory_ratio,
    device,
)
from configs.models.config_qwen import (
    model_path,
    prefetch_distance,
    store_capacity,
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
        f"Running model: {moe_name}, dataset: {dataset_name}")

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")

    inference(
        model=model,
        tokenizer=tokenizer,
        generate_config=generate_config,
        prompts=prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")

    print(
        f"Complete model: {moe_name}, dataset: {dataset_name}")

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")


if __name__ == "__main__":
    eval_mode = "offline"

    print(
        f"\nEvaluating... model: {model_path}, dataset: {dataset_path}s\n")

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
            "prefetch_distance": prefetch_distance,
            "store_capacity": store_capacity,
            "device": device,
            "eval_batch_size": eval_batch_size,
            "eval_max_length": max_length,
            "eval_mode": eval_mode,
        }
    )

    if eval_mode == "offline":
        model.engine.expert_tracer.expert_map_store.import_store_data(
            f"{state_path}/{moe_name}~{dataset_name}"
        )

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
    if "qwen" in moe_name.lower():
        generate_config = {
            "pad_token_id": tokenizer.pad_token_id}
    else:
        raise ValueError(f"Model {moe_name} not supported")

    main(
        model=model,
        tokenizer=tokenizer,
        generate_config=generate_config,
        prompts=prompts,
        moe_name=moe_name,
        dataset_name=dataset_name,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )
