import warnings
import sys
import os
from finemoe import MoE
import transformers
import json
import pickle
import random as rd
from configs.common.config_common import (
    device,
    state_path,
    offload_path,
    device_memory_ratio,
    eval_sample_size,
)
from configs.models.config_qwen import (
    model_path,
    prefetch_distance,
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
    dataset_path,
    sample_size,
    moe_name,
    dataset_name,
    max_length,
    max_new_tokens,
    min_new_tokens,
):
    input_ids_list = []
    output_ids_list = []

    input_ids, output_ids = inference(
        model=model,
        tokenizer=tokenizer,
        generate_config=generate_config,
        prompts=prompts,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )
    input_ids_list.extend(input_ids)
    output_ids_list.extend(output_ids)

    print(
        f"\nGenerated states for model {moe_name}, dataset {dataset_name}, sample_size {sample_size}\n")

    # Simulation
    traj_dict = {}

    for seq_id, prompt, input_ids, output_ids in zip(model.engine.expert_tracer.trace, prompts, input_ids_list, output_ids_list):
        traj_dict[seq_id] = {}
        traj_dict[seq_id]['matrix'] = model.engine.expert_tracer.trace[seq_id].matrix
        traj_dict[seq_id]['iters'] = model.engine.expert_tracer.trace[seq_id].iters

    with open(f"{state_path}/{moe_name}~{dataset_name}~{sample_size}.pkl", 'wb') as f:
        pickle.dump(traj_dict, f)

    print(
        f"\nStates saved for model {moe_name}, dataset {dataset_name}, sample_size {sample_size}\n")

    # Store data
    model.engine.expert_tracer.expert_map_store.export_store_data(
        f"{state_path}/{moe_name}~{dataset_name}"
    )

    print(
        f"\nStore data saved for model {moe_name}, dataset {dataset_name}, sample_size {sample_size}\n")


if __name__ == '__main__':
    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")

    print(f'\nPreparing... model: {model_path}, dataset: {dataset_path}\n')

    moe_name = model_path.split('/')[-1]
    dataset_name = dataset_path.split('/')[-1]

    # Load datasets
    with open(f"{state_path}/{dataset_name}~eval_prompts.json", "r") as f:
        prompt_json = json.load(f)
    prompts = [p["prompt"] for p in prompt_json]
    prompts = rd.sample(prompts, eval_sample_size)

    # Initialize MoE
    model = MoE(
        model_path,
        {
            "offload_path": os.path.join(offload_path, moe_name),
            "device_memory_ratio": device_memory_ratio,
            "prefetch_distance": prefetch_distance,
            "store_capacity": None,
            "device": device,
            "eval_batch_size": eval_sample_size,
            "eval_max_length": max_length,
            "eval_mode": "online",
        }
    )

    # Initialize tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        device=device,
        clean_up_tokenization_spaces=True,
        trust_remote_code=True,
        padding_side='left',
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
        dataset_path=dataset_path,
        sample_size=eval_sample_size,
        moe_name=moe_name,
        dataset_name=dataset_name,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")
