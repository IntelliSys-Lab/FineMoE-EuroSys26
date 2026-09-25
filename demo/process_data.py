import pickle
import numpy as np
import torch
from demo.configs.common.config_common import (
    result_path,
    state_path,
    eval_sample_size,
)
from demo.configs.models.config_qwen import (
    model_path,
)
from demo.configs.datasets.config_lmsys import (
    dataset_path,
)


def compute_heatmap(state_dict, moe_name, dataset_name):
    example = next(iter(state_dict.values()))

    np.savetxt(
        f"{result_path}/heatmap~coarse~{moe_name}~{dataset_name}.csv",
        example["matrix"].detach().cpu().numpy(), delimiter=",", fmt='%d')

    for i, it in enumerate(example["iters"]):
        if i < 3:
            np.savetxt(
                f"{result_path}/heatmap~fine~{i}~{moe_name}~{dataset_name}.csv",
                it["nodes"].detach().cpu().numpy(), delimiter=",", fmt='%d')


@torch.inference_mode()
def compute_entropy(state_dict, moe_name, dataset_name):
    eps = 1e-12
    coarse, fine, steps = [], [], []

    for entry in state_dict.values():
        C = entry['matrix'].to(torch.float32)
        coarse.append(C)

        nodes_list = [it['nodes'] for it in entry['iters']]
        if not nodes_list:
            continue
        S = torch.stack(nodes_list, 0).to(torch.float32)
        fine.append(S)

        cum = S.cumsum(0)
        P = (cum / cum.sum(-1, keepdim=True).clamp_min(eps)).clamp_min(eps)
        H = -(P * P.log()).sum(-1).mean(-1)
        steps.append(H)

    # Entropy per sequence and layer.
    if coarse:
        E = coarse[0].size(-1)
        C_all = torch.stack(coarse, 0).reshape(-1, E)
        Pc = (C_all / C_all.sum(-1, keepdim=True).clamp_min(eps)).clamp_min(eps)
        coarse_H = (-(Pc * Pc.log()).sum(-1))
    else:
        coarse_H = torch.empty(0)

    # Entropy per step and layer.
    if fine:
        E = fine[0].size(-1)
        F_all = torch.cat(fine, 0).reshape(-1, E)
        Pf = (F_all / F_all.sum(-1, keepdim=True).clamp_min(eps)).clamp_min(eps)
        fine_H = (-(Pf * Pf.log()).sum(-1))
    else:
        fine_H = torch.empty(0)

    # Align sequences to the shortest trace.
    if steps:
        m = min(s.numel() for s in steps)
        steps_H = torch.stack([s[:m] for s in steps], 0).mean(0)
    else:
        steps_H = torch.empty(0)

    np.savetxt(f"{result_path}/entropy~coarse~{moe_name}~{dataset_name}.csv",
               coarse_H.cpu().numpy(), delimiter=",", fmt="%.6f")
    np.savetxt(f"{result_path}/entropy~fine~{moe_name}~{dataset_name}.csv",
               fine_H.cpu().numpy(), delimiter=",", fmt="%.6f")
    np.savetxt(f"{result_path}/entropy~steps~{moe_name}~{dataset_name}.csv",
               steps_H.cpu().numpy(), delimiter=",", fmt="%.6f")


if __name__ == '__main__':
    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")

    print(
        f'\nProcessing states: model {model_path}, dataset {dataset_path}, sample_size {eval_sample_size}\n')

    moe_name = model_path.split('/')[-1]
    dataset_name = dataset_path.split('/')[-1]

    with open(f"{state_path}/{moe_name}~{dataset_name}~{eval_sample_size}.pkl", 'rb') as f:
        state_dict = pickle.load(f)

    compute_heatmap(
        state_dict=state_dict,
        moe_name=moe_name,
        dataset_name=dataset_name,
    )

    compute_entropy(
        state_dict=state_dict,
        moe_name=moe_name,
        dataset_name=dataset_name,
    )

    print("")
    print("**********")
    print("**********")
    print("**********")
