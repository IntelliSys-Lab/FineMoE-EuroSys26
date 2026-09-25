<!--
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
-->

# FineMoE-EuroSys26

----

This repo contains a demo implementation of the paper, [Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading](https://doi.org/10.1145/3767295.3769319).

> Large Language Models (LLMs) have gained immense success in revolutionizing various applications, including content generation, search and recommendation, and AI-assisted operation. To reduce high training costs, Mixture-of-Experts (MoE) architecture has become a popular backbone for modern LLMs. However, despite the benefits, serving MoE-based LLMs experience severe memory inefficiency due to sparsely activated experts. Recent studies propose to offload inactive experts from GPU memory to CPU memory to improve the serving efficiency of MoE models. However, they either incur high inference latency or high model memory footprints due to coarse-grained designs.
To tame the latency-memory trade-off in MoE serving, we present FineMoE, a fine-grained expert offloading system for MoE serving that achieves low inference latency with memory efficiency. We design FineMoE to extract fine-grained expert selection patterns from MoE models and semantic hints from input prompts to efficiently guide expert prefetching, caching, and offloading decisions. FineMoE is prototyped on top of HuggingFace Transformers and deployed on a six-GPU testbed. Experiments with open-source MoE models and real-world workloads show that FineMoE reduces inference latency by 47% and improves expert hit rate by 39% over state-of-the-art solutions.

----

FineMoE is built on top of [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity). We thank the MoE-Infinity team for their codebase!

We describe how to build and run this demo.

## General Hardware Prerequisite

- Operating system: Linux x86-64 (tested on Ubuntu)
- Python: 3.10–3.12
- Build: C++17 compiler
- GPU: one or more CUDA GPUs; RTX 3090 with 24 GB memory supported
- CPU: >= 8 cores
- Host memory: 192 GB recommended for the checkpoint and pinned expert weights
- Disk: >= 160 GB for the original checkpoint and prepared weights
- Network: required to download dependencies and the checkpoint

## Demo Instructions

Run the commands below from the repository root.

<a name="step-1"></a>

1. Download the GitHub repo.
```bash
git clone https://github.com/IntelliSys-Lab/FineMoE-EuroSys26
cd FineMoE-EuroSys26
```

<a name="step-2"></a>

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and the locked CUDA dependencies.
```bash
./setup.sh
```

<a name="step-3"></a>

3. Select the GPUs in [`config_common.py`](demo/configs/common/config_common.py) and review the settings in [`demo/configs`](demo/configs).
```python
devices = ["cuda:0"]
```
For two GPUs, set `devices = ["cuda:0", "cuda:1"]`. Indices refer to GPUs visible to the process.

<a name="step-4"></a>

4. Prepare the model and gather data for the demo.
```bash
uv run --locked python -m demo.prepare_data
```

<a name="step-5"></a>

5. Process the collected data.
```bash
uv run --locked python -m demo.process_data
```

<a name="step-6"></a>

6. Execute the demo.
```bash
uv run --locked python -m demo.eval
```

## Results and Figures

The demo writes entropy and heatmap CSV files and serving metrics under [`demo/results`](demo/results). Plot the results with Matplotlib:
```bash
uv run --locked python -m demo.plot_entropy
```
Figures are saved under [`demo/figures`](demo/figures).

## Experimental Settings and Workloads

Experiment settings are in [`demo/configs`](demo/configs).
This demo serves [`Qwen3.5-35B-A3B`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) (text only) on a small sample of [`lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m).
The dataset sample is in [`demo/states/lmsys-chat-1m~eval_prompts.json`](demo/states/lmsys-chat-1m~eval_prompts.json).
