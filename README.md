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

This repo contains a demo implementation of the paper, [Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26).

> Large Language Models (LLMs) have gained immense success in revolutionizing various applications, including content generation, search and recommendation, and AI-assisted operation. To reduce high training costs, Mixture-of-Experts (MoE) architecture has become a popular backbone for modern LLMs. However, despite the benefits, serving MoE-based LLMs experience severe memory inefficiency due to sparsely activated experts. Recent studies propose to offload inactive experts from GPU memory to CPU memory to improve the serving efficiency of MoE models. However, they either incur high inference latency or high model memory footprints due to coarse-grained designs.
To tame the latency-memory trade-off in MoE serving, we present FineMoE, a fine-grained expert offloading system for MoE serving that achieves low inference latency with memory efficiency. We design FineMoE to extract fine-grained expert selection patterns from MoE models and semantic hints from input prompts to efficiently guide expert prefetching, caching, and offloading decisions. FineMoE is prototyped on top of HuggingFace Transformers and deployed on a six-GPU testbed. Experiments with open-source MoE models and real-world workloads show that FineMoE reduces inference latency by 47% and improves expert hit rate by 39% over state-of-the-art solutions.

----

FineMoE is built on top of [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity). We genuinely thank the MoE-Infinity team for their fantastic codebase!

We describe how to build and run this demo.

## General Hardware Prerequisite

- Operating systems and versions: Ubuntu 22.04
- Resource requirement
  - GPU memory: >= 48 GB
  - CPU: >= 8 cores
  - Memory: >= 16 GB
  - Disk: >= 50 GB
  - Network: no requirement

## Demo Instructions

<a name="step-1"></a>

1. Download the GitHub repo.
```
git clone https://github.com/IntelliSys-Lab/FineMoE-EuroSys26
```

<a name="step-2"></a>

2. Set your [Huggingface user access token](https://huggingface.co/docs/hub/en/security-tokens) in the script [`setup.sh`](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26/tree/master/master/setup.sh) and install dependencies.
```
cd FineMoE-EuroSys26 && ./setup.sh
```

<a name="step-3"></a>

3. Go to [`demo`](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26/tree/master/demo).
```
cd demo
```

<a name="step-4"></a>

4. Prepare the model and gather data for the demo.
```
python prepare_data.py
```

<a name="step-5"></a>

5. Process the data before executing baselines.
```
python process_data.py
```

<a name="step-6"></a>

6. Execute the demo:
```
python eval.py
```

## Results and Figures

You should be able to check the results as CSV files under [`demo/results`](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26/tree/master/demo/results). We provide scripts to plot the results using Matplotlib:
```
python plot_entropy.py
```
Figures should be available under [`demo/figures`](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26/tree/master/demo/figures) after the script completes.

## Experimental Settings and Workloads

The experiment settings can be found in [`demo/configs`](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26/tree/master/demo/configs). 
This demo only shows serving [`Qwen1.5-MoE-A2.7B-Chat`](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B-Chat) on a small dataset sample of [`lmsys-chat-1m`](https://huggingface.co/datasets/lmsys/lmsys-chat-1m).
The dataset sample can be found in [`demo/states/lmsys-chat-1m~eval_prompts.json`](https://github.com/IntelliSys-Lab/FineMoE-EuroSys26/tree/master/demo/states/lmsys-chat-1m~eval_prompts.json)
