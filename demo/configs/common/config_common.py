from pathlib import Path
DEMO = Path(__file__).resolve().parents[2]
devices = ["cuda:0"]
figure_path = str(DEMO / "figures")
result_path = str(DEMO / "results")
state_path = str(DEMO / "states")
offload_path = str(DEMO / "offloads")
device_memory_ratio = 0.8
eval_sample_size = 64
eval_batch_size = 1
