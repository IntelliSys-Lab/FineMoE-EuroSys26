import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import glob
import os
import sys
import csv
from configs.common.config_common import (
    figure_path,
    result_path,
)
sys.path.append("../")

# =============================================================
# Figure 10: Eval latency bars (TTFT / TPOT / Hit Rate) per baseline
# =============================================================
# CSV header expected (case-insensitive): model,dataset,ttft,tpot,hit_rate

# ------------------ Paths ------------------
base = Path(result_path)
outdir = Path(figure_path)
outdir.mkdir(parents=True, exist_ok=True)

# ------------------ Patterns ------------------
# Allow any non-tilde segment for each of the three fields
# and be tolerant of hyphens/dots inside each segment.
eval_pat = re.compile(
    r"^eval~([^~]+)~([^~]+)~([^~]+)\.csv$", re.IGNORECASE)

# ------------------ Helpers ------------------


def read_eval_csv(path: str):
    """Return (ttft, tpot, hit_rate) as floats or (None, None, None)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return None, None, None
            # Skip header if present
            header = ",".join([c.strip().lower() for c in rows[0]])
            start_idx = 1 if header.startswith(
                "model,dataset,ttft,tpot") else 0
            for r in rows[start_idx:]:
                if len(r) < 5:
                    continue
                try:
                    ttft = float(r[2].strip())
                    tpot = float(r[3].strip())
                    hit_rate = float(r[4].strip())
                    return ttft, tpot, hit_rate
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return None, None, None


# Collect baselines
baselines = []
vals_ttft = []
vals_tpot = []
vals_hit = []

for path in glob.glob(str(base / "eval~*.csv")):
    fname = os.path.basename(path).strip()
    m = eval_pat.fullmatch(fname)
    if not m:
        print("Skipping non-matching file:", repr(fname))
        continue
    baseline, model_off, dataset_off = m.group(1), m.group(2), m.group(3)
    ttft, tpot, hit_rate = read_eval_csv(path)
    if None in (ttft, tpot, hit_rate):
        print("Missing values in:", repr(fname))
        continue
    baselines.append(baseline)
    vals_ttft.append(ttft)
    vals_tpot.append(tpot)
    vals_hit.append(hit_rate)

fig4_path = None

# If we have any baselines, draw the figure
if baselines:
    order = np.argsort(-np.array(vals_ttft))
    baselines = [baselines[i] for i in order]
    vals_ttft = [vals_ttft[i] for i in order]
    vals_tpot = [vals_tpot[i] for i in order]
    vals_hit = [vals_hit[i] for i in order]

    # Color/hatch mapping for recognizability
    style_map = {
        "FineMoE": {"color": "tab:orange", "hatch": "||"},
        "MoE-Infinity": {"color": "tab:purple", "hatch": ""},
        "ProMoE": {"color": "tab:green", "hatch": "//"},
        "Mixtral-Offloading": {"color": "tab:blue", "hatch": "///"},
    }
    colors = [style_map.get(b, {"color": "gray"})["color"] for b in baselines]
    hatches = [style_map.get(b, {"hatch": ""})["hatch"] for b in baselines]

    x = np.arange(len(baselines))
    width = 0.6

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # TTFT subplot
    bars1 = axes[0].bar(x, vals_ttft, width, edgecolor="black")
    for bar, hatch, c in zip(bars1, hatches, colors):
        bar.set_hatch(hatch)
        bar.set_color(c)
    axes[0].set_ylabel("TTFT (s)")
    axes[0].grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    # TPOT subplot
    bars2 = axes[1].bar(x, vals_tpot, width, edgecolor="black")
    for bar, hatch, c in zip(bars2, hatches, colors):
        bar.set_hatch(hatch)
        bar.set_color(c)
    axes[1].set_ylabel("TPOT (s)")
    axes[1].grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    # Hit Rate subplot
    bars3 = axes[2].bar(x, vals_hit, width, edgecolor="black")
    for bar, hatch, c in zip(bars3, hatches, colors):
        bar.set_hatch(hatch)
        bar.set_color(c)
    axes[2].set_ylabel("Expert Hit Rate")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    # X axis labels only on bottom
    axes[2].set_xticks(x, baselines)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=style_map.get(b, {"color": "gray"})["color"],
                            edgecolor="black", hatch=style_map.get(b, {"hatch": ""})["hatch"],
                            label=b) for b in baselines]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=min(len(baselines), 5), fontsize=12, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    fig4_path = str(outdir / "figure_10.png")
    fig.savefig(fig4_path, dpi=300)

if fig4_path:
    print("Saved latency experiments:", fig4_path)
else:
    print("No baselines parsed, nothing saved.")
