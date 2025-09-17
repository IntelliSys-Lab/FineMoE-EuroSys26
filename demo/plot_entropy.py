import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
import glob
import os
import sys
from matplotlib.patches import FancyBboxPatch
from configs.common.config_common import (
    figure_path,
    result_path,
)
sys.path.append("../")

# ------------------ Paths ------------------
base = Path(result_path)
outdir = Path(figure_path)
outdir.mkdir(parents=True, exist_ok=True)

# ------------------ Patterns ------------------
coarse_pat = re.compile(r"entropy~coarse~(.+?)~(.+?)\.csv$", re.IGNORECASE)
fine_pat = re.compile(r"entropy~fine~(.+?)~(.+?)\.csv$", re.IGNORECASE)
steps_pat = re.compile(r"entropy~steps~(.+?)~(.+?)\.csv$", re.IGNORECASE)

# heatmaps
h_coarse_pat = re.compile(r"heatmap~coarse~(.+?)~(.+?)\.csv$", re.IGNORECASE)
h_fine_pat = re.compile(r"heatmap~fine~(\d+)~(.+?)~(.+?)\.csv$", re.IGNORECASE)

# ------------------ Helpers ------------------


def read_floats(path: str) -> np.ndarray:
    """Read single-column numeric CSV into float array."""
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                vals.append(float(s.split(",")[0]))
            except ValueError:
                continue
    return np.array(vals, dtype=float) if vals else np.array([], dtype=float)


def read_matrix_csv(path: str) -> np.ndarray:
    """Read a 2D CSV (comma-separated) into a float matrix (rows = experts, cols = layers)."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split(",")
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                # skip junk rows
                continue
    return np.array(rows, dtype=float) if rows else np.zeros((0, 0), dtype=float)


def pick_iters(all_iters, k=4):
    """Pick k approximately evenly spaced iteration indices from sorted list."""
    if not all_iters:
        return []
    if len(all_iters) <= k:
        return sorted(all_iters)
    idx = np.linspace(0, len(all_iters) - 1, k).round().astype(int)
    return [sorted(all_iters)[i] for i in idx]


def draw_heatmap(ax, mat, vmin=None, vmax=None):
    im = ax.imshow(mat, aspect="auto", cmap="Reds",
                   vmin=vmin, vmax=vmax, origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# =============================================================
# Figure 3a: Heatmaps (coarse vs. fine panels)
# =============================================================
heat_coarse = {}   # {(dataset, model): matrix}
heat_fine = {}     # {(dataset, model): {iter: matrix}}

for path in glob.glob(str(base / "heatmap~*.csv")):
    fname = os.path.basename(path)

    m = h_coarse_pat.search(fname)
    if m:
        model, dataset = m.group(1), m.group(2)
        mat = read_matrix_csv(path)
        if mat.size:
            heat_coarse[(dataset, model)] = mat
        continue

    m = h_fine_pat.search(fname)
    if m:
        it = int(m.group(1))
        model, dataset = m.group(2), m.group(3)
        mat = read_matrix_csv(path)
        if mat.size:
            heat_fine.setdefault((dataset, model), {})[it] = mat
        continue

for (dataset, model), coarse_mat in heat_coarse.items():
    fine_mats_dict = heat_fine.get((dataset, model), {})
    chosen_iters = pick_iters(sorted(fine_mats_dict.keys()), k=4)
    fine_mats = [fine_mats_dict[i] for i in chosen_iters]

    # Coarse uses its own max; fine-grained each uses its own max (independent scaling)
    vmax_coarse = np.max(coarse_mat) if coarse_mat.size else None

    # Layout: left big pane (coarse) + right row of small panes in rounded box
    n_f = max(1, len(fine_mats))
    fig_w = 8 + n_f * 1.6
    fig_h = 4.0
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                            gridspec_kw={
                                                "width_ratios": [1.1, n_f * 0.9]},
                                            constrained_layout=False)

    # Left: coarse heatmap (normalized to its own max)
    draw_heatmap(ax_left, coarse_mat, vmin=0.0, vmax=vmax_coarse)
    ax_left.set_title("Coarse-grained expert heatmap", pad=10)
    ax_left.set_xlabel("Layers")
    ax_left.set_ylabel("Experts")

    # Right: background box
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.set_frame_on(False)

    pad = 0.02
    rect = FancyBboxPatch((0+pad, 0+pad), 1-2*pad, 1-2*pad,
                          transform=ax_right.transAxes,
                          boxstyle="round,pad=0.02,rounding_size=0.02",
                          linewidth=0.8, edgecolor="lightgray",
                          facecolor="#f5f5f5", zorder=0)
    ax_right.add_patch(rect)

    # Layout inside the rounded box so the grey backdrop fully covers all fine panels
inner_pad = 0.04  # padding inside the rounded box
box_left = inner_pad
box_right = 1 - inner_pad
box_bottom = inner_pad
box_top = 1 - inner_pad

# Available area inside the box for panels
left_margin = box_left + 0.02
right_margin = box_right - 0.02
available_w = right_margin - left_margin

# Panel sizing and spacing
gap_frac = 0.25
unit = 1.0 / (n_f + (n_f - 1) * gap_frac)
panel_w = available_w * unit
step = panel_w * (1 + gap_frac)
x0 = left_margin
# Vertical placement
v_pad = 0.08
y0 = box_bottom + v_pad
h = (box_top - box_bottom) - 2 * v_pad

# Place fine-grained heatmaps
mid_gap_idx = (n_f // 2) - 1 if n_f >= 3 else None
for i, mat in enumerate(fine_mats):
    ax_in = ax_right.inset_axes([x0 + i * step, y0, panel_w, h])
    vmax_local = np.max(mat) if mat.size else None
    draw_heatmap(ax_in, mat, vmin=0.0, vmax=vmax_local)
    if chosen_iters:
        if i == 0:
            ax_in.set_xlabel(f"iter {chosen_iters[0]}")
        elif i == len(fine_mats) - 1:
            ax_in.set_xlabel(f"iter {chosen_iters[-1]}")
    if mid_gap_idx is not None and i == mid_gap_idx:
        # Ellipsis between middle panels
        ax_right.text(x0 + (i + 1) * step - panel_w * 0.2, 0.5, "…",
                      transform=ax_right.transAxes, fontsize=16, va="center")

    ax_right.set_title("Fine-grained expert heatmap", pad=10)
    ax_right.annotate("aggregate",
                      xy=(-0.04, 0.5), xycoords=("axes fraction", "axes fraction"),
                      xytext=(-0.12, 0.5), textcoords=("axes fraction", "axes fraction"),
                      arrowprops=dict(arrowstyle="<-", lw=1.0),
                      ha="center", va="center", fontsize=10, rotation=90)

    fig.suptitle(f"{model} | {dataset}", y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    fig1_path = str(outdir / "figure_3a.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)

# =============================================================
# Collect entropy bars/lines data once
# =============================================================
bars = {}   # {(dataset, model): {"coarse": mean, "fine": mean}}
lines = {}  # {dataset: {model: np.array([...])}}

for path in glob.glob(str(base / "entropy~*.csv")):
    fname = os.path.basename(path)

    m = coarse_pat.search(fname)
    if m:
        model, dataset = m.group(1), m.group(2)
        vals = read_floats(path)
        if vals.size:
            bars.setdefault((dataset, model), {})[
                "coarse"] = float(np.mean(vals))
        continue

    m = fine_pat.search(fname)
    if m:
        model, dataset = m.group(1), m.group(2)
        vals = read_floats(path)
        if vals.size:
            bars.setdefault((dataset, model), {})[
                "fine"] = float(np.mean(vals))
        continue

    m = steps_pat.search(fname)
    if m:
        model, dataset = m.group(1), m.group(2)
        vals = read_floats(path)
        if vals.size:
            lines.setdefault(dataset, {})[model] = vals
        continue

# =============================================================
# Figure 3b: Grouped bars (coarse vs. fine)
# =============================================================
categories = sorted(bars.keys(), key=lambda x: (x[0].lower(), x[1].lower()))
labels = [f"{m}\n{d}" for (d, m) in categories]
coarse_vals = [bars[k].get("coarse", np.nan) for k in categories]
fine_vals = [bars[k].get("fine",   np.nan) for k in categories]

x = np.arange(len(categories))
width = 0.38

fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.bar(x - width/2, coarse_vals, width, color="red", label="Coarse-grained")
ax1.bar(x + width/2, fine_vals, width, color="blue", label="Fine-grained")

ax1.set_ylabel("Mean entropy")
ax1.set_xticks([])
ax1.set_xlabel("")
ax1.set_title(
    f"{list(bars.keys())[0][1]} | {list(bars.keys())[0][0]}: Coarse-grained vs Fine-grained")
ax1.legend(loc="best")
ax1.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
fig1.tight_layout()

fig2_path = str(outdir / "figure_3b.png")
fig1.savefig(fig2_path, dpi=300)

# =============================================================
# Figure 3c: Steps lines (single dataset, single figure)
# =============================================================
# Expect exactly one dataset in `lines`; fall back gracefully if empty
if len(lines) == 0:
    print("No steps CSVs found; skipping figure_3c.png")
else:
    dataset, model_series = next(
        iter(sorted(lines.items(), key=lambda kv: kv[0].lower())))
    fig2, ax = plt.subplots(1, 1, figsize=(12, 4), sharey=False)

    for model, series in sorted(model_series.items(), key=lambda x: x[0].lower()):
        iterations = np.arange(1, len(series) + 1)
        ax.plot(iterations, series, label=model)

    ax.set_title(
        f"{model} | {dataset}: Cumulative Mean Entropy Through Iterations")
    ax.set_xlabel("Inference iterations")
    ax.set_ylabel("Mean entropy")
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(loc="best", fontsize="small")

    fig2.tight_layout()
    fig3_path = str(outdir / "figure_3c.png")
    fig2.savefig(fig3_path, dpi=300)

print("Saved entropy experiments:", fig1_path, fig2_path, fig3_path)
