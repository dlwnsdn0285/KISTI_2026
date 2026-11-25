"""Step 27: REFRAG-style figures from artifacts/step27/curves.csv.

  ~/anaconda3/bin/python scripts/27_plot_latency.py      # needs matplotlib (base env)

acceleration.{png,pdf}  raw / method for TTFT and TTIT, method / raw for throughput (raw = 1.0 series)
absolute.{png,pdf}      measured TTFT (s), TTIT (ms/token), throughput (tokens/s) of raw, ours, online
x = median raw prompt tokens (log). Colors: categorical slots 1-3 of the dataviz reference palette; line style
and marker also differ per series, so identity never rests on color alone.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "artifacts" / "step27"
SERIES = {"ours": ("OURS (cached memory + block)", "#2a78d6", "-", "o"),
          "online": ("Online (compress at query time)", "#eb6834", "--", "s"),
          "raw": ("Raw (Mistral-7B, full text)", "#1baf7a", "-.", "D")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
rows = list(csv.DictReader((OUT / "curves.csv").open()))


def series(method, key, scale=1.0):
    pts = sorted((float(r["raw_input_tokens"]), float(r[key]) * scale, int(r["batch"])) for r in rows if r["method"] == method)
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts]


def style(ax, title, ylabel, logy=False):
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=14, color=INK)
    ax.set_xlabel("# Input Tokens", fontsize=12, color=INK)
    ax.set_ylabel(ylabel, fontsize=12, color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=10)
    for s in ax.spines.values():
        s.set_color(GRID)


def figure(panels, name, order):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, scale, title, ylabel, logy) in zip(axes, panels):
        for m in order:
            label, color, ls, mk = SERIES[m]
            x, y, b = series(m, key, scale)
            ax.plot(x, y, ls=ls, marker=mk, color=color, lw=2, ms=6, label=label, zorder=3 if m != "raw" else 2)
            if key == "throughput_tok_s" and m == "raw":   # raw batch shrinks with length (OOM); PISCO stays at 8
                for xi, yi, bi in zip(x, y, b):
                    if bi < 8:
                        ax.annotate(f"B={bi}", (xi, yi), textcoords="offset points", xytext=(-7, -4), ha="right", va="top", fontsize=8, color=MUTED)
        style(ax, title, ylabel, logy)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True, fontsize=11, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", OUT / f"{name}.png")


figure([("ttft_accel", 1, "TTFT Acceleration", "Acceleration", False),
        ("ttit_accel", 1, "TTIT Acceleration", "Acceleration", False),
        ("throughput_accel", 1, "Throughput Acceleration", "Acceleration", False)], "acceleration", ["ours", "online", "raw"])
figure([("ttft_s", 1, "TTFT", "Seconds (log)", True),
        ("ttit_s", 1000, "TTIT", "ms / token", False),
        ("throughput_tok_s", 1, "Throughput", "Tokens / s (log)", True)], "absolute", ["ours", "online", "raw"])
