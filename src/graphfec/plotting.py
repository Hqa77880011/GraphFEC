"""Plots generated exclusively from evaluation output."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_results(inputs, output):
    rows = []
    synthetic = False
    for path in inputs:
        with Path(path).open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))
        metadata = Path(path).parent / "evaluation.json"
        if metadata.exists():
            synthetic |= json.loads(metadata.read_text(encoding="utf-8")).get("synthetic", False)
    if not rows:
        raise ValueError("No evaluation rows to plot")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    for channel in sorted({row["channel"] for row in rows}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        palette = ["#3e6482", "#658d99", "#579181", "#7a8790", "#849b69", "#8a7692", "#a08d73"]
        for method_index, method in enumerate(dict.fromkeys(row["method"] for row in rows)):
            selected = [r for r in rows if r["channel"] == channel and r["method"] == method]
            if not selected:
                continue
            rates = sorted({float(r["loss_rate"]) for r in selected})
            for ax, metric, label in zip(
                axes,
                ("accuracy", "mae", "cosine"),
                ("Top-1 accuracy (%)", "Activation MAE", "Cosine similarity"),
            ):
                scale = 100 if metric == "accuracy" else 1
                means = [
                    np.mean([float(r[metric]) for r in selected if float(r["loss_rate"]) == p])
                    * scale
                    for p in rates
                ]
                x_values = (
                    np.array(rates) * 100
                    if channel != "trace"
                    else [100 * np.mean([float(r["observed_loss"]) for r in selected])]
                )
                ax.plot(
                    x_values,
                    means,
                    marker="o" if method_index < 7 else "s",
                    markersize=3,
                    color=palette[method_index % len(palette)],
                    linestyle="-" if method_index < 7 else "--",
                    label=method,
                )
                ax.set(
                    xlabel="Packet loss (%)" if channel != "trace" else "Observed trace loss (%)",
                    ylabel=label,
                )
                ax.grid(alpha=0.2)
        axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.suptitle(f"{channel}: arithmetic means across supplied run/trace rows")
        if synthetic:
            fig.text(
                0.5,
                0.01,
                "Synthetic functional-check data; not benchmark results.",
                ha="center",
                fontsize=9,
            )
        fig.tight_layout(rect=(0, 0.05 if synthetic else 0, 1, 1))
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{channel}_metrics.{extension}", dpi=180)
        plt.close(fig)
    # Direct component comparison at 10% independent loss, where available.
    selected = [r for r in rows if r["channel"] == "bernoulli" and float(r["loss_rate"]) == 0.1]
    if selected:
        methods = list(dict.fromkeys(r["method"] for r in selected))
        means = [
            100 * np.mean([float(r["accuracy"]) for r in selected if r["method"] == m])
            for m in methods
        ]
        fig, ax = plt.subplots(figsize=(7, max(3, len(methods) * 0.35)))
        ax.barh(methods, means, color="#528a9e")
        ax.set(xlabel="Top-1 accuracy (%)", title="10% Bernoulli loss", xlim=(0, 100))
        if synthetic:
            fig.text(
                0.5,
                0.01,
                "Synthetic functional-check data; not benchmark results.",
                ha="center",
                fontsize=9,
            )
        fig.tight_layout(rect=(0, 0.06 if synthetic else 0, 1, 1))
        fig.savefig(output / "component_comparison.png", dpi=180)
        plt.close(fig)
    print(f"Saved plots to {output}")
