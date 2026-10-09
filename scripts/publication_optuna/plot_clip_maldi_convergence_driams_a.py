#!/usr/bin/env python3
"""Plot CLIP-MALDI convergence curves for DRIAMS-A split experiments.

Uses only:
- model directory: CLIP_Transformer (CLIP MALDI)
- logger version: version_0
- folds: all except k=6

Inputs:
- bin/ml/lightning_logs_DRIAMS_A_for_score/genera/genera
- bin/ml/lightning_logs_DRIAMS_A_for_score/genera/species

Outputs:
- figures/publication_optuna/convergence/clip_maldi_driams_a_convergence.(png|pdf|svg)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCALAR_TAG = "val_loss_epoch"
EXCLUDED_FOLDS = {6}
MODEL_DIR = "CLIP_Transformer"
VERSION_DIR = "version_0"

RUNS: Dict[str, str] = {
    "Genera -> Genera": "bin/ml/lightning_logs_DRIAMS_A_for_score/genera/genera",
    "Genera -> Species": "bin/ml/lightning_logs_DRIAMS_A_for_score/genera/species",
}

RUN_COLORS = {
    "Genera -> Genera": "#2A6F97",
    "Genera -> Species": "#0FA67A",
}


def configure_style() -> None:
    font_size = 16
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "Arial",
            "font.size": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "axes.grid": False,
        }
    )


def parse_fold(path: Path) -> int | None:
    match = re.search(r"k=(\d+)", path.as_posix())
    if not match:
        return None
    return int(match.group(1))


def discover_event_files(root: Path) -> List[tuple[int, Path]]:
    pattern = f"k=*/{MODEL_DIR}/{VERSION_DIR}/events.out.tfevents.*"
    event_files = []
    for event_file in sorted(root.glob(pattern)):
        fold = parse_fold(event_file)
        if fold is None or fold in EXCLUDED_FOLDS:
            continue
        event_files.append((fold, event_file))
    return event_files


def read_scalar_series(event_file: Path, tag: str) -> pd.DataFrame:
    accumulator = EventAccumulator(str(event_file))
    accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    if tag not in tags:
        raise ValueError(f"Tag '{tag}' not found in {event_file}")

    events = accumulator.Scalars(tag)
    if not events:
        raise ValueError(f"No scalar events for '{tag}' in {event_file}")

    return pd.DataFrame(
        {
            "step": [int(ev.step) for ev in events],
            "value": [float(ev.value) for ev in events],
        }
    )


def aggregate_run(root: Path, run_name: str, tag: str) -> tuple[pd.DataFrame, List[int]]:
    fold_files = discover_event_files(root)
    if not fold_files:
        raise FileNotFoundError(f"No event files found for {run_name} under {root}")

    fold_curves = []
    used_folds: List[int] = []
    for fold, event_file in fold_files:
        curve = read_scalar_series(event_file, tag)
        curve["fold"] = fold
        fold_curves.append(curve)
        used_folds.append(fold)

    combined = pd.concat(fold_curves, ignore_index=True)
    summary = (
        combined.groupby("step", as_index=False)["value"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "value_mean",
                "std": "value_std",
                "min": "value_min",
                "max": "value_max",
                "count": "n_folds",
            }
        )
    )
    summary["value_std"] = summary["value_std"].fillna(0.0)
    return summary, sorted(used_folds)


def plot_convergence(run_summaries: Dict[str, pd.DataFrame], run_folds: Dict[str, List[int]], out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))

    for run_name, summary in run_summaries.items():
        color = RUN_COLORS.get(run_name, "#333333")
        steps = summary["step"].to_numpy()
        mean_vals = summary["value_mean"].to_numpy()
        std_vals = summary["value_std"].to_numpy()

        label = f"{run_name} (folds: {', '.join(str(k) for k in run_folds[run_name])})"
        ax.plot(steps, mean_vals, color=color, linewidth=2.8, label=label)
        ax.fill_between(
            steps,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.set_title("CLIP MALDI Convergence (version 0, excluding fold 6)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.tick_params(axis="both", colors="black")
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(out_stem.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CLIP-MALDI convergence for DRIAMS-A genera splits")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root")
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("figures/publication_optuna/convergence/clip_maldi_driams_a_convergence"),
        help="Output path stem without extension",
    )
    parser.add_argument(
        "--scalar-tag",
        type=str,
        default=SCALAR_TAG,
        help="TensorBoard scalar tag to plot for convergence",
    )
    args = parser.parse_args()

    configure_style()
    root = args.project_root.resolve()

    run_summaries: Dict[str, pd.DataFrame] = {}
    run_folds: Dict[str, List[int]] = {}
    for run_name, run_rel_path in RUNS.items():
        summary, folds = aggregate_run(root / run_rel_path, run_name, args.scalar_tag)
        run_summaries[run_name] = summary
        run_folds[run_name] = folds

    out_stem = (root / args.output_stem).resolve()
    plot_convergence(run_summaries, run_folds, out_stem)
    print(f"Wrote {out_stem.with_suffix('.png')}")
    print(f"Wrote {out_stem.with_suffix('.pdf')}")
    print(f"Wrote {out_stem.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
