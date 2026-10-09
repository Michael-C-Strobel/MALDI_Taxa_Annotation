#!/usr/bin/env python3
"""Generate publication-ready figures and tables from Optuna studies.

This script is standalone and does not modify existing project files.
It reads one or more Optuna SQLite study DBs and exports:
- Trial summary tables (CSV and LaTeX)
- Best hyperparameter tables (CSV and LaTeX)
- Top-k trial tables (CSV and LaTeX)
- Publication-style figures (PNG, PDF, SVG)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from optuna.importance import get_param_importances
from optuna.trial import TrialState


DEFAULT_RUNS = [
    (
        "genera_species",
        "bin/ml/lightning_logs_driams_optuna/genera/species/CLIP_MALDI/optuna_study.db",
        "bin/ml/lightning_logs_driams_optuna/genera/species/CLIP_MALDI/CLIP_MALDI.json",
    ),
    (
        "genera_genera",
        "bin/ml/lightning_logs_driams_optuna/genera/genera/CLIP_MALDI/optuna_study.db",
        "bin/ml/lightning_logs_driams_optuna/genera/genera/CLIP_MALDI/CLIP_MALDI.json",
    ),
]

COLORS = {
    "cosine": "#fc8d62",
    "cosine_intensity_agnostic": "#e78ac3",
    "clip_transformer": "#66c2a5",
    "multinomial_classifier": "#8da0cb",
    "cosine_10": "#fc8d62",
    "cosine_7": "#e78ac3",
    "cosine_5": "#66c2a5",
    "cosine_3": "#8da0cb",
    "cosine_1": "#ffc82f",
    "cosine_10_intensity_agnostic": "#fc8d62",
    "cosine_7_intensity_agnostic": "#e78ac3",
    "cosine_5_intensity_agnostic": "#66c2a5",
    "cosine_3_intensity_agnostic": "#8da0cb",
    "cosine_1_intensity_agnostic": "#ffc82f",
    "Theoretical Max": "#727272",
    "maldi_transformer_ts": "#a6d854",
    "cross_encoder": "#ff2f2f",
}

RUN_COLORS = {
    "genera_species": COLORS["clip_transformer"],
    "genera_genera": COLORS["multinomial_classifier"],
}


@dataclass(frozen=True)
class RunSpec:
    label: str
    db_path: Path
    best_json_path: Path | None


def parse_run_spec(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Invalid --run '{raw}'. Expected LABEL=PATH_TO_DB")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = Path(path.strip())
    if not label:
        raise ValueError(f"Invalid --run '{raw}'. LABEL cannot be empty")
    return label, path


def parse_best_json_spec(raw: str) -> Tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(
            f"Invalid --best-json '{raw}'. Expected LABEL=PATH_TO_JSON"
        )
    label, path = raw.split("=", 1)
    label = label.strip()
    path = Path(path.strip())
    if not label:
        raise ValueError(f"Invalid --best-json '{raw}'. LABEL cannot be empty")
    return label, path


def resolve_run_specs(args: argparse.Namespace) -> List[RunSpec]:
    if args.use_defaults:
        runs = [
            RunSpec(label=label, db_path=Path(db), best_json_path=Path(best_json))
            for (label, db, best_json) in DEFAULT_RUNS
        ]
    elif args.run:
        runs = [
            RunSpec(label=label, db_path=db_path, best_json_path=None)
            for (label, db_path) in [parse_run_spec(item) for item in args.run]
        ]
    else:
        runs = [
            RunSpec(
                label="genera_species",
                db_path=Path(DEFAULT_RUNS[0][1]),
                best_json_path=Path(DEFAULT_RUNS[0][2]),
            )
        ]

    if args.best_json:
        best_json_map = dict(parse_best_json_spec(item) for item in args.best_json)
        updated_runs: List[RunSpec] = []
        for run in runs:
            updated_runs.append(
                RunSpec(
                    label=run.label,
                    db_path=run.db_path,
                    best_json_path=best_json_map.get(run.label, run.best_json_path),
                )
            )
        runs = updated_runs

    return runs


def configure_plot_style() -> None:
    unified_font_size = 16
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": unified_font_size,
            "axes.titlesize": unified_font_size,
            "axes.labelsize": unified_font_size,
            "xtick.labelsize": unified_font_size,
            "ytick.labelsize": unified_font_size,
            "legend.fontsize": unified_font_size,
            "font.family": "Arial",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_study(db_path: Path) -> optuna.study.Study:
    storage = f"sqlite:///{db_path.resolve()}"
    return optuna.load_study(study_name="optuna_study", storage=storage)


def extract_trials_dataframe(study: optuna.study.Study) -> pd.DataFrame:
    df = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    df = df.rename(columns={"number": "trial", "value": "objective"})
    return df


def complete_trials(study: optuna.study.Study) -> List[optuna.trial.FrozenTrial]:
    return [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]


def best_is_lower(study: optuna.study.Study) -> bool:
    return study.direction == optuna.study.StudyDirection.MINIMIZE


def summarize_run(label: str, study: optuna.study.Study, df: pd.DataFrame) -> Dict[str, float | int | str]:
    comp = df[df["state"] == "COMPLETE"].copy()
    pruned = int((df["state"] == "PRUNED").sum())
    failed = int((df["state"] == "FAIL").sum())
    complete_n = int(len(comp))

    if complete_n == 0:
        return {
            "run": label,
            "direction": str(study.direction),
            "n_trials_total": int(len(df)),
            "n_complete": 0,
            "n_pruned": pruned,
            "n_failed": failed,
            "best_objective": np.nan,
            "median_objective": np.nan,
            "iqr_objective": np.nan,
            "objective_p05": np.nan,
            "objective_p95": np.nan,
        }

    q1 = float(comp["objective"].quantile(0.25))
    q3 = float(comp["objective"].quantile(0.75))

    return {
        "run": label,
        "direction": str(study.direction).split(".")[-1],
        "n_trials_total": int(len(df)),
        "n_complete": complete_n,
        "n_pruned": pruned,
        "n_failed": failed,
        "best_objective": float(comp["objective"].min() if best_is_lower(study) else comp["objective"].max()),
        "median_objective": float(comp["objective"].median()),
        "iqr_objective": float(q3 - q1),
        "objective_p05": float(comp["objective"].quantile(0.05)),
        "objective_p95": float(comp["objective"].quantile(0.95)),
    }


def build_best_hparams_table(label: str, study: optuna.study.Study) -> pd.DataFrame:
    rows = []
    best_params = study.best_trial.params
    for name, value in sorted(best_params.items()):
        rows.append({"run": label, "parameter": name, "best_value": value})
    return pd.DataFrame(rows)


def top_k_trials_table(label: str, study: optuna.study.Study, top_k: int) -> pd.DataFrame:
    records = []
    for t in complete_trials(study):
        row = {"run": label, "trial": t.number, "objective": float(t.value)}
        row.update({f"param_{k}": v for k, v in t.params.items()})
        records.append(row)

    if not records:
        return pd.DataFrame(columns=["run", "trial", "objective"])

    df = pd.DataFrame(records)
    ascending = best_is_lower(study)
    df = df.sort_values("objective", ascending=ascending).head(top_k).reset_index(drop=True)
    return df


def save_table(df: pd.DataFrame, stem: Path, float_format: str = "%.5f") -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    latex = df.to_latex(index=False, escape=False, float_format=float_format)
    stem.with_suffix(".tex").write_text(latex)


def save_fig(fig: plt.Figure, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(out_stem.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def plot_optimization_history(label: str, df: pd.DataFrame, is_minimize: bool) -> plt.Figure:
    comp = df[df["state"] == "COMPLETE"].sort_values("trial").copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if comp.empty:
        ax.text(0.5, 0.5, "No COMPLETE trials", ha="center", va="center")
        ax.set_title(f"Optimization History: {label}")
        return fig

    comp["best_so_far"] = comp["objective"].cummin() if is_minimize else comp["objective"].cummax()
    run_color = RUN_COLORS.get(label, COLORS["clip_transformer"])
    ax.scatter(comp["trial"], comp["objective"], s=18, alpha=0.55, color=COLORS["cosine_10"], label="Trial objective")
    ax.plot(comp["trial"], comp["best_so_far"], lw=2.2, color=run_color, label="Best so far")
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Objective")
    ax.set_title(f"Optimization History: {label}")
    ax.legend(frameon=False)
    return fig


def plot_objective_distribution(label: str, df: pd.DataFrame) -> plt.Figure:
    comp = df[df["state"] == "COMPLETE"].copy()
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    if comp.empty:
        ax.text(0.5, 0.5, "No COMPLETE trials", ha="center", va="center")
        ax.set_title(f"Objective Distribution: {label}")
        return fig

    sns.histplot(comp, x="objective", bins=30, kde=True, color=RUN_COLORS.get(label, COLORS["multinomial_classifier"]), ax=ax)
    ax.set_title(f"Objective Distribution: {label}")
    ax.set_xlabel("Objective")
    return fig


def plot_param_importance(label: str, study: optuna.study.Study) -> plt.Figure:
    def _clean_param_label(name: str) -> str:
        return name.replace("params_", "").replace("_", " ").strip()

    importances = get_param_importances(study)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if not importances:
        ax.text(0.5, 0.5, "No parameter importances available", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        names = list(importances.keys())[::-1]
        values = [importances[n] for n in names]
        labels = [_clean_param_label(n) for n in names]
        ax.barh(labels, values, color=RUN_COLORS.get(label, COLORS["cosine"]))
        ax.set_xlabel("Importance")

    ax.set_title("")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        spine = ax.spines[side]
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)

    ax.tick_params(axis="both", colors="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    return fig


def _log_scale_if_positive(ax: plt.Axes, series: pd.Series, axis: str = "x") -> None:
    vals = series.dropna()
    if len(vals) and (vals > 0).all() and (vals.max() / vals.min() > 100):
        if axis == "x":
            ax.set_xscale("log")
        else:
            ax.set_yscale("log")


def plot_param_vs_objective_grid(label: str, df: pd.DataFrame) -> plt.Figure:
    grid_font_size = 20
    pretty_label = label.replace("_", " ")
    comp = df[df["state"] == "COMPLETE"].copy()
    param_cols = [c for c in comp.columns if c.startswith("params_")]
    n = len(param_cols)

    if n == 0:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, "No parameter columns available", ha="center", va="center")
        ax.set_title(f"Parameter Effects: {label}")
        return fig

    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.2 * ncols, 3.9 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for i, col in enumerate(param_cols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        sns.scatterplot(data=comp, x=col, y="objective", s=22, alpha=0.7, color=RUN_COLORS.get(label, COLORS["clip_transformer"]), ax=ax)
        if comp[col].nunique() > 7:
            sns.regplot(data=comp, x=col, y="objective", scatter=False, lowess=True, ax=ax, color=COLORS["Theoretical Max"])
        _log_scale_if_positive(ax, comp[col], axis="x")
        clean_param_name = col.replace("params_", "").replace("_", " ")
        ax.set_xlabel(clean_param_name, fontsize=grid_font_size)
        ax.set_ylabel("Objective", fontsize=grid_font_size)
        ax.set_ylim(0.5, 3.3)
        ax.tick_params(axis="both", labelsize=grid_font_size)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.suptitle(f"Parameter-Objective Relationships: {pretty_label}", y=1.01, fontsize=grid_font_size)
    fig.tight_layout()
    return fig


def plot_comparative_distribution(combined_df: pd.DataFrame) -> plt.Figure:
    comp = combined_df[combined_df["state"] == "COMPLETE"].copy()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    if comp.empty:
        ax.text(0.5, 0.5, "No COMPLETE trials across runs", ha="center", va="center")
        ax.set_title("Objective Comparison Across Splits")
        return fig

    run_palette = {run: RUN_COLORS.get(run, COLORS["cosine"]) for run in comp["run"].unique()}
    sns.violinplot(data=comp, x="run", y="objective", inner="box", cut=0, palette=run_palette, ax=ax)
    sns.stripplot(data=comp, x="run", y="objective", color="black", size=2.6, alpha=0.35, ax=ax)
    ax.set_xlabel("Dataset split")
    ax.set_ylabel("Objective")
    ax.set_title("Objective Comparison Across Splits")
    return fig


def plot_comparative_convergence(run_to_df: Dict[str, pd.DataFrame], run_to_is_min: Dict[str, bool]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.8, 5.3))

    plotted = 0
    for run, df in run_to_df.items():
        comp = df[df["state"] == "COMPLETE"].sort_values("trial").copy()
        if comp.empty:
            continue
        comp["best_so_far"] = comp["objective"].cummin() if run_to_is_min[run] else comp["objective"].cummax()
        ax.plot(comp["trial"], comp["best_so_far"], lw=2.4, label=run, color=RUN_COLORS.get(run, COLORS["cross_encoder"]))
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No COMPLETE trials across runs", ha="center", va="center")
    else:
        ax.legend(frameon=False)

    ax.set_xlabel("Trial number")
    ax.set_ylabel("Best objective so far")
    ax.set_title("Convergence Comparison Across Splits")
    return fig


def ensure_paths_exist(runs: Iterable[RunSpec], project_root: Path) -> None:
    missing = []
    for run in runs:
        if not (project_root / run.db_path).exists():
            missing.append(str(project_root / run.db_path))
    if missing:
        missing_msg = "\n".join(missing)
        raise FileNotFoundError(f"Missing study DB files:\n{missing_msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready Optuna figures and tables")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root used to resolve relative input paths",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/publication_optuna"),
        help="Output folder for figures and tables",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec as LABEL=PATH_TO_OPTUNA_DB (repeatable)",
    )
    parser.add_argument(
        "--best-json",
        action="append",
        default=[],
        help="Best-json spec as LABEL=PATH_TO_BEST_JSON (repeatable)",
    )
    parser.add_argument(
        "--use-defaults",
        action="store_true",
        help="Use both predefined runs (genera_species and genera_genera)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top complete trials to export per run",
    )

    args = parser.parse_args()
    project_root = args.project_root.resolve()
    out_dir = (project_root / args.output_dir).resolve()

    runs = resolve_run_specs(args)
    ensure_paths_exist(runs, project_root)
    configure_plot_style()

    summary_rows: List[Dict[str, float | int | str]] = []
    best_rows: List[pd.DataFrame] = []
    top_rows: List[pd.DataFrame] = []

    combined_trials: List[pd.DataFrame] = []
    run_to_df: Dict[str, pd.DataFrame] = {}
    run_to_is_min: Dict[str, bool] = {}

    for run in runs:
        study = load_study(project_root / run.db_path)
        df = extract_trials_dataframe(study)
        df["run"] = run.label

        summary_rows.append(summarize_run(run.label, study, df))
        best_rows.append(build_best_hparams_table(run.label, study))
        top_rows.append(top_k_trials_table(run.label, study, args.top_k))

        combined_trials.append(df)
        run_to_df[run.label] = df
        run_to_is_min[run.label] = best_is_lower(study)

        run_fig_dir = out_dir / "figures" / run.label
        save_fig(plot_optimization_history(run.label, df, best_is_lower(study)), run_fig_dir / "optimization_history")
        save_fig(plot_objective_distribution(run.label, df), run_fig_dir / "objective_distribution")
        save_fig(plot_param_importance(run.label, study), run_fig_dir / "param_importance")
        save_fig(plot_param_vs_objective_grid(run.label, df), run_fig_dir / "param_vs_objective_grid")

    summary_df = pd.DataFrame(summary_rows)
    best_df = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
    top_df = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    all_trials_df = pd.concat(combined_trials, ignore_index=True) if combined_trials else pd.DataFrame()

    table_dir = out_dir / "tables"
    save_table(summary_df, table_dir / "study_summary")
    save_table(best_df, table_dir / "best_hyperparameters")
    save_table(top_df, table_dir / f"top_{args.top_k}_trials")

    if not all_trials_df.empty and len(runs) > 1:
        comp_fig_dir = out_dir / "figures" / "comparison"
        save_fig(plot_comparative_distribution(all_trials_df), comp_fig_dir / "objective_distribution_comparison")
        save_fig(plot_comparative_convergence(run_to_df, run_to_is_min), comp_fig_dir / "convergence_comparison")

    manifest_lines = [
        "Generated publication assets:",
        f"- Output directory: {out_dir}",
        f"- Runs: {', '.join(r.label for r in runs)}",
        f"- Tables: {table_dir}",
        f"- Figures: {out_dir / 'figures'}",
    ]
    (out_dir / "MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n")

    print("\n".join(manifest_lines))


if __name__ == "__main__":
    main()
