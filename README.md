# Genus annotation from MALDI-TOF mass spectra

Code and frozen analysis outputs for the mSystems manuscript, *Machine learning-augmented annotation of bacterial genera using MALDI-TOF mass spectra*.

The publication workflow uses DRIAMS-A to train a contrastive transformer and reproduce the MALDI Transformer baseline, then evaluates DRIAMS-A, DRIAMS-B/C/D, and RKI. The banana-slug application includes its saved IDBac search results and genome annotation/QC workup in this repository.

## Publication assets

- `scratch/publication_figures/`: frozen CSV metrics and SVG figures. These are the outputs checked against the manuscript; do not overwrite them when reproducing an analysis.
- `scratch/publication_figures.ipynb`: original analysis notebook, including exploratory sections.
- `publication/idbac_wgs/`: 47-strain supplementary search evaluation, cleaned notebook, and deployed model provenance.
- `scratch/publication_metrics.ipynb`: curated publication analysis, with regenerated outputs written separately.
- `scratch/publication_figures_helpers.py` and `scratch/publication_figures_metrics.py`: embedding loading, pair construction, metric calculation, and plotting.
- `publication/banana_slug/`: frozen search results, 16S/CheckM2/GTDB tables, original figures, reproduction notebook, and genome-workup source scripts.
- [Manuscript-to-results mapping](docs/manuscript_results.md).
- [Full reproduction workflow](docs/reproducibility.md).
- `python scripts/publication/verify_artifacts.py`: verify the frozen input/result hashes without recomputing metrics.
- [Outstanding provenance and methods questions](docs/submission_audit.md).

The published supplementary PR-AUC tables match all 128 checked values in the frozen CSV files to four decimal places. The separate 47-strain WGS/IDBac evaluation includes its combined search tables, focused notebook, saved figure panels, and production-matched model snapshot in `publication/idbac_wgs/`. Its saved top-1 counts and genus-weighted accuracies match the manuscript. The manuscript itself is not modified by this repository curation.

## Install

Run the following commands separately from the repository root. They also paste into PowerShell without shell continuation syntax. Training and preprocessing scripts run on Linux; PowerShell can be used to launch the commands through your existing SSH session.

```text
git submodule update --init --recursive
conda env create -f bin/environment.yml -p ./conda_env
conda env create -f bin/ml/environment.yml -p ./ml_maldi_nn_conda_env
```

The MALDI Transformer submodule is pinned at `afccd3c2fd4ff1e71701f397db6858077831f8ee`. Keep that revision when reproducing the baseline. The ML environment additionally needs the submodule's dependencies and the analysis packages listed in `publication/requirements-analysis.txt`; see the full workflow. Environment specifications describe the software requirements; they do not guarantee bitwise identity across CUDA hardware or package solver revisions.

## Reproduction paths

1. **Read the reported metrics:** open the frozen CSVs and SVGs in `scratch/publication_figures/`. This requires no training or inference.
2. **Recompute metrics:** restore the frozen metadata/split files, acquire and preprocess the original spectra, run inference using the publication checkpoints, then execute `scratch/publication_metrics.ipynb`. Saved embeddings are generated locally rather than distributed as an additional artifact.
3. **Retrain:** follow the same preprocessing and fixed splits, then train with the saved DRIAMS hyperparameters. Final models were not hyperparameter-tuned on IDBac.
4. **47-strain IDBac evaluation:** execute `publication/idbac_wgs/notebooks/idbac_wgs_publication.ipynb` from its notebook directory. It reads the included combined search results and regenerates supplementary top-k accuracy figures without the 16S comparison.
5. **Banana-slug figures:** execute `publication/banana_slug/notebooks/banana_slug_publication.ipynb` from its notebook directory. It reads the included frozen hit table and reproduces the ranked-hit plots and pooled precision curve without rerunning genome tools or a remote IDBac query.

Publication evaluations use folds **0–5**. Fold 6 is excluded because its test split is the validation split of fold 0 used for hyperparameter optimization. Species-disjoint folds are used for the main evaluations; genus-disjoint folds are used for the genus generalization experiment. Preserve the exact split tensors and taxonomy labels rather than regenerating them against a current taxonomy database.

## License and attribution

Repository-authored code is released under the MIT license; see `LICENSE`. External software, submodules, and source datasets retain their own licenses and attribution. See `publication/banana_slug/PROVENANCE.md` and `publication/idbac_wgs/PROVENANCE.md` for the included workup, search results, and workflow-model provenance.
