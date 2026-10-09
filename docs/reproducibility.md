# Reproduce the publication analyses

Run Linux workflow commands from the repository root unless a step explicitly changes directory. Commands are written on single lines, without Bash or PowerShell continuation characters. Activate the appropriate conda environment in the shell that runs the command. Do not overwrite frozen publication outputs.

## 1. Software

```text
conda env create -f bin/environment.yml -p ./conda_env
conda env create -f bin/ml/environment.yml -p ./ml_maldi_nn_conda_env
conda activate ./ml_maldi_nn_conda_env
python -m pip install -r publication/requirements-analysis.txt
python -m pip install -e bin/ml/models/MaldiTransformer/maldi-nn
```

The submodule's software/license is independent of this project's MIT license. Check the recorded environment package inventory before solving a new environment; dependencies unpinned in the older YAML files can change.

## 2. Inputs, taxonomy, and folds

Keep the original data layout:

| Data | Download/input path | Frozen metadata | Contrastive/cosine tensor path |
|---|---|---|---|
| DRIAMS-A | `data/driams/downloads/DRIAMS-A/` | `data/driams/preprocessing/merged_metadata.csv` | `data/driams/processed_data/spectra/` |
| DRIAMS-B | `data/driams/downloads/DRIAMS-B/` | `data/driams-B/preprocessing/merged_metadata.csv` | `data/driams-B/preprocessing/spectra/` |
| DRIAMS-C | `data/driams/downloads/DRIAMS-C/` | `data/driams-C/preprocessing/merged_metadata.csv` | `data/driams-C/preprocessing/spectra/` |
| DRIAMS-D | `data/driams/downloads/DRIAMS-D/` | `data/driams-D/preprocessing/merged_metadata.csv` | `data/driams-D/preprocessing/spectra/` |
| RKI | `data/RKI/downloads/RKI.4.0/` | `data/RKI/processed/rki_metadata.csv` | `data/RKI/processed_to_pt/spectra/` |

Use the original dataset versions referenced by `scripts/driams/download_all.sh` and `scripts/RKI/download.sh`; the latter points to RKI 4.0, Zenodo record 7702375. Raw spectra are not stored in Git. Download scripts describe the acquisition locations; inspect their versioned URLs rather than substituting the latest release.

Frozen split tensors are `data/driams/processed_data/species/{train,val,test}_fold_0.pt` through fold 6 and the equivalent `genera/` files. They can contain repeated accessions, one per spectrum; do not deduplicate the tensors. Publication evaluations use only folds 0–5, excluding fold 6 because its test split equals the tuning validation split. The frozen metadata supplies the taxonomy used to score the outputs. ETE3/NCBI regeneration against a newer database is a new annotation step, not a substitute for the publication snapshot.

## 3. Contrastive/cosine spectrum preprocessing

Use `conda_env` for mzML conversion and R/MALDIQuant. This is the DRIAMS-A example; use the matching raw and output directories for B/C/D.

```text
conda activate ./conda_env
python bin/process_data.py --input_file "data/driams/downloads/DRIAMS-A/raw/**/*.txt" --driams_csv data/driams/preprocessing/merged_metadata.csv --output_mzML_dir data/driams/raw/converted_to_mzml/driams-a --output_dir data/driams/preprocessing/baseline_corrected/driams
python bin/driams/merge_json_files.py --json_files data/driams/preprocessing/baseline_corrected/driams.json --output_file data/driams/preprocessing/baseline_corrected.json
conda activate ./ml_maldi_nn_conda_env
python bin/ml/preprocess.py --input_dir data/driams/preprocessing --metadata_file data/driams/preprocessing/merged_metadata.csv --output_dir data/driams/processed_data
```

The final command explicitly converts the merged `baseline_corrected.json` into individual Nx2 m/z–intensity `.pt` files, selecting the 200 most intense peaks. It avoids relying on the dataset loader's implicit conversion behavior. The R script is resolved relative to its source file, so running from the repository root works. Its `metadata.json` output is a copied CSV despite the filename; the model reads the original metadata CSV.

For an independent taxonomy regeneration, the current metadata command uses **`--input_csvs`**, not the stale `--input_csv` spelling in older shell scripts:

```text
python bin/driams/generate_metadata_file.py --input_csvs "data/driams/downloads/DRIAMS-A/id/**/*_clean.csv" --output_file data/driams/preprocessing/regenerated_metadata.csv
```

Do not replace the frozen merged metadata with this output. The code's MALDIQuant transforms include Savitzky–Golay half-window 20, SNIP 50, MAD peak half-window 10/SNR 4, bin tolerance 0.001, frequency filter 0.70, and m/z trimming 2000–20000. The methods audit records discrepancies with manuscript wording.

RKI's source conversion has a separate entry point:

```text
conda activate ./conda_env
python bin/process_rki.py --input_dir data/RKI/downloads/RKI.4.0 --output_csv_path data/RKI/processed/regenerated_metadata.csv --output_mzML_dir data/RKI/processed/mzML --output_dir data/RKI/processed/BaselineCorrected --n_jobs 8
```

Preserve the frozen `rki_metadata.csv`. Identify the resulting JSON filename before copying/merging it into `data/RKI/processed_to_pt/baseline_corrected.json` and using the explicit tensor converter. Older RKI wrappers contain duplicate `--output_dir` options and commented downloads; do not treat them as a one-command pipeline.

## 4. Separate MALDI Transformer preprocessing

The baseline uses the pinned `maldi-nn` preprocessing, not MALDIQuant tensors. The raw DRIAMS root must contain DRIAMS-A/B/C/D as expected by `bin/process_MALDI_Transformer_data.py`.

```text
python bin/process_MALDI_Transformer_data.py --input_raw_path data/driams/downloads --output_dir data/driams/processed_data/MaldiTransformer
python bin/resplit_maldi_transformer_data.py --input_h5torch_path data/driams/processed_data/MaldiTransformer/MaldiTransformer_peaks.h5torch --split_dir data/driams/processed_data/species --output_h5torch_dir data/driams/processed_data/MaldiTransformer_Genus_Labels/species --metadata_path data/driams/preprocessing/merged_metadata.csv --accessions
python bin/convert_maldi_transformer_to_pt.py --input_h5torch_path data/driams/processed_data/MaldiTransformer/MaldiTransformer_peaks.h5torch --output_pt_dir data/driams/processed_data/MaldiTransformer_Genus_Labels/species/spectra/all
```

Repeat the resplit/conversion for `genera` when preparing genus-disjoint baseline inputs. The baseline training files must come from `MaldiTransformer_Genus_Labels`, because the target is genus. Older unlabelled baseline launchers do not reproduce the final experiment. Independent-dataset and RKI conversions are described in their existing `preprocess_for_MaldiTransformer` scripts; preserve the paths in the inference configuration.

## 5. Publication checkpoints or optional retraining

Use the exact checkpoint inventory and SHA-256 manifest supplied with the curated release. The contrastive checkpoints are selected `best-checkpoint.ckpt` files, rather than duplicate last-epoch checkpoints. Baseline selection follows the checkpoint used by the saved inference run; do not automatically select a newer run just because it reached more steps.

To retrain one contrastive species-disjoint fold, run from `bin/ml`:

```text
cd bin/ml
python train.py --model_type CLIP_MALDI --dataset driams --target genera --split_method species --batch_size 32 --n_epochs 1000 --train_for_score --hparam_dir lightning_logs_driams_optuna/genera/species/CLIP_MALDI/CLIP_MALDI.json -k 0
```

For genus-disjoint training, change both `--split_method` and the hyperparameter JSON path to `genera`. Repeat for folds 0–5. Final hyperparameters were selected on DRIAMS. Older launchers refer to an absent IDBac JSON; the code falls back to the DRIAMS JSON for the requested target/split/model. Use an explicit DRIAMS path to remove that ambiguity.

The final training launchers use batch 32 and 1000 epochs. Training selects m/z 3000–20000, normalizes intensities, retains top 150 peaks, and pads with -1. Preserve the existing inference transforms too: their sequence differs from training. Frozen checkpoints are the route to reproduce the original outputs; retraining can differ because of random initialization and hardware.

The baseline command, also from `bin/ml`, is:

```text
python abstracted_train_malditransformer.py ../../data/driams/processed_data/MaldiTransformer_Genus_Labels/species/maldi_transformer_fold_0.h5torch ./MALDI-Transformer_Reproduction/k=0 M --batch_size 1024
```

For genus-disjoint training use the `genera` H5torch file and `MALDI-Transformer_Reproduction_Genus_Disjoint` log directory. The script sets the M architecture, top 200 peaks, p=0.15, lambda=0.01, and 500000 steps. Checkpoint history includes an earlier species fold-4 run; reproduce the checkpoint tied to the publication inference rather than silently replacing it.

## 6. Inference and metric notebook

The submission inference runner enumerates exact checkpoints, folds, datasets, and helper-compatible output paths. From the repository root, list commands with `python scripts/publication/inference.py`; run them with `python scripts/publication/inference.py --execute`. The runner launches each command from `bin/ml`. Inspect its command listing before execution. No saved embedding bundle is required: inference generates the feather tables used by the metric helpers.

For DRIAMS-A contrastive species-disjoint inference, `--version 0 --new_paths` selects the original score directory and avoids changing the expected DRIAMS-A output layout:

```text
python bare_inference.py --model CLIP_Transformer --dataset DRIAMS-A --target genera --split_type species --inference_set train --run_for_score --new_paths --version 0 -k 0
python bare_inference.py --model CLIP_Transformer --dataset DRIAMS-A --target genera --split_type species --inference_set test --run_for_score --new_paths --version 0 -k 0
```

Repeat train/test for folds 0–5 and `split_type genera`. Baseline inference uses `--model MaldiTransformerWrapper --maldi_nn_preprocessing --checkpoint_path` with the exact checkpoint. Independent DRIAMS-B/C/D and RKI evaluation uses all spectra and the six DRIAMS-A-trained checkpoints. Cosine models use no checkpoint; intensity-agnostic comparisons are derived by the metric helpers. Output paths must match the helper's dataset-specific layout, including baseline `inference/DRIAMS-A/genera/species` and independent-dataset `inference/<dataset>/all` directories. The legacy explicit-checkpoint path behavior does not produce every helper path directly; this is addressed by the curated inference runner, not by changing the metric formulas.

Open the curated notebook from `scratch` after inference:

```text
cd ../../scratch
jupyter lab publication_metrics.ipynb
```

The notebook documents the original metric calls and writes to `regenerated_publication_figures/`. Binary-pair metrics can require substantial memory/disk/CPU because the paper uses exhaustive pairs. Preserve macro/micro averaging, balanced/unbalanced pairs, species filters, seeds, and folds exactly. Use the frozen `publication_figures/` directory for comparison.

## 7. Banana-slug application

For figure reproduction, install the analysis packages and run from `publication/banana_slug/notebooks`:

```text
jupyter lab banana_slug_publication.ipynb
```

The notebook reads the included enriched IDBac hit table, fixes the original six normalized GTDB genus labels, applies distance <=0.232, and regenerates top-10/all-hit plots, precision@k, and pooled threshold precision. It does not invoke genome QC, BLAST, GTDB, or a live search. Original outputs are preserved in `figures/fig6_iterations/`; new outputs go to `regenerated_figures/fig6_iterations/`.

For genome-analysis provenance see `publication/banana_slug/README.md`, its source scripts and environment, the frozen summary tables, and `PROVENANCE.md`. The preserved full presentation notebook requires the included tables and has schematic exploratory sections; the focused notebook contains the data-driven publication calculations. The two GNPS2 task links and external IDBac workflow/model-weight repository document remote-search provenance. Remote task parameter/checkpoint metadata has not been independently inspected here. The separate 47-strain experiment remains a source-analysis gap.
