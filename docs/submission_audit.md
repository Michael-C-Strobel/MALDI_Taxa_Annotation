# Submission audit

## Required provenance

- Locate the source notebook/scripts and result tables for the separate 47-strain WGS/IDBac requery experiment. A roster in the manuscript is not enough to reproduce its metrics.
- Preserve the banana-slug search table as the figure-level reproduction input. The original ML search task/checkpoint has not yet been identified, so rerunning the remote search end to end is not yet documented.
- Figure 6B mirror-plot source is not located; this is an optional provenance gap under the agreed scope.

## Methods wording to review with the manuscript authors

No manuscript edits or scientific transformations are made to resolve these discrepancies.

- `bin/preprocess_data.R` uses MAD peak detection half-window 10; the manuscript describes 50.
- Final contrastive training installs checkpoint selection callbacks, but does not install the imported EarlyStopping callback. The final launchers specify 1000 epochs; the hyperparameter JSON's N_EPOCHS value describes the tuning configuration.
- The binary-curve CSV variants matching the supplementary PR-AUC tables retain singleton genera, although the methods describe a multi-species genus restriction.
- Contrastive training and inference use different transform sequences in the preserved implementation. Document both; changing them would change the experiment.
- Saved Optuna studies contain 311 species and 256 genus trials; the manuscript refers to 300. Check trial states and intended reported cohort before interpreting this as completed-trial counts.
- Banana-slug figure scoring uses six fixed normalized GTDB genus labels. Preserve these labels and the 16S evidence separately rather than replacing them with a new taxonomy call.

## Validation scope

Static inspection only for this curation: syntax, CLI arguments, relative paths, notebook structure, exact artifact hashes, checkpoint references, and output-to-manuscript mapping. Training, inference, metrics, remote search, and genome workups are not rerun. Earlier metric-selection tests passed, but they do not establish the reported numeric results; the frozen CSV/manuscript comparison supplies that evidence.

The original `scratch/publication_figures.ipynb` has an unfinished exploratory cell 167. It is preserved as provenance; the focused `publication_metrics.ipynb` omits the exploratory tail and passes code-cell syntax checks.
