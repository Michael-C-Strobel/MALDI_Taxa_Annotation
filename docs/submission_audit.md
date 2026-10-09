# Submission audit

## Required provenance

- The 47-strain WGS/IDBac evaluation is included in `publication/idbac_wgs/`. Frozen hit tables and saved top-1 output match the manuscript roster and reported accuracies. The copied active model/source matches production version 2026.04.23.00; exact historical task model/database metadata remains unverified.
- Banana-slug remote-search provenance is documented through the two GNPS2 tasks cited in the manuscript and linked in `publication/banana_slug/PROVENANCE.md`. The frozen search table supports local figure reproduction. Automated task-page retrieval returned HTTP 403, so remote parameter/checkpoint metadata has not been independently verified here.
- Figure 6B mirror-plot source is not located; this is an optional provenance gap under the agreed scope.

## Methods wording to review with the manuscript authors

These observations compare the manuscript methods with the preserved source implementation and outputs.

- `bin/preprocess_data.R` uses MAD peak detection half-window 10; the manuscript describes 50.
- Final contrastive training installs checkpoint selection callbacks, but does not install the imported EarlyStopping callback. The final launchers specify 1000 epochs; the hyperparameter JSON's N_EPOCHS value describes the tuning configuration.
- The binary-curve CSV variants matching the supplementary PR-AUC tables retain singleton genera, although the methods describe a multi-species genus restriction.
- Contrastive training and inference use different transform sequences in the preserved implementation. Both sequences are retained in the reproduction workflow.
- Saved Optuna studies contain 311 species and 256 genus trials; the manuscript refers to 300. These totals include saved trial records; they do not by themselves establish the number of completed trials.
- Banana-slug figure scoring uses six fixed normalized GTDB genus labels. The scoring labels and 16S evidence are supplied separately in the frozen workup tables.
- The 47-strain source uses a near-zero-distance exclusion for self matches and has seven queries without another same-genus knowledgebase member, differing from the manuscript statement that all queries had potential correct hits. Its pre-cohort 42-query hit count differs from the final 40 transformer queries with predictions. The historical theoretical-reference curve uses a `k - 1` condition that can overstate its ceiling. These conventions are documented in `publication/idbac_wgs/PROVENANCE.md`; measured accuracy scoring is preserved.

## Validation scope

Static inspection only for this curation: syntax, CLI arguments, relative paths, notebook structure, exact artifact hashes, checkpoint references, and output-to-manuscript mapping. Training, inference, metrics, remote search, and genome workups are not rerun. Earlier metric-selection tests passed, but they do not establish the reported numeric results; the frozen CSV/manuscript comparison supplies that evidence.

The original `scratch/publication_figures.ipynb` has an unfinished exploratory cell 167. It is preserved as provenance; the focused `publication_metrics.ipynb` omits the exploratory tail and passes code-cell syntax checks.
