# Manuscript items and preserved results

This mapping describes the audited publication outputs. It does not modify the manuscript or regenerate any measurements.

| Manuscript item | Repository source | Interpretation |
|---|---|---|
| Dataset/fold counts | Frozen metadata and `data/driams/processed_data/{species,genera}/*_fold_*.pt`; inference accessions | Counts depend on the evaluated cohort; do not substitute raw metadata totals. |
| Figures 1–2; supplementary PR-AUC table 1 | `scratch/publication_figures/fig/binary_curves/` | All-pair and cross-species PR/ROC outputs; select the matching balanced/unbalanced variant. |
| Figure 3 | `scratch/publication_figures/fig/top_k_recall/` | Cross-species, within-test top-k recall; macro and micro outputs. |
| Figure 4 | `scratch/publication_figures/data/nn_accuracy/` and `fig/nn_accuracy/` | Sparse retrieval for k=1–10; ten sampling seeds and six folds for trained models. |
| Figure 5; supplementary PR-AUC table 2 | `scratch/publication_figures/fig/generalization_binary_curves/` and `fig/genus_disjoint_generalization_binary_curves/` | DRIAMS-A/RKI species and genus generalization. |
| Figure 6A | `publication/banana_slug/notebooks/figures/fig6_iterations/tiles_top10_with_precision_gauge.png` | Ranked top-10 IDBac hits for six strains; source is `data/idbac/enriched_db_results.tsv`. |
| Figure 6B | Source mirror plot not located | Non-gating figure provenance gap; no replacement plot is invented. |
| Supplementary banana-slug hit tiles | `publication/banana_slug/notebooks/figures/fig6_iterations/tiles_all_hits_with_precision_gauge.png` | All hits at distance <=0.232. |
| Supplementary banana-slug threshold precision | `publication/banana_slug/notebooks/figures/fig6_iterations/threshold_vs_precision.png` | Pooled precision across six hit-bearing strains; 185 hits and precision 0.886 at 0.232. |
| Supplementary Optuna importance/loss plots | `scripts/publication_optuna/generate_optuna_publication_assets.py`; frozen DRIAMS studies | Frozen outputs in `figures/publication_optuna/`; separate species/genus studies. Use saved studies instead of rerunning tuning. |
| Supplementary cosine bin-size ablation | Publication notebook cosine-ablation section and frozen binary-curve outputs | 1, 3, 5, 7, and 10 Da bins. |
| Supplementary 47-strain WGS/IDBac evaluation | Source analysis/results not yet located | Required before claiming every supplementary experiment is reproducible. The manuscript roster alone is insufficient. |

The two supplementary PR-AUC tables were checked against the preserved CSVs: all 128 entries match to four decimals. Outputs without `_no_singleton_genera` match these tables. Keep those variants frozen while resolving the manuscript's singleton-filter wording.

## Cohort counts

| Dataset/cohort | Spectra | Genera | Species |
|---|---:|---:|---:|
| DRIAMS-A species-disjoint evaluation union | 103383 | 228 | 719 |
| DRIAMS-A genus-disjoint evaluation union | 103137 | 149 | 624 |
| DRIAMS-B | 5500 | 78 | 235 |
| DRIAMS-C | 4696 | 50 | 116 |
| DRIAMS-D | 10436 | 20 | 48 |
| RKI | 11055 | 73 | 232 |

The species-disjoint cohort contains 246 additional spectra across 79 additional genera and 95 additional species relative to the genus-disjoint cohort. The additional genera have only 1–7 spectra each. Sparse genera are retained in species-disjoint training; the archived genus-disjoint splits exclude these records. Do not change the splits to reconcile the counts.
