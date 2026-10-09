# Manuscript items and preserved results

This mapping identifies the preserved source outputs for the manuscript figures, tables, and cohort counts.

| Manuscript item | Repository source | Interpretation |
|---|---|---|
| Dataset/fold counts | Frozen metadata and `data/driams/processed_data/{species,genera}/*_fold_*.pt`; inference accessions | Counts describe the evaluated cohorts represented by the saved splits and inference accessions, rather than the complete raw metadata table. |
| Figures 1–2; supplementary PR-AUC table 1 | `scratch/publication_figures/fig/binary_curves/` | All-pair and cross-species PR/ROC outputs, with balanced and unbalanced variants for macro and micro evaluation. |
| Figure 3 | `scratch/publication_figures/fig/top_k_recall/` | Cross-species, within-test top-k recall; macro and micro outputs. |
| Figure 4 | `scratch/publication_figures/data/nn_accuracy/` and `fig/nn_accuracy/` | Sparse retrieval for k=1–10; ten sampling seeds and six folds for trained models. |
| Figure 5; supplementary PR-AUC table 2 | `scratch/publication_figures/fig/generalization_binary_curves/` and `fig/genus_disjoint_generalization_binary_curves/` | DRIAMS-A/RKI species and genus generalization. |
| Figure 6A | `publication/banana_slug/notebooks/figures/fig6_iterations/tiles_top10_with_precision_gauge.png` | Ranked top-10 IDBac hits for six strains; source is `data/idbac/enriched_db_results.tsv`. |
| Figure 6B | Source mirror plot not located | The mirror-plot source is not included in the preserved analysis files. |
| Supplementary banana-slug hit tiles | `publication/banana_slug/notebooks/figures/fig6_iterations/tiles_all_hits_with_precision_gauge.png` | All hits at distance <=0.232. |
| Supplementary banana-slug threshold precision | `publication/banana_slug/notebooks/figures/fig6_iterations/threshold_vs_precision.png` | Pooled precision across six hit-bearing strains; 185 hits and precision 0.886 at 0.232. |
| Supplementary Optuna importance/loss plots | `scripts/publication_optuna/generate_optuna_publication_assets.py`; frozen DRIAMS studies | Frozen outputs in `figures/publication_optuna/`, derived from the saved species-disjoint and genus-disjoint studies. |
| Supplementary cosine bin-size ablation | Publication notebook cosine-ablation section and frozen binary-curve outputs | 1, 3, 5, 7, and 10 Da bins. |
| Supplementary 47-strain WGS/IDBac evaluation | `publication/idbac_wgs/` combined hit tables, focused notebook, and saved panels | The 47 query identifiers match the manuscript roster; saved top-1 counts match 32 versus 26, and genus-weighted accuracies match 38.7% versus 19.9%. The 16S comparison is excluded. |

All 128 entries in the two supplementary PR-AUC tables match the CSV variants **without** `_no_singleton_genera` to four decimal places. Among the filtered variants, 112 corresponding entries are available and none match the manuscript values; filtered counterparts are absent for the remaining 16 genus-disjoint all-pair entries. The row-by-row comparison is preserved in [`publication/manuscript_pr_auc_comparison.csv`](../publication/manuscript_pr_auc_comparison.csv), with the exact manuscript file hash in its accompanying summary.

## Cohort counts

The manuscript's species-disjoint fold table matches the preserved species-disjoint memberships. Its main dataset Table 1 lists the DRIAMS-A genus-disjoint cohort totals (103,137 spectra, 149 genera, 624 species). The species-disjoint evaluation union has separate totals, shown below.

| Dataset/cohort | Spectra | Genera | Species |
|---|---:|---:|---:|
| DRIAMS-A species-disjoint evaluation union | 103383 | 228 | 719 |
| DRIAMS-A genus-disjoint evaluation union | 103137 | 149 | 624 |
| DRIAMS-B | 5500 | 78 | 235 |
| DRIAMS-C | 4696 | 50 | 116 |
| DRIAMS-D | 10436 | 20 | 48 |
| RKI | 11055 | 73 | 232 |

The species-disjoint cohort contains 246 additional spectra across 79 additional genera and 95 additional species relative to the genus-disjoint cohort. The additional genera have only 1–7 spectra each. Sparse genera are retained in species-disjoint training; the preserved genus-disjoint splits exclude these records. These differences reflect the membership of the preserved split sets.
