# IDBac 47-strain WGS search evaluation

This directory contains the inputs, selected source sections, saved figure panels, and focused notebook for supplementary Figure 6. The active source analysis is `data/ML_Queries_Round_1/results.ipynb`, but it reads the **combined `ML_Queries_Full` dataset**, rather than the earlier Round 1 or Round 2 search tables.

The 50-row query metadata excludes three uncertain annotations, leaving exactly the 47 IDBac identifiers in the manuscript's supplementary strain roster. The saved top-1 output has 32 correct transformer annotations and 26 correct intensity-agnostic cosine annotations. Genus-weighted top-1 accuracies are 0.3874509804 and 0.1988235294, matching the manuscript's 38.7% and 19.9%.

## Reproduce the figure and metrics

From the repository root, install the analysis dependencies and open the focused notebook:

```text
python -m pip install -r publication/requirements-analysis.txt
cd publication/idbac_wgs/notebooks
jupyter lab idbac_wgs_publication.ipynb
```

The notebook reads the frozen search tables and WGS genus labels, excludes near-zero distances and the three uncertain queries, and calculates top-k accuracy for k = 1–5. Missing predictions contribute zero; the denominator remains k. Standard averaging weights strains equally. Macro averaging first averages strains within each genus, then weights genera equally. New CSV, PNG, and SVG outputs go to `regenerated_results/`.

The 16S rRNA comparison, BLAST inputs, query-query PR curve, exploratory recall plots, and earlier search rounds are omitted. The existing metadata retains auxiliary taxonomy annotations used by the source's database-genus normalization; these are not evaluated as a separate method.

## Contents

- `data/`: combined transformer and cosine search results, query metadata, knowledgebase annotation snapshot, strain-name mapping, saved top-1 output, and the manuscript's 47-strain roster.
- `notebooks/idbac_wgs_publication.ipynb`: focused analysis with relative paths and checks against the saved top-1 values.
- `notebooks/source_sections.json`: selected original source cells for comparison; this is provenance rather than an executable notebook.
- `figures/`: original supplementary panels extracted from the source notebook's saved outputs.
- `workflow_snapshot/`: relevant inference/search source and the active ONNX model, checked against the production deployment. This is a provenance snapshot, not a standalone complete Nextflow distribution. The complete workflow is maintained upstream.
- `validation.json`: static comparison of the saved top-1 results and manuscript roster.
- `deployment_verification.json`: production workflow version and byte-for-byte source/model checks.
- `source_inventory.json`: source paths and hashes for copied inputs and workflow files.

See [PROVENANCE.md](PROVENANCE.md) for workflow/model selection and the remaining methods caveats.
