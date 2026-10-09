# Provenance and version selection

## Source analysis and results

Source checkout: `/home/user/LabData/michael_s/SourceCode/Workflows/IDBac_Analysis_Workflow`, Git commit `330e529eb97daee24bd3f4e2fb778dc5247c3e8f`. The notebook and query datasets were untracked working files. The active ONNX model and R preprocessing script differed from tracked Git versions; the copied snapshot therefore preserves the actual working files, rather than reconstructing them from the Git commit.

The relevant source notebook is `data/ML_Queries_Round_1/results.ipynb`, saved July 21, 2026. Its active input paths point to `data/ML_Queries_Full/`. Earlier Round 1/Round 2 data and the unrelated `Scratch/Manuscript_Search_Analysis/plotting.ipynb` are not used in this evaluation.

Frozen combined search tables correspond to these task identifiers, as recorded in their source filenames:

- [Transformer: fbb91febffc54470b91d3b84e1ab7293](https://gnps2.org/status?task=fbb91febffc54470b91d3b84e1ab7293).
- [Intensity-agnostic cosine: fedcb489d6f7441eb738260b4bb352ab](https://gnps2.org/status?task=fedcb489d6f7441eb738260b4bb352ab).

Both enriched search tables have source modification times on April 2, 2026. The source notebook's cells 23 and 26 provide the original standard and genus-weighted figure panels. Their first PNG outputs are preserved in `figures/`. The saved `gemini_first_output.csv` supplies the 47-row top-1 reference table; its name is preserved for traceability.

Static comparison establishes an exact match between the 47 saved query identifiers and the manuscript roster. It also establishes 32 versus 26 correct top-1 predictions and genus-weighted accuracies 38.745098% versus 19.882353%. No remote search, inference, training, or notebook execution was performed during this curation.

## Production workflow and model

The production GNPS2 directory `/data/nf_data/server/env/workflows/idbac_analysis_workflow` was inspected read-only on October 9, 2026. Its `workflowinput.yaml` reports version **2026.04.23.00**. The local working source matches the deployed inference, preprocessing, transform, search, and utility files byte for byte. The copied active model `CLIP_Transformer.onnx` also matches production:

```text
SHA-256: 49083116475560b806ae957c2b392b252e8cd33733a26ee5f0986f466e89f1c9
Size: 18726708 bytes
Source and deployment modification time: April 1, 2026
```

The older `CLIP_Transformer_1_28_2026.onnx` and `CLIP_Transformer_old_on_4_1_2026.onnx` files have different hashes and are excluded. The preserved inference code embeds each replicate separately and takes the mean embedding, matching the manuscript's description. Machine-readable checks are in `deployment_verification.json`.

These checks establish consistency with the inspected production deployment. They do not independently establish the exact historical model/database snapshot used by each April search: task-page requests returned HTTP 403 and the task directories were not present at the expected production paths. The frozen search tables remain the authoritative inputs for reproducing the reported evaluation. A new live query against a changed knowledgebase is a separate analysis.

The complete workflow is maintained at [Wang-Bioinformatics-Lab/IDBac_Analysis_Workflow](https://github.com/Wang-Bioinformatics-Lab/IDBac_Analysis_Workflow). Copied workflow files retain their upstream authorship; this repository's license does not override terms applying to external software or datasets.

## Preserved scoring and methods caveats

- Three uncertain annotations are excluded: `01KA9R5KF1QKAGN54SDPMNK7SC`, `01KA9R4XXSNR6R8Z38DAYG2YHE`, and `01KA9R753Q02A6K9AC8F0RRNRW`.
- Query labels come from WGS taxonomy, including the source's manual Rothia reassignment. Database missing-genus normalization follows the original metadata fallbacks.
- Self-match exclusion in the source is implemented as `distance > 1e-4`, rather than an identifier comparison. The focused notebook preserves this behavior. Static inspection confirms that all 50 excluded rows in each search table are identity matches, with no other rows excluded by the near-zero condition.
- The source writes the transformer cutoff as `1 - 0.232`. All frozen transformer distances are below 0.19, so the focused notebook's explicit distance cutoff 0.232 retains exactly the same rows. The cosine cutoff remains 0.65.
- Missing predictions count as incorrect for every included strain. The saved top-1 table has 40 strains with transformer predictions and 29 with cosine predictions after filtering. The source's pre-cohort result inspection reports 42 transformer query filenames with hits; this is a different count from the final scored cohort.
- Seven included queries have no other same-genus knowledgebase member under the source's annotation/count convention. This differs from the manuscript wording that all queries had potential correct hits; these queries remain in the denominator, consistent with the preserved source analysis.
- The dashed theoretical-reference curves preserve the source's `count_others_in_kb >= k - 1` convention. That condition can overstate the available-hit ceiling; it does not affect either measured accuracy curve. It is retained for original-figure comparison and is not independently validated as a theoretical bound.
