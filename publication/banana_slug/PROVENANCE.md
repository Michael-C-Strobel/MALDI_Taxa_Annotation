# Banana-slug application provenance

Copied from `Laura_Genome_Workups` on BotNet-VM, originally `/home/user/mstro016/SourceCode/Laura_Genome_Workups`, using the inspection snapshot retained at `/tmp/msystems-workups.kf58AD`. No analysis was rerun during repository curation.

Included material:

- Original genome-workup README, environment specification, scripts, and notebook builder.
- Original presentation notebook and rendered figures.
- Frozen 16S sequences, BLAST combined results, 16S annotations/clustering QC, CheckM2 summary, GTDB summary/raw bacterial summary, and enriched IDBac query hits.
- `notebooks/banana_slug_publication.ipynb`, extracted from original cells 1 and 28–32. Outputs are cleared in this derived notebook; its figure destination is `regenerated_figures/`, preserving the original PNGs.

The scoring labels are SL1003=Mucilaginibacter and SL1033, SL1072, SL1097, SL1102, SL1105=Pseudomonas, taken from normalized GTDB genera in the original plotting code. At distance <=0.232, the preserved source analysis reports 185 pooled hits with precision 0.886. Query grouping extracts the SL identifier from `query_filename`; correctness compares `db_genus` with the fixed scoring genus. Rank order follows the original sorting/filtering code and its handling of ties.

CheckM2 retains 11 of 17 assemblies: SL1002, SL1003, SL1004, SL1033, SL1046, SL1064, SL1072, SL1097, SL1102, SL1105, SL1107. Six of these strains return hits at the figure threshold. The original notebook also contains schematic figures with hard-coded prose counts; the derived publication notebook uses the saved hit table instead.

Spectral deposit: MassIVE MSV000101529. The manuscript identifies the banana-slug IDBac searches through these GNPS2 tasks:

- [Standard search: `6d495bdc11104d0986816ce57a82d39c`](https://gnps2.org/status?task=6d495bdc11104d0986816ce57a82d39c).
- [Intensity-agnostic cosine search: `224cf46807644952ba9dbbc9fb0611b4`](https://gnps2.org/status?task=224cf46807644952ba9dbbc9fb0611b4).

The manuscript links the [IDBac analysis workflow and model weights](https://github.com/Wang-Bioinformatics-Lab/IDBac_Analysis_Workflow). These task IDs provide the remote-search provenance; the included enriched hit table provides the input for local figure reproduction. Automated retrieval of the task pages returned HTTP 403 during this verification, so the remote task parameters and checkpoint metadata have not been independently checked here.

Large WGS assemblies, raw reads, local reference databases, intermediate protein/marker files, and duplicate zip archives are not needed for replotting the supplied tables. The included original workup README documents the genome-analysis workflow and input layout; rerunning those tools requires the deposited assemblies and the corresponding reference versions. Its live NCBI/GTDB outputs may differ from this frozen snapshot.
