## Synthetic NLQ augmentation (narrations → NLQ-like queries)

This repository provides a pipeline to create NLQ-like synthetic supervision from Ego4D narrations.

### Artifacts included
- `pretrain_catalog_v2_multi.json`: structured catalog + metadata (`meta`, `clips`).
- `pretrain_catalog_v2_multi_flat.json`: flat records with timestamps, `matched_template`, and narration context.

### Pipeline steps
1) **Create catalog DB** (candidate narration windows)
   - Script: `create_catalog_db_v2.py`
   - Output: a DB containing candidate windows + duration buckets + template metadata.
   - Note: supports `--geom_only` for geometry-only runs.

2) ** Add density scores**
   - Script: `update_density_db.py`
   - Output: updated DB with density scores for sampling.

3) **Anchor–Fill sampling (match train statistics)**
   - Script: `generate_clips_anchor.py`
   - Reads: NLQ train annotations (TRQ) + catalog DB.
   - Writes: flat list of synthetic windows (GEQ-like), ready for query generation.

4) **LLM query generation (template-conditioned + constrained)**
   - Script: `query_generation.py`
   - Uses: prompt templates + optional GBNF grammars + validators to ensure template compliance.
   - Writes: augmented dataset (window + generated question), ready for pretraining.

5) **Analysis / validation**
   - Script: `analyze_multi_clip.py`
   - Produces: duration ECDF/hist, qs_rel/qe_rel distributions, relative coverage ECDF + KS.

### Figures
See `pretrain_catalog/` for plots comparing train vs generated:
- Template distribution
- Duration distribution + ECDF
- qs_rel / qe_rel per bucket
- Relative coverage ECDF (per bucket) + KS
- Template×bucket matching
