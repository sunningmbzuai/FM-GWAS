# Gene-Level Association Study via Genomic Foundation Model in the Human Phenotype Project
FM-based gene-level association study on Human Phenotype Project (HPP).

## 1. Data and model user need to prepare

- **Cohort file** (one per trait): must contain
  `participant_id`, `sex` (0=F, 1=M), `age`, and `label` (the trait's
  phenotype/outcome column — continuous for linear regression, binary for
  logistic regression, auto-detected).
- **Gene raw sequence file** (one `<gene_id>.tsv` per candidate gene under
  `--input_path` in step 3a): must contain `ref_seq`, a pair of
  mutation/variant sequence columns (`mutation_seq_1`/`_2`,
  `mutated_seq_1`/`_2`, or `variant_seq_1`/`_2`), and a subject identifier
  column (`participant_id`).
- **GeneUNet model**: download compiled .pt2 file from gcloud, and put it under `TransUNetWithTracks/compiled_A100/`, next to `aido_dna3ag_binary.py`

## 2. Conda environment setup (dynamic-length model)

Gene embedding extraction uses a compiled AOTInductor build of the
`TransUNetWithTracks` (AIDO.DNA3-AG) model,
`TransUNetWithTracks/compiled_A100/aido_dna3ag_1bp4_dyn_a100.pt2` — a
genuinely dynamic-length build (batch 1-32, length any multiple of 128 in
[256, 128000]) that runs on NVIDIA A100 (sm_80).

Build the env with the provided script:

```bash
bash TransUNetWithTracks/compiled_A100/setup_hpp_dynlen_a100_env.sh
```

This creates conda env `hpp_dynlen_a100` (python 3.10) and installs:
- `torch==2.13.0` (+cu130), `transformers==4.57.1`, `einops`, `pandas`, `ninja`
- `flash-attn==2.8.3.post1`, built **from source** scoped to
  `TORCH_CUDA_ARCH_LIST="8.0"` (no prebuilt wheel exists for this torch/CUDA
  pairing; building unscoped would also compile sm_90/100/120 kernels for
  4x the work)

See
`TransUNetWithTracks/compiled_A100/README.md` for full details.

## 3. HPP candidate significant genes per trait

Per-trait, sex-stratified and full-cohort candidate gene lists (from
`FM_gene_scores/`, filtered to `p_F_analytic < 2.5e-6` and deduplicated on
`gene_id`) live under:

```
HPP_union_gene/{trait}_extreme.tsv     # full cohort
HPP_union_gene/{trait}_extreme_F.tsv   # female-only
HPP_union_gene/{trait}_extreme_M.tsv   # male-only
```

`{trait}` is one of: `Cholesterol`, `Height`, `Hyperlipidemia`,
`Hypertension`, `Obesity`, `Osteoporosis`, `T2D`, `VAT`.

Each file has one row per significant gene, `gene_id` is an Ensembl gene ID
(ENSG, versioned, e.g. `ENSG00000000419.8`). Sex/cohort combinations with no
gene passing the threshold are omitted (e.g. no `Hypertension_extreme_M.tsv`
or `Obesity_extreme_M.tsv`).

The union gene list used to drive embedding extraction (step 3a below) is at
`HPP_union_gene/candidate_genes.tsv` — 862 unique `gene_id`s, the union
across all files above.

## 4. Gene-level association (FM-GWAS)

Two stages: (a) extract gene embeddings with the compiled model, (b) run the
permutation-based association test per trait.

### 4a. Extract gene embeddings

```bash
conda run -n hpp_dynlen_a100 python get_hpp_embedding-AG_clean.py \
  --model_path TransUNetWithTracks \
  --compiled_binary TransUNetWithTracks/compiled_A100/aido_dna3ag_1bp4_dyn_a100.pt2 \
  --gpu A100 \
  --input_path <gene_raw_sequence_dir> \
  --output_path <embedding_output_dir> \
  --gene_annotation_file HPP_union_gene/candidate_genes.tsv \
  --world_size <N> --rank <r>
```

- `--gene_annotation_file` only needs a `gene_id` column (e.g.
  `HPP_union_gene/candidate_genes.tsv`)
- `--input_path` must contain one `<gene_id>.tsv` per candidate gene (see
  "Gene raw sequence file" above).
- `--world_size`/`--rank` shard the gene list across parallel jobs; run one
  process per rank (`0..world_size-1`).
- Output: `<embedding_output_dir>/<gene_id>/<gene_id>.pt` (aggregated
  per-gene embeddings), consumed directly by `insample_hpp_AG.py` in 3b.

### 4b. Run association per trait

```bash
conda run -n hpp_dynlen_a100 python insample_hpp_AG.py <trait> <sex> <rank> \
  --world_size <N> --cohort extreme \
  --feature_root <embedding_output_dir> \
  --gene_file HPP_union_gene/<trait>_extreme[_F|_M].tsv \
  --cohort_file <cohort_tsv_for_trait> \
  --save_path results_insample_UKBB/<trait>_extreme_<sex>
```

- `<trait>`: e.g. `Height`, `VAT`, ... (matches `HPP_union_gene` file names)
- `<sex>`: `F`, `M`, or `All`
- `<rank>`/`--world_size`: shard the candidate gene list the same way as 3a;
  run one process per rank
- `--feature_root`: the `<embedding_output_dir>` from step 3a (contains
  `<gene_id>/<gene_id>.pt`)
- `--gene_file`: candidate gene list for this trait/sex, from
  `HPP_union_gene/` (e.g. `HPP_union_gene/Height_extreme_F.tsv` when
  `<sex>=F`) — only needs a `gene_id` column, same as `candidate_genes.tsv`
  in 3a (the script no longer filters by `gene_len`)
- `--cohort_file`: cohort tsv for `<trait>` (see below)
- `--save_path`: output directory for per-gene result tsv files
- Output: `<save_path>/<gene_id>.tsv` (per-gene `F_obs`/`p_F_analytic` rows
  across embedding types — this is the format `FM_gene_scores/` files are
  already in)

## 5. Replication check: HPP findings vs. UKBB external cohort

Run the same pipeline (step 4, both 4a and 4b) on an independent UKBB
cohort — same candidate gene list, your own UKBB `--cohort_file` and gene
raw sequence files (see "Inputs you need to prepare" above) — to get
per-gene `F_obs`/`p_F_analytic` results in the same shape as
`FM_gene_scores/`, e.g. under `results_insample_UKBB/<trait>_extreme_<sex>/`.

From those, build UKBB significant-gene lists the exact same way
`HPP_union_gene/` was built from `FM_gene_scores/` in step 3: filter to
`p_F_analytic < 2.5e-6`, deduplicate on `gene_id`, one file per
trait/setting, mirroring the naming convention:

```
UKBB_union_gene/{trait}_extreme.tsv     # full cohort
UKBB_union_gene/{trait}_extreme_F.tsv   # female-only
UKBB_union_gene/{trait}_extreme_M.tsv   # male-only
```

Each `UKBB_union_gene/{trait}_extreme[_F|_M].tsv` is the union of
significant genes for that trait, across the same per-setting significance
test used for HPP — i.e. the UKBB analogue of the corresponding
`HPP_union_gene/{trait}_extreme[_F|_M].tsv`.

**Comparison:** for each trait/setting, an HPP-significant gene (from
`HPP_union_gene/{trait}_extreme[_F|_M].tsv`) is **recovered** if its
`gene_id` also appears in the matching
`UKBB_union_gene/{trait}_extreme[_F|_M].tsv`. A high recovery rate
(`|HPP ∩ UKBB| / |HPP|`) for a trait/setting is evidence that the HPP
finding replicates in an independent cohort rather than being an HPP-only
artifact; genes present in HPP but absent from UKBB are the ones that did
not replicate.

