# Gene-Level Association Study via Genomic Foundation Model in the Human Phenotype Project

FM-GWAS: a foundation-model-based gene-level association study on the Human
Phenotype Project (HPP), with replication in UK Biobank (UKBB).

## About GB.DNA3

Gene embeddings in FM-GWAS are computed with **GB.DNA3**, GenBio AI's DNA
foundation model. For GenBio's open-source projects, see
[GenBio AI on GitHub](https://github.com/genbio-ai).

GB.DNA3 is a proprietary model of GenBio AI. **It is not distributed with this
repository**: this repo contains no GB.DNA3 code, configuration files,
tokenizer or weights. Access to GB.DNA3 is provided through GenBio AI. Users
with access supply the model checkpoint (or its compiled binary) at run time
through the command-line arguments described in step 4a.

This repository contains only the FM-GWAS analysis code:

- `extract_gene_embedding.py` — extracts per-gene, per-participant embeddings
  with GB.DNA3 (step 4a).
- `association_test.py` — gene-level association test (analytic partial F
  test, and logistic LRT for binary traits) on the embeddings (step 4b).

## 1. Data and model the user needs to prepare

- **Cohort file** (one per trait): must contain
  `participant_id`, `sex` (0=F, 1=M), `age`, and `label` (the trait's
  phenotype/outcome column — continuous for linear regression, binary for
  logistic regression, auto-detected).
- **Gene raw sequence file** (one `<gene_id>.tsv` per candidate gene under
  `--input_path` in step 4a): must contain `ref_seq`, a pair of
  mutation/variant sequence columns (`mutation_seq_1`/`_2`,
  `mutated_seq_1`/`_2`, or `variant_seq_1`/`_2`), and a subject identifier
  column (`participant_id`).
- **GB.DNA3 model**: obtained from GenBio AI (not included in this repo; see
  "About GB.DNA3" above). Either a model checkpoint directory (passed as
  `--model_path`) or a pre-compiled AOTInductor `.pt2` binary together with
  its loader (passed as `--compiled_binary`).

## 2. Environment

Gene embedding extraction was run on NVIDIA A100 GPUs (sm_80) with:

- python 3.10
- `torch==2.13.0` (+cu130), `transformers==4.57.1`, `einops`, `pandas`, `ninja`
- `flash-attn==2.8.3.post1`, built **from source** scoped to
  `TORCH_CUDA_ARCH_LIST="8.0"` (no prebuilt wheel exists for this torch/CUDA
  pairing; building unscoped would also compile sm_90/100/120 kernels for
  4x the work)

The association test (step 4b) only needs `numpy`, `pandas`, `scipy`,
`scikit-learn`, `torch` (to read the `.pt` embedding files) and optionally
`tqdm`.

The commands below assume a conda env named `fmgwas`; replace it with your
own env name.

## 3. Candidate gene lists per trait

HPP results and gene lists are not distributed with this repository. The
pipeline expects candidate gene lists that you build from your own
association results (filter to `p < 2.5e-6` and deduplicate on `gene_id`),
one per trait and cohort setting:

```
HPP_union_gene/{trait}_extreme.tsv     # full cohort
HPP_union_gene/{trait}_extreme_F.tsv   # female-only
HPP_union_gene/{trait}_extreme_M.tsv   # male-only
```

The traits analysed in the study are `Cholesterol`, `Height`,
`Hyperlipidemia`, `Hypertension`, `Obesity`, `Osteoporosis`, `T2D` and `VAT`.

Each file has one row per gene; only a `gene_id` column is required, holding
a versioned Ensembl gene ID (e.g. `ENSG00000000419.8`). Sex/cohort
combinations with no gene passing the threshold can be omitted.

The union of all these lists, `HPP_union_gene/candidate_genes.tsv` (also just
a `gene_id` column), drives embedding extraction in step 4a.

## 4. Gene-level association (FM-GWAS)

Two stages: (a) extract gene embeddings with GB.DNA3, (b) run the
association test per trait.

### 4a. Extract gene embeddings

```bash
conda run -n fmgwas python extract_gene_embedding.py \
  --model_path <GB.DNA3_checkpoint_dir> \
  --compiled_binary <GB.DNA3_compiled_binary.pt2> \
  --gpu A100 \
  --input_path <gene_raw_sequence_dir> \
  --output_path <embedding_output_dir> \
  --gene_annotation_file HPP_union_gene/candidate_genes.tsv \
  --world_size <N> --rank <r>
```

- `--model_path`: GB.DNA3 checkpoint directory provided by GenBio AI (also
  used to load the tokenizer).
- `--compiled_binary` (optional): compiled GB.DNA3 `.pt2` binary provided by
  GenBio AI. If omitted, the model is loaded from `--model_path`.
- `--compiled_loader` (optional, default `gb_dna3_binary`): name of the
  loader module provided by GenBio AI together with the compiled binary; it
  must sit in the same directory as the `.pt2` file.
- `--gene_annotation_file` only needs a `gene_id` column (e.g.
  `HPP_union_gene/candidate_genes.tsv`).
- `--input_path` must contain one `<gene_id>.tsv` per candidate gene (see
  "Gene raw sequence file" above).
- `--world_size`/`--rank` shard the gene list across parallel jobs; run one
  process per rank (`0..world_size-1`).
- Output: `<embedding_output_dir>/<gene_id>/<gene_id>.pt` (aggregated
  per-gene embeddings), consumed directly by `association_test.py` in 4b.

### 4b. Run association per trait

```bash
conda run -n fmgwas python association_test.py run <trait> <sex> \
  --rank <rank> --world_size <N> \
  --feature_root <embedding_output_dir> \
  --gene_tsv HPP_union_gene/<trait>_extreme[_F|_M].tsv \
  --cohort_file <cohort_tsv_for_trait> \
  --covariates <covariates_list> \
  --gene_pca variance:0.95 \
  --tests LRT \
  --out_dir results/<trait>_extreme_<sex>
```

- `<trait>`: e.g. `Height`, `VAT`, ... (matches `HPP_union_gene` file names)
- `<sex>`: `F`, `M`, or `All`
- `<rank>`/`--world_size`: shard the candidate gene list the same way as 4a;
  run one process per rank
- `--feature_root`: the `<embedding_output_dir>` from step 4a (contains
  `<gene_id>/<gene_id>.pt`)
- `--gene_tsv`: candidate gene list for this trait/sex, from
  `HPP_union_gene/` (e.g. `HPP_union_gene/Height_extreme_F.tsv` when
  `<sex>=F`) — only needs a `gene_id` column, same as `candidate_genes.tsv`
  in 4a
- `--cohort_file`: cohort tsv for `<trait>` (see step 1)
- `--covariates`: `age,sex` for the `All` cohort, and `age` for the `F` and
  `M` cohorts
- `--tests`: `F`, `LRT`, or `F,LRT` (LRT applies to binary 0/1 labels only)
- `--out_dir`: output directory for per-gene result files
- Output: `<out_dir>/per_gene/<gene_id>.tsv` (per-gene `p_F` / `p_LRT` rows
  across embedding types and PCA settings)

After all ranks finish, merge the per-gene results and add BH q-values:

```bash
conda run -n fmgwas python association_test.py merge \
  --out_dir results/<trait>_extreme_<sex>
```

This writes `<out_dir>/ftest_results_merged.tsv` (and `missing_genes.tsv`).
Run `python association_test.py run -h` for the full list of options
(cell-embedding covariates, PCA settings, etc.).

## 5. Replication check: HPP findings vs. UKBB external cohort

Run the same pipeline (step 4, both 4a and 4b) on an independent UKBB
cohort — same candidate gene list, your own UKBB `--cohort_file` and gene
raw sequence files (see step 1) — e.g. with
`--out_dir results_UKBB/<trait>_extreme_<sex>`.

From those results, build UKBB significant-gene lists the exact same way
`HPP_union_gene/` was built in step 3: filter to `p < 2.5e-6`, deduplicate
on `gene_id`, one file per trait/setting, mirroring the naming convention:

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

## Acknowledgements

GB.DNA3 was developed by [GenBio AI](https://github.com/genbio-ai). We thank
GenBio AI for providing access to the model.
