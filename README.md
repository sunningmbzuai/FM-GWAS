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
through the command-line arguments described in step 3a.

This repository contains only the FM-GWAS analysis code:

- `extract_gene_embedding.py` — extracts per-gene, per-participant embeddings
  with GB.DNA3 (step 3a).
- `association_test.py` — gene-level association test (analytic partial F
  test, and logistic LRT for binary traits) on the embeddings (step 3b).

## 1. Data and model the user needs to prepare

- **Cohort file** (one per trait): must contain
  `participant_id`, `sex` (0=F, 1=M), `age`, and `label` (the trait's
  phenotype/outcome column — continuous for linear regression, binary for
  logistic regression, auto-detected).
- **Gene raw sequence file** (one `<gene_id>.tsv` per candidate gene under
  `--input_path` in step 3a): must contain `ref_seq`, a pair of
  mutation/variant sequence columns (`mutation_seq_1`/`_2`,
  `mutated_seq_1`/`_2`, or `variant_seq_1`/`_2`), and a subject identifier
  column (`participant_id`).
- **Candidate gene list(s)**: TSV files with a single required `gene_id`
  column (versioned Ensembl gene ID, e.g. `ENSG00000000419.8`). Use one list
  covering all genes to embed in step 3a, and one list per trait/cohort
  setting (`All`, `F`, `M`) to test in step 3b.
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

The association test (step 3b) only needs `numpy`, `pandas`, `scipy`,
`scikit-learn`, `torch` (to read the `.pt` embedding files) and optionally
`tqdm`.

The commands below assume a conda env named `fmgwas`; replace it with your
own env name.


## 3. Gene-level association (FM-GWAS)

Two stages: (a) extract gene embeddings with GB.DNA3, (b) run the
association test per trait.

### 3a. Extract gene embeddings

```bash
conda run -n fmgwas python extract_gene_embedding.py \
  --model_path <GB.DNA3_checkpoint_dir> \
  --compiled_binary <GB.DNA3_compiled_binary.pt2> \
  --gpu A100 \
  --input_path <gene_raw_sequence_dir> \
  --output_path <embedding_output_dir> \
  --gene_annotation_file <candidate_gene_list.tsv> \
  --world_size <N> --rank <r>
```

- `--model_path`: GB.DNA3 checkpoint directory provided by GenBio AI (also
  used to load the tokenizer).
- `--compiled_binary` (optional): compiled GB.DNA3 `.pt2` binary provided by
  GenBio AI. If omitted, the model is loaded from `--model_path`.
- `--compiled_loader` (optional, default `gb_dna3_binary`): name of the
  loader module provided by GenBio AI together with the compiled binary; it
  must sit in the same directory as the `.pt2` file.
- `--gene_annotation_file`: list of all genes to embed; only needs a
  `gene_id` column (see step 1).
- `--input_path` must contain one `<gene_id>.tsv` per candidate gene (see
  "Gene raw sequence file" above).
- `--world_size`/`--rank` shard the gene list across parallel jobs; run one
  process per rank (`0..world_size-1`).
- Output: `<embedding_output_dir>/<gene_id>/<gene_id>.pt` (aggregated
  per-gene embeddings), consumed directly by `association_test.py` in 3b.

### 3b. Run association per trait

```bash
conda run -n fmgwas python association_test.py run <trait> <sex> \
  --rank <rank> --world_size <N> \
  --feature_root <embedding_output_dir> \
  --gene_tsv <gene_list_for_trait_sex.tsv> \
  --cohort_file <cohort_tsv_for_trait> \
  --covariates <covariates_list> \
  --gene_pca variance:0.95 \
  --tests LRT \
  --out_dir results/<trait>_<sex>
```

- `<trait>`: e.g. `Height`, `VAT`
- `<sex>`: `F`, `M`, or `All`
- `<rank>`/`--world_size`: shard the candidate gene list the same way as 3a;
  run one process per rank
- `--feature_root`: the `<embedding_output_dir>` from step 3a (contains
  `<gene_id>/<gene_id>.pt`)
- `--gene_tsv`: candidate gene list for this trait/sex; only needs a
  `gene_id` column (see step 1)
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
  --out_dir results/<trait>_<sex>
```

This writes `<out_dir>/ftest_results_merged.tsv` (and `missing_genes.tsv`).
Run `python association_test.py run -h` for the full list of options
(cell-embedding covariates, PCA settings, etc.).

## 4. Replication check: HPP findings vs. UKBB external cohort

Run the same pipeline (step 3, both 3a and 3b) on an independent UKBB
cohort, with the same candidate gene lists and your own UKBB `--cohort_file`
and gene raw sequence files (see step 1), e.g. with
`--out_dir results_UKBB/<trait>_<sex>`.

For each trait/setting, take the significant genes from the HPP results and
from the UKBB results in the same way (filter the merged results to
`p < 2.5e-6` and deduplicate on `gene_id`).

**Comparison:** for each trait/setting, an HPP-significant gene is
**recovered** if its `gene_id` is also significant in the matching UKBB
trait/setting. A high recovery rate (`|HPP ∩ UKBB| / |HPP|`) for a
trait/setting is evidence that the HPP finding replicates in an independent
cohort rather than being an HPP-only artifact; genes significant in HPP but
not in UKBB are the ones that did not replicate.

## Acknowledgements

GB.DNA3 was developed by [GenBio AI](https://github.com/genbio-ai). We thank
GenBio AI for providing access to the model.
