# FM-GWAS External Replication in the UK Biobank

This branch contains the **external replication check** for the FM-GWAS gene
associations discovered in the Human Phenotype Project (HPP). The HPP
associations are re-tested in an independent cohort, the UK Biobank (UKB),
with the same method, gene definitions, traits and significance threshold. An
HPP association counts as replicated when the same gene reaches significance
for the same trait in UKB.

The branch holds only the replication pipeline. The FM-GWAS method itself
(the embedding and association scripts, the model code and the HPP gene
lists) lives on the `main` branch of this repository. The pipeline calls the
embedding script, the model code and the HPP gene lists from a separate
checkout of `main`. The association script is the one exception: an exact
copy of `main`'s `insample_hpp_AG.py` is kept in `fmgwas/` (see
`fmgwas/UPSTREAM.md`), so results never depend on which commit the checkout
is at. No genotypes, phenotypes, gene lists, embeddings or results are
included here.

## Study design

| | HPP (discovery) | UKB (external replication) |
|---|---|---|
| Genotypes | low-coverage WGS, imputed | directly genotyped array (about 800K SNPs) |
| Genome build | GRCh37 | GRCh37 |
| Phasing | Eagle v2, population-based | Eagle v2, population-based |
| Gene sequences | TSS to TES, protein-coding, under 128 kb | same definition (GENCODE v19) |
| Model | AIDO.DNA3-AG | same model |
| Association | FM-GWAS F-test (quantitative) or logistic LRT (binary) per embedding type | same tests, same code |
| Significance | p < 2.5e-6, minimum across embedding types | same threshold, plus Bonferroni over the targets |

**Traits.** Cholesterol, Height and visceral adipose tissue (VAT) are tested
as quantitative traits. Hyperlipidemia, Hypertension, Type 2 diabetes, Obesity
and Osteoporosis are tested as binary traits. Each trait is tested in the full
cohort and in females and males separately: 24 trait-by-sex cells. A cell runs
when it has targets, including cells where HPP produced no gene list (for
example Hypertension in males).

**Targets.** By default the run tests the HPP novel associations: one target
per gene, trait and sex. It can also test the full HPP candidate gene union.

**Cohort difference.** UKB array genotyping carries far fewer variants per
gene than HPP sequencing. Some target genes contain no genotyped variant, and
others have only one or two distinct haplotypes across the cohort. The
pipeline reports these genes as untestable, with the reason, instead of
dropping them silently.

## Pipeline

### 1. Reference and build check

`setup_grch37_gtf.sh` downloads the GENCODE v19 annotation.
`validate_ukb_build.py` then confirms, before any heavy compute, that the
reference FASTA, the annotation and the UKB array calls agree on genome
build, contig naming and strand. It compares contig lengths against GRCh37
and checks that the reference base at sampled array SNPs matches one of the
two called alleles. The pipeline stops on any mismatch.

### 2. Phasing

`setup_eagle.sh` installs Eagle v2 and the hg19 genetic map.
`run_eagle_phasing_ukb.sh` restricts the array data to QC-passed participants
and chromosomes 1 to 22, then phases each chromosome with Eagle directly from
the PLINK files, with no reference panel. This is the same population-based
method used for HPP. The haplotypes in every gene sequence are therefore
statistically phased, not an arbitrary reference/alternate split.

### 3. Gene sequences

`build_gene_sequences.py` writes one table per gene, holding each
participant's reference sequence and two haplotype sequences.

- **Window:** transcription start to transcription end, with no flanking
  sequence, as in the FM-GWAS Methods.
- **Genes:** protein-coding genes on chromosomes 1 to 22. Genes longer than
  128 kb are excluded, not truncated.
- **Orientation:** sequences run 5' to 3' along the gene (`ukb_sequence.py`),
  because the embedding step reads the TSS window from the start of each
  sequence.
- **Memory:** sequences are streamed to disk one participant at a time
  (`gene_seq_writer.py`), so peak memory does not grow with cohort size.

### 4. Cohorts

`build_cohorts.py` builds one cohort table for every trait and sex (24
tables), with participant ID, sex, age and label, for QC-passed participants
only.
Phenotypes come from UKB release 676772, which honours participant
withdrawals. Trait definitions are kept in `ukb_fields.py` and
`ukb_traits.py`, so they can be reviewed on their own.

- **Quantitative traits** use the measured value. A top-versus-bottom-decile
  labelling is available as an option.
- **Binary traits** follow the FM-GWAS Methods. A participant is a case if
  any source supports it: a hospital ICD-10 code, self-reported illness,
  relevant medication, or a guideline measurement cutoff. As in HPP,
  osteopenia counts as osteoporosis and prediabetes counts as Type 2
  diabetes.

`ukb_source.py` reads the 23,550-column UKB release one column at a time, so
the full table is never loaded into memory.

### 5. Target genes

`build_target_genes.py` merges the HPP novel-association exports into one
target list. `target_genes.py` restricts sequence building, embedding and
testing to those genes. A target that HPP found in one sex only is still
tested in the pooled UKB cohort. Gene IDs are matched without version
suffixes, because Ensembl versions differ between annotation releases.

### 6. Embeddings, with sequence deduplication (GPU)

`run_embeddings.sh` runs on an A100 node and embeds every gene with
AIDO.DNA3-AG, through the FM-GWAS embedding script.

**Deduplication.** Array genotypes leave a gene with only a few dozen
distinct haplotype sequences across about 44,000 participants, so embedding
every participant row would repeat the same computation thousands of times.

- `dedup_gene_sequences.py` collapses each gene to its distinct sequences.
  It writes them two per row, so every forward pass embeds two new sequences.
  It also records an index that maps each participant's two haplotypes back
  to their distinct sequences.
- The model embeds only the distinct sequences.
- `expand_embeddings.py` uses the index to rebuild each participant's
  embedding.

The model never sees participant identity, and it embeds each haplotype
independently. The rebuilt embeddings are therefore exactly what embedding
every row would produce, not an approximation, at a small fraction of the
compute. While the GPU embeds one batch of genes, the next batch is
deduplicated on the CPU.

**Model package.** The model runs from a portable compiled package
(`aido_dna3ag_portable.py`). It needs no flash-attention build and no custom
operators, and it is not tied to one GPU architecture.

**Feature export.** `export_features.py` writes each gene's embedding fields
to a plain numerical table (`.npz`), converted to numbers exactly as the
association script converts them when it reads the embedding file itself.
The script reads the table in place of the embedding file and builds every
embedding type from it, so results are identical. Every step after this one
runs on CPU, with no PyTorch.

**Scale.** Genes can be split across GPUs in independent shards, and an
interrupted run resumes where it stopped. `check_embeddings.sh` reports
progress and checks embedding file integrity.

### 7. Association

`run_assoc.sh` runs the FM-GWAS association script (`fmgwas/insample_hpp_AG.py`,
unmodified) for every trait-by-sex cell. `assoc_settings.py` holds the
settings, chosen per trait:

- **Binary traits** (Hyperlipidemia, Hypertension, Type 2 diabetes, Obesity,
  Osteoporosis): logistic likelihood-ratio test (`--tests LRT`) on the 15
  `concat_tss_*` embedding types.
- **Quantitative traits** (Cholesterol, Height, VAT): partial F-test on all 60
  embedding types.
- **Both:** per-gene PCA keeping 95% of the variance (`--gene_pca
  variance:0.95`), with age and sex as covariates (age only in single-sex
  cells). The PCA is exact and stops at the rank of the feature matrix, and
  both tests drop directions the covariates already span.
- **Parallelism:** the script's own `--rank`/`--world_size` split, with more
  ranks for cells with more untested genes and BLAS threads pinned for each
  worker. The script's `merge` then writes one table per cell with
  Benjamini-Hochberg q-values.
- **Completeness:** a gene without features yet is recorded as failed by the
  script; `assoc_pending.py` clears that record once its features exist, so
  the gene is retried. A cell is marked complete only when every gene in it
  has features and has been tested (`cell_features.py`).

### 8. Replication

- **Recovery rate.** `compute_recovery.py` reports the fraction of
  HPP-significant genes that reach p < 2.5e-6 in UKB, using each gene's
  minimum p-value across embedding types (F-test p for quantitative traits,
  LRT p for binary traits).
- **Per-target replication.** `summarize_replication.py` writes three tables:
  - the UKB p-value for every embedding type, with the test it came from and
    the script's status for it (for example a logistic fit that separated)
  - one row per target, with the HPP p-value, the best UKB p-value and its
    embedding type, components kept, variant counts and replication flags
  - replication rates at the genome-wide threshold and at Bonferroni
    correction over the targets, over all targets and over the testable ones
- **Variant counts.** `count_gene_variants.py` reports how many genotyped
  variants each target carries in UKB and bins the genes by variant count,
  for comparison with HPP.
- **Testability.** `untestable_genes.py` lists the targets UKB cannot test,
  and why. Either no sequence could be built (no genotyped variant, not
  protein-coding, or longer than 128 kb), or every participant carries the
  same haplotype.
- **Calibration checks.** `assoc_diagnostics.py` and `diagnose_assoc.py`
  compare the analytic F-test with the matrix rank and with a permutation
  test for a single gene, to confirm its p-value is valid. They reproduce the
  earlier fixed-component recipe and are kept as a diagnostic.
  `check_gene_by_sex.py` shows one gene's results in every trait-by-sex cell.

## Relation to the FM-GWAS scripts on `main`

The statistics are `main`'s, unchanged. Everything this branch adds sits
before or after the two FM-GWAS scripts and changes no number they compute.

**Association (`insample_hpp_AG.py`, `main` 89aba56).** Used as an exact copy
in `fmgwas/` (`fmgwas/UPSTREAM.md` gives the commit and hash; a test fails if
the copy differs). `run_assoc.sh` calls it the way `main`'s README describes
in step 4b, with these differences:

| `main` README, step 4b | This branch |
|---|---|
| `--gene_file`, `--save_path` | `--gene_tsv`, `--out_dir`: the script's actual option names (the README's names are not accepted by the 89aba56 script) |
| `--covariates`: `age,sex` for All, `age` for F and M | `age,sex` for every cell; the script drops `sex` itself in single-sex cells, so the model is the same |
| `--gene_pca variance:0.95` | same |
| `--tests LRT` | `LRT` for binary traits, on the 15 `concat_tss_*` embedding types; `F` for quantitative traits (the LRT needs a 0/1 label), on all 60 (`assoc_settings.py`) |
| `--feature_root <embedding_output_dir>` (`<gene>/<gene>.pt`) | the exported tables (`<gene>/<gene>.npz`), which the script reads in place of the `.pt` with identical results (see below) |
| one `--rank` per worker, chosen by hand | ranks per cell in proportion to its untested genes, run in parallel, then the script's `merge` |
| `--gene_file HPP_union_gene/<trait>_extreme[_F\|_M].tsv` | the cell's target genes (`target_gene_lists/`), including targets no HPP gene list contains |

**Embedding (`get_hpp_embedding-AG_clean.py`).** Run from the `main` checkout
with three small patches that do not change the method (see Setup). It embeds
each gene's distinct haplotype sequences once, instead of every participant's
row; `expand_embeddings.py` then rebuilds the per-participant file.

**Why results are identical.** `tests/test_fmgwas_agreement.py` builds one
gene's embeddings twice, once per participant the way `main` does and once
through deduplication, expansion and `.npz` export, runs the association
script on both exactly as `run_assoc.sh` does, and requires equal result
files for the F-test and the LRT. The two rules this rests on: the model sees
each haplotype independently, so embedding a distinct sequence once gives the
same numbers as embedding it in every row; and `export_features.py` converts
the embedding fields to numbers with the same rule the script applies to a
`.pt` file (bfloat16 to float32, other types unchanged).

**Outputs.** As the script writes them: `<trait>_<sex>/per_gene/<gene>.tsv`
and `<trait>_<sex>/ftest_results_merged.tsv` (with BH q-values) under
`assoc_results_fmgwas_89aba56/`. Replication uses `p_LRT` for binary traits
and `p_F` for quantitative traits, and reports the script's status for each
test (for example a logistic fit that separated) without filtering on it.

## Running the replication

### Setup

1. **Paths.** Copy `paths.env.example` to `paths.env` and fill it in, or
   export the same variables. It sets the inputs (UKB genotypes, QC sample
   list, phenotypes, GRCh37 reference), the FM-GWAS checkout, the compiled
   model, the conda root and the output directory. The code contains no
   paths. `config.py` and `paths.sh` read these variables, and each stage
   stops with a clear message when a variable it needs is missing.
2. **FM-GWAS checkout.** Clone this repository's `main` branch separately
   and point `UKBV_FMGWAS_ROOT` at that clone.
3. **Environments.**

| Environment | Used for | Needs |
|---|---|---|
| `wasp_env` | build check, phasing, gene sequences | pysam, pandas, bcftools, tabix |
| `microbiome` | cohorts, targets, recovery | pandas, pyarrow |
| `deep_learning` | embeddings | PyTorch 2.5 with CUDA, scikit-learn, transformers |
| `fmgwas_assoc` | association | numpy, scipy, pandas, scikit-learn (built by `setup_assoc_env.sh`) |

   The environment names are set in `config.py`. Each driver checks its
   imports before it starts and never installs packages.

4. **Patches to the FM-GWAS embedding script.** None of them changes the
   method. Each is idempotent and keeps a backup of the original file. The
   association script is not patched.

| Script | Change | Applied |
|---|---|---|
| `fix_seq_column_detection.py` | embedding script finds the haplotype sequence columns | once, before embedding |
| `fix_batch_size_override.py` | embedding batch size can be pinned, for 40 GB A100 cards | once, before embedding |
| `fix_hf_dynamic_imports.py` | Hugging Face no longer reports the model's own modules as missing packages | once, before embedding |

### Execution

Statistical stages run on a CPU host. Only embedding runs on a GPU host.
Both hosts must see the same output directory.

1. **CPU:** `run_pipeline.sh` runs the reference setup, build check,
   phasing, cohorts, target lists and gene sequences.
2. **GPU:** `run_embeddings.sh` deduplicates, embeds, expands and exports the
   feature tables.
3. **CPU:** `run_pipeline.sh` again runs the association tests (through
   `run_assoc.sh`) and the recovery rate.
4. **CPU:** `count_gene_variants.py`, `summarize_replication.py` and
   `untestable_genes.py` produce the replication tables.

Every stage is resumable. A finished stage leaves a marker and is skipped on
later runs. A stage whose inputs are not ready yet is deferred and retried on
the next run instead of failing.

## Tests

`tests/` holds unit tests for the pure-function parts of the pipeline:
sequence writing and orientation, deduplication and expansion, trait
labelling, target selection, pending-gene tracking, recovery and replication
summaries. `test_fmgwas_agreement.py` checks the vendored association script
against `main` by hash, then runs it end to end on synthetic data, once on
embeddings computed per participant and once through deduplication,
expansion and feature export, and requires identical results for both the
F-test and the LRT. The tests run with pytest and need no data or cluster
access.
