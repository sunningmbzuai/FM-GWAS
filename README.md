# FM-GWAS External Replication in the UK Biobank

This branch contains the **external replication check** for the FM-GWAS gene
associations discovered in the Human Phenotype Project (HPP). The HPP
associations are re-tested in an independent cohort, the UK Biobank (UKB),
with the same method, gene definitions, traits and significance threshold. An
HPP association counts as replicated when the same gene reaches significance
for the same trait in UKB.

The branch holds only the replication pipeline. The FM-GWAS method itself
(the embedding and association scripts, the model code and the HPP gene
lists) lives on the `main` branch of this repository. The pipeline calls
those scripts from a separate checkout of `main`. No genotypes, phenotypes,
gene lists, embeddings or results are included here.

## Study design

| | HPP (discovery) | UKB (external replication) |
|---|---|---|
| Genotypes | low-coverage WGS, imputed | directly genotyped array (about 800K SNPs) |
| Genome build | GRCh37 | GRCh37 |
| Phasing | Eagle v2, population-based | Eagle v2, population-based |
| Gene sequences | TSS to TES, protein-coding, under 128 kb | same definition (GENCODE v19) |
| Model | AIDO.DNA3-AG | same model |
| Association | FM-GWAS F-test per embedding type | same test, same code |
| Significance | p < 2.5e-6, minimum across embedding types | same threshold, plus Bonferroni over the targets |

**Traits.** Cholesterol, Height and visceral adipose tissue (VAT) are tested
as quantitative traits. Hyperlipidemia, Hypertension, Type 2 diabetes, Obesity
and Osteoporosis are tested as binary traits. Each trait is tested in the full
cohort and in females and males separately, matching the HPP gene lists. That
gives 22 trait-by-sex cells, because HPP has no male gene list for
Hypertension or Obesity.

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

`build_cohorts.py` builds one cohort table per trait and sex, with
participant ID, sex, age and label, for QC-passed participants only.
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

**Feature export.** `export_features.py` turns each gene's embeddings into a
plain numerical feature table holding the same views the association script
builds. Every step after this one runs on CPU, with no PyTorch.

**Scale.** Genes can be split across GPUs in independent shards, and an
interrupted run resumes where it stopped. `check_embeddings.sh` reports
progress and checks embedding file integrity.

### 7. Association

`run_assoc.sh` runs the FM-GWAS association script for every trait-by-sex
cell.

- **Test:** per-gene PCA keeping 95% of the variance in each embedding type,
  with age and sex as covariates, followed by the analytic F-test. This
  matches the current FM-GWAS code on `main`.
- **Rank guard:** PCA never returns more components than the feature matrix
  actually spans (`rank_capped_pca.py`, `assoc_rank_capped.py`). Components
  past the rank would be arbitrary directions that still count toward the
  F-test's degrees of freedom, which makes p-values invalid at low-variant
  genes.
- **Parallelism:** each cell's untested genes are split into shards that run
  in parallel (`shard_genes.py`), with BLAS threads pinned for each worker.
- **Completeness:** a cell is marked complete only when every gene in it has
  features and has been tested (`cell_features.py`).

### 8. Replication

- **Recovery rate.** `compute_recovery.py` reports the fraction of
  HPP-significant genes that reach p < 2.5e-6 in UKB, using each gene's
  minimum p-value across embedding types.
- **Per-target replication.** `summarize_replication.py` writes three tables:
  - the UKB p-value for every embedding type
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
  test for a single gene, to confirm its p-value is valid.
  `check_gene_by_sex.py` shows one gene's results in every trait-by-sex cell.

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
| `fmgwas_assoc` | association | numpy, scikit-learn, statsmodels (built by `setup_assoc_env.sh`) |

   The environment names are set in `config.py`. Each driver checks its
   imports before it starts and never installs packages.

4. **Patches to the FM-GWAS scripts.** None of them changes the method. Each
   is idempotent and keeps a backup of the original file.

| Script | Change | Applied |
|---|---|---|
| `fix_assoc_feature_tables.py` | association script reads the exported feature tables, so it runs without PyTorch | automatically by `run_assoc.sh` |
| `fix_assoc_pca95.py` | brings older copies of the association script up to the 95%-variance PCA version; no change to current `main` | automatically by `run_assoc.sh` |
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
labelling, target selection, sharding, PCA rank capping, recovery and
replication summaries. They run with pytest and need no data or cluster
access.
