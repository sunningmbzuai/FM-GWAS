"""Settings and paths for the UKB -> FM-GWAS replication pipeline.

Analysis settings (gene window, trait list, label mode) are plain constants.

Paths come from the environment, never from this file. Each UKBV_* variable
can be exported, or set in a paths.env file next to this module (KEY=VALUE
lines; see paths.env.example). The environment wins over the file. paths.sh
reads the same variables for the shell drivers, so both halves agree.

Path attributes resolve lazily: importing config never fails, and a missing
variable raises a clear error only when a stage asks for that path. Unit tests
therefore run without any of it set.
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
PATHS_ENV_FILE = os.path.join(HERE, "paths.env")

# How the three quantitative traits (Cholesterol, Height, VAT) are labelled.
# The paper analyses "8 curated continuous and binary phenotypes" and uses the
# continuous traits as measured values, so "continuous" is correct for the
# replication; "extreme" (top decile 1 / bottom decile 0, middle dropped) is
# kept only as a deliberate deviation. The "_extreme" in the HPP gene-list
# FILENAMES refers to the gene selection, not to the phenotype labels.
# Disease traits are always binary regardless of this setting.
QUANT_LABEL_MODE = "continuous"

# Gene window. The paper (Methods, "Human Phenotype Project") defines gene
# sequences as TSS -> TES with NO flanks, so both flanks are 0. They are kept
# as config rather than deleted so a flank experiment is a one-line change.
GENE_UPSTREAM_BP = 0
GENE_DOWNSTREAM_BP = 0
# "protein-coding genes with sequence lengths shorter than 128 kb". Genes over
# this are EXCLUDED, not truncated -- a truncated gene is a different sequence,
# not a shorter one, and the paper's 16,384-gene set comes from exclusion.
MAX_GENE_LEN = 128 * 1024
GENE_TYPE = "protein_coding"   # GTF gene_type filter, per Methods

# ---- key-association mode -------------------------------------------------
# Default is the whole HPP union (thousands of genes, a full embedding pass per
# gene). With RESTRICT_TO_TARGET_GENES the run narrows to the headline novel
# hits listed in TARGET_GENES_TSV: only those genes get sequences, embeddings
# and association tests, and recovery is scored against them alone.
# build_target_genes.py regenerates the TSV from the HPP exports; edit those
# and rerun, never the TSV by hand.
RESTRICT_TO_TARGET_GENES = True
TARGET_GENES_TSV = os.path.join(HERE, "hpp_top_novel_genes.tsv")

# Conda env names, resolved under UKBV_CONDA_ROOT by the shell drivers.
PREP_ENV = "wasp_env"      # pysam + pandas: build check, gene sequences
PHENO_ENV = "microbiome"   # pyarrow + pandas: cohorts, targets, recovery
GPU_ENV = "deep_learning"  # precompiled for the model -- never install into it

# ---- traits (must match HPP_union_gene/{trait}_extreme[_F|_M].tsv stems) --
TRAITS = [
    "Cholesterol", "Height", "Hyperlipidemia", "Hypertension",
    "Obesity", "Osteoporosis", "T2D", "VAT",
]
SEX_SUFFIXES = {"All": "", "F": "_F", "M": "_M"}  # missing file => skip that sex


# ---- paths ------------------------------------------------------------------
# Required variables, with what each one must point at.
REQUIRED_VARS = {
    "UKBV_OUT_ROOT": "directory every pipeline output is written under",
    "UKBV_UKB_BFILE": "PLINK prefix of the UKB genotyped array calls (GRCh37)",
    "UKBV_QC_FAM": ".fam listing the QC-passed UKB samples",
    "UKBV_PHENO_DIR": "directory holding ukb676772.parquet and phenotypes_export/",
    "UKBV_GRCH37_FASTA": "GRCh37 reference FASTA (human_g1k_v37)",
    "UKBV_CONDA_ROOT": "conda root holding the pipeline's environments",
    "UKBV_FMGWAS_ROOT": "checkout of the FM-GWAS repository (main branch)",
}


def _read_env_file(path: str) -> dict[str, str]:
    """KEY=VALUE lines; blank lines, comments and an `export ` prefix allowed."""
    values: dict[str, str] = {}
    if not os.path.isfile(path):
        return values
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            value = value.split(" #", 1)[0].strip().strip("'\"")
            values[key.strip()] = os.path.expandvars(value)
    return values


def _setting(name: str, default: str | None = None) -> str:
    value = os.environ.get(name) or _read_env_file(PATHS_ENV_FILE).get(name) or default
    if not value:
        raise RuntimeError(
            f"{name} is not set ({REQUIRED_VARS.get(name, 'see paths.env.example')}). "
            f"Export it, or add it to {PATHS_ENV_FILE}.")
    return value


def _out(sub: str = "") -> str:
    root = _setting("UKBV_OUT_ROOT")
    return f"{root}/{sub}" if sub else root


def _pheno(sub: str) -> str:
    return f"{_setting('UKBV_PHENO_DIR')}/{sub}"


def _fmgwas(sub: str = "") -> str:
    root = _setting("UKBV_FMGWAS_ROOT")
    return f"{root}/{sub}" if sub else root


def _eagle(sub: str = "") -> str:
    root = _setting("UKBV_EAGLE_DIR", _out("eagle_local"))
    return f"{root}/{sub}" if sub else root


# Each path is resolved on its own, so a stage needs only the variables its
# own paths depend on (recovery needs UKBV_OUT_ROOT, not the genotypes).
_PATHS = {
    # ---- inputs -------------------------------------------------------------
    "UKB_BFILE": lambda: _setting("UKBV_UKB_BFILE"),  # .bed/.bim/.fam, GRCh37 array
    "QC_FAM": lambda: _setting("UKBV_QC_FAM"),        # QC-passed sample list
    # Release 676772 (Jan 2024) honours participant withdrawals. 502,244 rows x
    # 23,550 columns -- read column-selectively, never whole (ukb_source.py).
    # The phenotypes export is the same release, one tidied table per assay;
    # it supplies the DXA and blood-biomarker columns.
    "UKB_MAIN_PARQUET": lambda: _pheno("ukb676772.parquet"),
    "UKB_BLOOD_PARQUET": lambda: _pheno("phenotypes_export/ukb676772_blood_tests.parquet"),
    "UKB_DXA_PARQUET": lambda: _pheno("phenotypes_export/ukb676772_dxa.parquet"),
    "GRCH37_FASTA": lambda: _setting("UKBV_GRCH37_FASTA"),
    # gencode.v19 is the last GENCODE release native to GRCh37;
    # setup_grch37_gtf.sh downloads it here if missing.
    "GRCH37_GTF": lambda: _setting(
        "UKBV_GRCH37_GTF", _out("reference/gencode.v19.annotation.gtf")),
    "FMGWAS_ROOT": lambda: _fmgwas(),
    "HPP_UNION_GENE_DIR": lambda: _fmgwas("HPP_union_gene"),
    "MODEL_PATH": lambda: _fmgwas("TransUNetWithTracks"),
    "CONDA_ROOT": lambda: _setting("UKBV_CONDA_ROOT"),
    "PLINK2": lambda: _setting("UKBV_PLINK2", "plink2"),
    # Eagle must live on real local (non-FUSE) disk: binaries on FUSE mounts
    # often fail to execute. setup_eagle.sh fills it.
    "EAGLE_LOCAL_DIR": lambda: _eagle(),
    "EAGLE_BIN": lambda: _eagle("eagle"),
    "EAGLE_GENETIC_MAP": lambda: _eagle("tables/genetic_map_hg19_withX.txt.gz"),
    # ---- this pipeline's own outputs ----------------------------------------
    "OUT_ROOT": lambda: _out(),
    "COHORT_DIR": lambda: _out("cohorts"),           # <trait>_<All|F|M>.tsv
    "GENE_SEQ_DIR": lambda: _out("gene_sequences"),  # <gene_id>.tsv
    "FEATURE_ROOT": lambda: _out("embeddings"),      # embedding script output
    # Deduplicated embedding path: dedup_gene_sequences.py writes the distinct
    # sequences here, the embedding script runs on those, and
    # expand_embeddings.py gathers the result back to one row per participant
    # before assoc sees it.
    "UNIQUE_GENE_SEQ_DIR": lambda: _out("gene_sequences_unique"),
    "UNIQUE_FEATURE_ROOT": lambda: _out("embeddings_unique"),
    # Torch-free feature tables written by export_features.py on the GPU node;
    # assoc reads these instead of the .pt files.
    "FEATURE_TABLE_ROOT": lambda: _out("feature_tables"),
    # insample_hpp_AG.py output per trait/sex. Versioned with FM-GWAS
    # d17eae23a6 (95%-variance PCA, raw covariates).
    "ASSOC_DIR": lambda: _out("assoc_results_pca95"),
    # Filtered copies of HPP_union_gene/<trait>_extreme*.tsv, written by
    # target_genes.py; FM-GWAS reads these instead of the full lists.
    "TARGET_GENE_DIR": lambda: _out("target_gene_lists"),
    # Eagle phasing of the QC-passed autosomes (run_eagle_phasing_ukb.sh).
    "PHASING_DIR": lambda: _out("phasing"),
    "PHASED_VCF": lambda: _out("phasing/all.norm.phased.vcf.gz"),
    # Always empty for UKB: PLINK bim rows are biallelic, so there are no
    # multi-allelic blind spots to mask. Kept so build_gene_sequences.py's
    # load_blindspots() has a path that exists.
    "PHASING_BLINDSPOTS_BED": lambda: _out("phasing/multiallelic_blindspots.bed"),
    # QC-passed, autosome-only binary subset of UKB_BFILE that Eagle reads.
    "UKB_QC_SUBSET": lambda: _out("ukb_qc_subset"),
}


def __getattr__(name: str) -> str:
    """Resolve path attributes on first use (PEP 562)."""
    if name in _PATHS:
        return _PATHS[name]()
    raise AttributeError(f"module 'config' has no attribute {name!r}")
