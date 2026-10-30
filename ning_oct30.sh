#!/usr/bin/env bash
# Ning's open requests, run as one batch (2026-10-30):
#   26 Oct  replicate the FDR-level novel genes in UKB
#   23 Oct  positive control: LRT on the known genes, all 60 embeddings
#   23 Oct  how each UKB case group is made up, against the HPP case rates
#
# Source: novel_gene_FDR_1026_no_height.tsv, committed as
# novel_genes_fdr_1026.tsv. Only Bonferroni=False rows are run (86 rows, 81
# genes); the True rows were replicated from the earlier exports. Height is not
# in this export yet. The six known genes (known_genes_1023.tsv) join the same
# target list, 92 rows in all. Tests (assoc_settings.py): F for Cholesterol and
# VAT, logistic LRT on all 60 embeddings for the disease traits.
#
# One driver for the whole run, calling the existing stage scripts in order:
#   targets  build_target_genes.py -> hpp_top_novel_genes.tsv
#   cases    case_breakdown.py     -> case classes and UKB vs HPP case rates
#   prep     run_pipeline.sh       -> target gene lists + sequences (CPU)
#   gpu      run_embeddings.sh     -> embeddings + feature tables (A100)
#   assoc    run_assoc.sh          -> association, recovery, replication (CPU),
#            then the replication tables again for each list on its own
#
#   bash ning_oct30.sh             # every stage in order, on one host
#   bash ning_oct30.sh cases       # CPU host, any time (reads phenotypes only)
#   bash ning_oct30.sh prep        # CPU host
#   bash ning_oct30.sh gpu [run_embeddings.sh args, e.g. --world_size 4 --rank 0]
#   bash ning_oct30.sh assoc       # CPU host, after every gpu shard finished
#
# Results go to their own directory (ASSOC_DIR below): each cell's
# run_manifest.json pins its gene list, so the 1019 results directory would
# refuse this list, and the 1019 replication tables stay where they are.
# (assoc_results_fmgwas_89aba56_all60_fdr1026 is a stopped first attempt
# without the known genes; it can be deleted.)
# Every stage resumes, so rerunning any of them only does what is left.
set -uo pipefail
export PYTHONUNBUFFERED=1

export UKBV_OUT_ROOT=/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ukbb_validation/ukb_run
export UKBV_CONDA_ROOT=/home/zacharyl/miniconda3
export UKBV_FMGWAS_ROOT=/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ukbb_validation/FM-GWAS
export UKBV_UKB_BFILE=/net/mraid20/ifs/wisdom/segal_lab/jasmine/gil/ukb_snp_all
export UKBV_QC_FAM=/net/mraid20/ifs/wisdom/segal_lab/jasmine/gil/qc_output_tabular/ukb_qc_final.fam
export UKBV_PHENO_DIR=/net/mraid20/ifs/wisdom/segal_lab/jafar/UKBioBank/Data
export UKBV_GRCH37_FASTA=/net/mraid20/ifs/wisdom/segal_lab/genie/10K/genetics/NewImputation/merged_panel/1000GP_ref_panel/human_g1k_v37.fasta
export UKBV_COMPILED_BINARY=/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ukbb_validation/aido_dna3ag_1bp4_dyn_portable.pt2
export UKBV_GRCH37_GTF=/net/mraid20/ifs/wisdom/segal_lab/jasmine/gil/annotation/gencode.v19.annotation.gtf
export UKBV_EAGLE_DIR=/net/mraid20/ifs/wisdom/segal_lab/jasmine/zach/ukbb_validation/eagle_local
export UKBV_PLINK2=/net/mraid20/export/genie/Bin/plink2a/plink2
export UKBV_ASSOC_DIR="${UKBV_OUT_ROOT}/assoc_results_fmgwas_89aba56_all60_oct30"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PHENO_PY="${UKBV_CONDA_ROOT}/envs/microbiome/bin/python3"
MARKER_DIR="${UKBV_OUT_ROOT}/.stage_markers"
EXPECTED_TARGETS=92   # 86 FDR (Bonferroni=False) + 6 known genes

log() { printf '[%s] [ning_oct30] %s\n' "$(date +'%F %T')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

stage_targets() {
    "${PHENO_PY}" build_target_genes.py || die "build_target_genes.py failed"
    local n
    n=$(( $(wc -l < hpp_top_novel_genes.tsv) - 1 ))
    (( n == EXPECTED_TARGETS )) \
        || die "hpp_top_novel_genes.tsv has $n targets, expected $EXPECTED_TARGETS"
    log "targets: $n rows (FDR genes + known-gene positive controls)"
}

stage_cases() {
    "${PHENO_PY}" case_breakdown.py || die "case_breakdown.py failed"
}

# The replication tables for one source list: its rows of the target list,
# summarised on their own so the FDR genes and the controls each get their
# own rates and targeted (0.05/n) threshold.
summarize_one() {   # $1 = source export, $2 = output name
    local subset="${UKBV_ASSOC_DIR}/targets_$2.tsv"
    "${PHENO_PY}" - "$1" "$subset" <<'PY' || die "could not write the $2 target subset"
import sys
import pandas as pd
import build_target_genes as B
import target_genes as T
keys = {(T.strip_version(r[B.GENE_ID_COLUMN]), r[B.TRAIT_COLUMN], r[B.SEX_COLUMN])
        for r in B.union_sources([sys.argv[1]])}
targets = pd.read_csv(B.OUTPUT_TSV, sep="\t")
row_keys = zip(targets.gene_id.map(T.strip_version), targets.trait, targets.sex)
targets[[k in keys for k in row_keys]].to_csv(sys.argv[2], sep="\t", index=False)
PY
    "${PHENO_PY}" summarize_replication.py --targets "$subset" \
        --out "${UKBV_ASSOC_DIR}/ukbb_replication_$2" \
        || die "replication summary for $2 failed"
}

stage_prep() {
    # The earlier run left gene_sequences.done, which would skip the new
    # genes. build_gene_sequences.py skips genes already on disk, so dropping
    # the marker only builds the missing ones.
    rm -f "${MARKER_DIR}/gene_sequences.done"
    bash run_pipeline.sh || die "run_pipeline.sh failed"
}

stage_gpu() {
    command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
        || die "no GPU visible -- run 'bash ning_oct30.sh gpu' on the A100 node"
    bash run_embeddings.sh "$@" || die "run_embeddings.sh failed"
}

stage_assoc() {
    # Exit 3 means the feature tables cannot be read here (not exported yet).
    bash run_assoc.sh
    case $? in
        0) summarize_one novel_genes_fdr_1026.tsv fdr_1026
           summarize_one known_genes_1023.tsv known_1023
           log "assoc done: ukbb_replication_fdr_1026.* and ukbb_replication_known_1023.* in ${UKBV_ASSOC_DIR}" ;;
        3) die "assoc deferred: feature tables missing -- run the gpu stage first" ;;
        *) die "some trait/sex cells failed -- rerun 'bash ning_oct30.sh assoc' to retry" ;;
    esac
}

STAGE="${1:-all}"
(( $# > 0 )) && shift
log "stage: $STAGE, results: $UKBV_ASSOC_DIR"
case "$STAGE" in
    targets) stage_targets ;;
    cases)   stage_cases ;;
    prep)    stage_targets; stage_prep ;;
    gpu)     stage_gpu "$@" ;;
    assoc)   stage_assoc ;;
    all)     stage_targets; stage_cases; stage_prep; stage_gpu "$@"; stage_assoc ;;
    -h|--help) sed -n '2,33p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown stage '$STAGE' (targets | cases | prep | gpu | assoc | all)" ;;
esac
