#!/usr/bin/env bash
# Ning's 2026-10-30 request: replicate the FDR-level novel genes in UKB.
#
# Source: novel_gene_FDR_1026_no_height.tsv, committed as
# novel_genes_fdr_1026.tsv. Only Bonferroni=False rows are run (86 rows, 81
# genes); the True rows were replicated from the earlier exports. Height is not
# in this export yet. Tests (assoc_settings.py): F for Cholesterol and VAT,
# logistic LRT on all 60 embeddings for the disease traits.
#
# One driver for the whole run, calling the existing stage scripts in order:
#   targets  build_target_genes.py -> hpp_top_novel_genes.tsv
#   prep     run_pipeline.sh       -> target gene lists + sequences (CPU)
#   gpu      run_embeddings.sh     -> embeddings + feature tables (A100)
#   assoc    run_assoc.sh          -> association, recovery, replication (CPU)
#
#   bash ning_oct30.sh             # every stage in order, on one host
#   bash ning_oct30.sh prep        # CPU host
#   bash ning_oct30.sh gpu [run_embeddings.sh args, e.g. --world_size 4 --rank 0]
#   bash ning_oct30.sh assoc       # CPU host, after every gpu shard finished
#
# Results go to their own directory (ASSOC_DIR below): each cell's
# run_manifest.json pins its gene list, so the 1019 results directory would
# refuse this list, and the 1019 replication tables stay where they are.
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
export UKBV_ASSOC_DIR="${UKBV_OUT_ROOT}/assoc_results_fmgwas_89aba56_all60_fdr1026"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PHENO_PY="${UKBV_CONDA_ROOT}/envs/microbiome/bin/python3"
MARKER_DIR="${UKBV_OUT_ROOT}/.stage_markers"
EXPECTED_TARGETS=86

log() { printf '[%s] [ning_oct30] %s\n' "$(date +'%F %T')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

stage_targets() {
    "${PHENO_PY}" build_target_genes.py || die "build_target_genes.py failed"
    local n
    n=$(( $(wc -l < hpp_top_novel_genes.tsv) - 1 ))
    (( n == EXPECTED_TARGETS )) \
        || die "hpp_top_novel_genes.tsv has $n targets, expected $EXPECTED_TARGETS"
    log "targets: $n rows from novel_genes_fdr_1026.tsv (Bonferroni=False)"
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
        0) log "assoc done: results, recovery and replication tables in ${UKBV_ASSOC_DIR}" ;;
        3) die "assoc deferred: feature tables missing -- run the gpu stage first" ;;
        *) die "some trait/sex cells failed -- rerun 'bash ning_oct30.sh assoc' to retry" ;;
    esac
}

STAGE="${1:-all}"
(( $# > 0 )) && shift
log "stage: $STAGE, results: $UKBV_ASSOC_DIR"
case "$STAGE" in
    targets) stage_targets ;;
    prep)    stage_targets; stage_prep ;;
    gpu)     stage_gpu "$@" ;;
    assoc)   stage_assoc ;;
    all)     stage_targets; stage_prep; stage_gpu "$@"; stage_assoc ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown stage '$STAGE' (targets | prep | gpu | assoc | all)" ;;
esac
