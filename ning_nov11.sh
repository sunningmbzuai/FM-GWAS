#!/usr/bin/env bash
# Ning's 2026-11-11 request: 8 HPP novel Height genes (novel_genes_height_1111.tsv)
# through the same cross-trait scan as ning_nov03b.sh: every UKB trait, All / F /
# M, all 60 embeddings, F test or logistic LRT.
#
# SUDS3 and TRPM4 were already in the 170-gene 1103 scan; the other six were in
# no list we had. All 8 run here so Ning gets one self-contained package (the two
# repeats give the 1103 numbers again: same features, traits and code).
#
#   bash ning_nov11.sh prep       # CPU: target lists + gene sequences
#   bash ning_nov11.sh gpu [run_embeddings.sh args]
#                                 # GPU: embed genes without a .pt, export tables
#   bash ning_nov11.sh status     # CPU: which genes are testable, and why not
#   bash ning_nov11.sh scan       # CPU: features check, PCA, scan, summary + zip
#   bash ning_nov11.sh eta        # progress of a running pca/scan stage
#
# The trait catalog and values are the 1103 ones (gene_scan_1103/traits), so the
# traits match the 170-gene scan exactly. Output: UKBV_OUT_ROOT/gene_scan_1111,
# zip UKBV_OUT_ROOT/ukbb_gene_scan_1111.zip. Every stage resumes.
set -uo pipefail
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
source "${HERE}/paths.sh"
PHENO_PY="${CONDA_ROOT}/envs/microbiome/bin/python3"
PREP_PY="${CONDA_ROOT}/envs/wasp_env/bin/python3"

GENES="${HERE}/novel_genes_height_1111.tsv"
EXPECTED_GENES=8
RUN="${OUT_ROOT}/gene_scan_1111"
TARGET_GENE_DIR="${OUT_ROOT}/target_gene_lists_1111"
GENE_SEQ_DIR="${OUT_ROOT}/gene_sequences"

# Own target list and target dir: the default target_gene_lists/ is left as the
# earlier runs wrote it.
export UKBV_TARGET_GENES_TSV="$GENES"
export UKBV_TARGET_GENE_DIR="$TARGET_GENE_DIR"
export UKBV_GENE_ANNOTATION="${TARGET_GENE_DIR}/candidate_genes.tsv"
# ning_nov03b.sh reads these to scan this list instead of the 170 genes.
export SCAN_GENES="$GENES"
export SCAN_EXPECTED_GENES="$EXPECTED_GENES"
export SCAN_RUN="$RUN"
export SCAN_TRAITS="${OUT_ROOT}/gene_scan_1103/traits"
export SCAN_ZIP="ukbb_gene_scan_1111"
export SCAN_TITLE="UKB scan: 8 HPP novel Height genes (11 Nov) x every UKB trait, All / F / M"

log() { printf '[%s] [ning_nov11] %s\n' "$(date +'%F %T')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

check_genes() {
    [[ -f "$GENES" ]] || die "missing $GENES"
    local n=$(( $(wc -l < "$GENES") - 1 ))
    (( n == EXPECTED_GENES )) || die "$n genes in $GENES, expected $EXPECTED_GENES"
}

# One line per gene: name, id, and whether it has a sequence / feature table.
gene_table() {
    local id name seq feat
    while IFS=$'\t' read -r _ _ id name; do
        [[ "$id" == gene_id ]] && continue
        seq=no; feat=no
        compgen -G "${GENE_SEQ_DIR}/${id%%.*}*.tsv" >/dev/null && seq=yes
        compgen -G "${OUT_ROOT}/feature_tables/${id%%.*}*/*.npz" >/dev/null && feat=yes
        log "  $(printf '%-8s %-19s sequence=%-3s features=%s' "$name" "$id" "$seq" "$feat")"
    done < "$GENES"
}

stage_prep() {
    check_genes
    "${PHENO_PY}" target_genes.py || die "target_genes.py failed"
    [[ -f "$UKBV_GENE_ANNOTATION" ]] || die "target_genes.py wrote no $UKBV_GENE_ANNOTATION"
    # Genes already on disk are skipped, so only the new ones are built. A gene
    # with no sequence afterwards is untestable; the log above names the reason
    # (not protein-coding on chr1-22, over 128 kb, or no array variant).
    "${PREP_PY}" build_gene_sequences.py || die "build_gene_sequences.py failed"
    log "after prep:"
    gene_table
    log "next, on a GPU node: bash ning_nov11.sh gpu"
}

stage_gpu() {
    check_genes
    [[ -f "$UKBV_GENE_ANNOTATION" ]] || die "run 'bash ning_nov11.sh prep' first"
    command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
        || die "no GPU visible -- run 'bash ning_nov11.sh gpu' on a GPU node"
    # Embeds every gene under gene_sequences/ without a .pt (only the new ones
    # if earlier runs finished), then expands and exports the feature tables.
    bash run_embeddings.sh "$@" || die "run_embeddings.sh failed -- rerun to resume"
    log "after gpu:"
    gene_table
    log "next, on the CPU host: bash ning_nov11.sh status; bash ning_nov11.sh scan"
}

stage_status() {
    check_genes
    mkdir -p "$RUN"
    "${PHENO_PY}" untestable_genes.py --targets "$GENES" --out "${RUN}/gene_status.tsv" \
        || die "untestable_genes.py failed"
    log "per-gene status in ${RUN}/gene_status.tsv"
    gene_table
}

stage_scan() {
    check_genes
    [[ -f "${SCAN_TRAITS}/participants.tsv" ]] || die "missing ${SCAN_TRAITS} (the 1103 traits stage)"
    local stage
    # genes: features check (genes without tables are reported, not tested).
    for stage in genes pca scan summary; do
        bash ning_nov03b.sh "$stage" || die "ning_nov03b.sh $stage failed -- rerun resumes"
    done
    log "done: ${OUT_ROOT}/${SCAN_ZIP}.zip"
}

STAGE="${1:-}"
(( $# > 0 )) && shift
log "stage: ${STAGE:-<none>}"
case "$STAGE" in
    prep)   stage_prep ;;
    gpu)    stage_gpu "$@" ;;
    status) stage_status ;;
    scan)   stage_scan ;;
    eta)    bash ning_nov03b.sh eta ;;
    -h|--help|"") sed -n '2,19p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown stage '$STAGE' (prep | gpu | status | scan | eta)" ;;
esac
