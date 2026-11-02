#!/usr/bin/env bash
# Ning's 2026-11-02 request: rerun the disease-trait genes of
# novel_gene_FDR_1026_no_height.tsv (committed as novel_genes_fdr_1026.tsv)
# with the UKB case/control definitions aligned to HPP (UKBB_case_control.csv,
# ukb_traits_hpp.py). Every disease row this time, Bonferroni and FDR-only:
# 79 rows, 73 genes. Cholesterol and VAT are not rerun (labels unchanged).
#
# Osteoporosis is run both ways Ning proposed:
#   bmd        ICD-10 M80-M81 + self-report + min total femur BMD < 0.88 g/cm2,
#              controls with a measured normal BMD (closest to HPP; main run)
#   diagnosis  ICD-10 M80-M81 + self-report only, everyone else a control
#
# Embeddings do not depend on the labels, so there is no GPU stage: the
# features check confirms every target gene already has a feature table.
#
#   bash ning_nov02.sh bmd       # answers Ning's BMD questions (phenotypes only)
#   bash ning_nov02.sh targets   # target lists + filtered gene files
#   bash ning_nov02.sh features  # every target gene has features?
#   bash ning_nov02.sh cohorts   # HPP-aligned cohorts, both osteoporosis options
#   bash ning_nov02.sh cases     # case classes and UKB vs HPP case rates
#   bash ning_nov02.sh assoc     # association, then the replication tables
#   bash ning_nov02.sh           # all of the above, in order
#
# Run on a CPU host with the cluster paths (paths.env). Everything is written
# to new directories under UKBV_OUT_ROOT (*_hpp1102*); earlier cohorts and
# results are left alone. Every stage resumes.
set -uo pipefail
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
source "${HERE}/paths.sh"
PHENO_PY="${CONDA_ROOT}/envs/microbiome/bin/python3"

DISEASE_TRAITS=(Hyperlipidemia Hypertension Obesity Osteoporosis T2D)
SOURCE="novel_genes_fdr_1026.tsv"
EXPECTED_TARGETS=79
ASSOC_THREADS=8
ASSOC_SPARE_CORES=2
FEATURE_TABLE_ROOT="${OUT_ROOT}/feature_tables"
# Earlier replication tables under the paper labels, for the comparison:
# the FDR-only rows ran on 30 Oct, the Bonferroni rows in the 1019 run.
OLD_TABLES=(
    "${OUT_ROOT}/assoc_results_fmgwas_89aba56_all60_oct30/ukbb_replication_fdr_1026.tsv"
    "${OUT_ROOT}/assoc_results_fmgwas_89aba56_all60/ukbb_replication.tsv"
)

log() { printf '[%s] [ning_nov02] %s\n' "$(date +'%F %T')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# Settings of one run: $1 = bmd | diagnosis. The diagnosis run only has
# Osteoporosis targets, so it tests nothing else.
use_variant() {
    export UKBV_LABEL_DEFS=hpp_1102
    export UKBV_OSTEO_DEF="$1"
    local tag="hpp1102"
    [[ "$1" == "diagnosis" ]] && tag="hpp1102_osteo_dx"
    export UKBV_TARGET_GENES_TSV="${HERE}/targets_${tag}.tsv"
    export UKBV_TARGET_GENE_DIR="${OUT_ROOT}/target_gene_lists_${tag}"
    export UKBV_COHORT_DIR="${OUT_ROOT}/cohorts_${tag}"
    export UKBV_ASSOC_DIR="${OUT_ROOT}/assoc_results_fmgwas_89aba56_all60_${tag}"
}

stage_bmd() {
    use_variant bmd
    "${PHENO_PY}" bmd_report.py || die "bmd_report.py failed"
}

stage_targets() {
    use_variant bmd
    "${PHENO_PY}" build_target_genes.py --source "$SOURCE" --keep-validated \
        --traits "${DISEASE_TRAITS[@]}" --out "$UKBV_TARGET_GENES_TSV" \
        || die "build_target_genes.py failed"
    local n=$(( $(wc -l < "$UKBV_TARGET_GENES_TSV") - 1 ))
    (( n == EXPECTED_TARGETS )) || die "$n targets, expected $EXPECTED_TARGETS"
    "${PHENO_PY}" target_genes.py || die "target_genes.py failed (bmd)"

    use_variant diagnosis
    "${PHENO_PY}" build_target_genes.py --source "$SOURCE" --keep-validated \
        --traits Osteoporosis --out "$UKBV_TARGET_GENES_TSV" \
        || die "build_target_genes.py failed (osteoporosis)"
    "${PHENO_PY}" target_genes.py || die "target_genes.py failed (diagnosis)"
}

# The 30 Oct and 1019 runs embedded these genes. A gene without a feature
# table would only come back input_failed, so stop here instead.
stage_features() {
    use_variant bmd
    local file missing all_missing=""
    for file in "${UKBV_TARGET_GENE_DIR}"/*_extreme*.tsv; do
        [[ -f "$file" ]] || continue
        missing=$("${PHENO_PY}" cell_features.py "$file" "$FEATURE_TABLE_ROOT")
        [[ -n "$missing" ]] && all_missing+="${missing}"$'\n'
    done
    if [[ -n "$all_missing" ]]; then
        log "genes without a feature table:"
        printf '%s' "$all_missing" | sort -u >&2
        die "embed these first (prep + gpu stages, as in ning_oct30.sh)"
    fi
    log "every target gene has a feature table under $FEATURE_TABLE_ROOT"
}

stage_cohorts() {
    local v
    for v in bmd diagnosis; do
        use_variant "$v"
        "${PHENO_PY}" build_cohorts.py || die "build_cohorts.py failed ($v)"
    done
}

stage_cases() {
    local v
    for v in bmd diagnosis; do
        use_variant "$v"
        "${PHENO_PY}" case_breakdown.py || die "case_breakdown.py failed ($v)"
    done
}

# Target rows of one Bonferroni group: $1 = True | False, $2 = output name.
summarize_group() {
    local subset="${UKBV_ASSOC_DIR}/targets_$2.tsv"
    "${PHENO_PY}" - "$SOURCE" "$UKBV_TARGET_GENES_TSV" "$1" "$subset" <<'PY' \
        || die "could not write the $2 target subset"
import sys
import pandas as pd
source, targets_path, flag, out = sys.argv[1:]
bare = lambda s: s.astype(str).str.split(".").str[0]
src = pd.read_csv(source, sep="\t")
src = src[src["Bonferroni"].astype(str) == flag]
keys = set(zip(bare(src.gene_id), src.trait, src.sex))
targets = pd.read_csv(targets_path, sep="\t")
mask = [k in keys for k in zip(bare(targets.gene_id), targets.trait, targets.sex)]
targets[mask].to_csv(out, sep="\t", index=False)
PY
    "${PHENO_PY}" summarize_replication.py --targets "$subset" \
        --out "${UKBV_ASSOC_DIR}/ukbb_replication_$2" \
        || die "replication summary for $2 failed"
}

assoc_variant() {
    use_variant "$1"
    local cores jobs
    cores=$(nproc 2>/dev/null || echo 8)
    jobs=$(( (cores - ASSOC_SPARE_CORES) / ASSOC_THREADS ))
    (( jobs < 1 )) && jobs=1
    log "assoc ($1): $jobs rank(s) x $ASSOC_THREADS threads -> $UKBV_ASSOC_DIR"
    bash run_assoc.sh --jobs "$jobs" --threads "$ASSOC_THREADS"
    case $? in
        0) ;;
        3) die "assoc deferred: feature tables not readable here" ;;
        *) die "some cells failed ($1) -- rerun 'bash ning_nov02.sh assoc'" ;;
    esac
    summarize_group True bonferroni
    summarize_group False fdr_only
    "${PHENO_PY}" compare_label_runs.py \
        --new "${UKBV_ASSOC_DIR}/ukbb_replication.tsv" --old "${OLD_TABLES[@]}" \
        --export "$SOURCE" --out "${UKBV_ASSOC_DIR}/paper_vs_hpp_labels.tsv" \
        || die "label comparison failed ($1)"
}

stage_assoc() {
    assoc_variant bmd
    assoc_variant diagnosis
    log "done: ukbb_replication*.tsv and paper_vs_hpp_labels.tsv in"
    log "  ${OUT_ROOT}/assoc_results_fmgwas_89aba56_all60_hpp1102"
    log "  ${OUT_ROOT}/assoc_results_fmgwas_89aba56_all60_hpp1102_osteo_dx"
}

STAGE="${1:-all}"
log "stage: $STAGE"
case "$STAGE" in
    bmd)      stage_bmd ;;
    targets)  stage_targets ;;
    features) stage_targets; stage_features ;;
    cohorts)  stage_cohorts ;;
    cases)    stage_cases ;;
    assoc)    stage_assoc ;;
    all)      stage_bmd; stage_targets; stage_features; stage_cohorts
              stage_cases; stage_assoc ;;
    -h|--help) sed -n '2,27p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown stage '$STAGE' (bmd | targets | features | cohorts | cases | assoc | all)" ;;
esac
