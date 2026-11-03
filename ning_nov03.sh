#!/usr/bin/env bash
# Ning's 2026-11-03 correction to the 2 Nov rerun (ning_nov02.sh): controls
# are the "control" class only, i.e. the paper reading's controls
# (ukb_traits.py). Every other non-case -- medication only, a lab or
# blood-pressure cutoff, a T-score -- is dropped, not counted as a control.
# Label set hpp_1103 (ukb_traits_hpp.py), two case schemes:
#
#   trait           scheme 1 (icd_selfrep)       scheme 2 (icd_only)
#   Hypertension    I10-I15 + self-report        I10-I15
#   Hyperlipidemia  E78 + self-report            E78
#   Osteoporosis    M80-M81 + self-report        M80-M81
#   T2D             E11                          E11
#
# T2D is the same in both schemes. Obesity is not rerun (labels confirmed).
# The targets are the disease rows of novel_genes_fdr_1026.tsv for these four
# traits (67 rows), a subset of the 2 Nov targets, whose genes all passed the
# features check there, so there is no GPU stage.
#
#   bash ning_nov03.sh targets   # target list + filtered gene files
#   bash ning_nov03.sh cohorts   # hpp_1103 cohorts, both schemes
#   bash ning_nov03.sh cases     # case classes, excluded classes, case rates
#   bash ning_nov03.sh assoc     # association, then the replication tables
#   bash ning_nov03.sh score     # replication tables only (assoc already done)
#   bash ning_nov03.sh           # all of the above, in order
#
# Run on a CPU host with the cluster paths (paths.env). Everything is written
# to new directories under UKBV_OUT_ROOT (*_hpp1103_*); earlier cohorts and
# results are left alone. Every stage resumes.
set -uo pipefail
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
source "${HERE}/paths.sh"
PHENO_PY="${CONDA_ROOT}/envs/microbiome/bin/python3"

TRAITS=(Hyperlipidemia Hypertension Osteoporosis T2D)
SCHEMES=(icd_selfrep icd_only)
SOURCE="novel_genes_fdr_1026.tsv"
EXPECTED_TARGETS=67
NOV02_TARGETS="${HERE}/targets_hpp1102.tsv"
TARGETS_TSV="${HERE}/targets_hpp1103.tsv"
TARGET_GENE_DIR="${OUT_ROOT}/target_gene_lists_hpp1103"
# Targeted threshold 0.05/N, with N as in the 2 Nov run (not the 67 here):
# 24 for Osteoporosis (the diagnosis-only run), 79 for the other traits.
N_TESTS=(Osteoporosis=24 default=79)
ASSOC_THREADS=8
ASSOC_SPARE_CORES=2
# Earlier replication tables, for the comparisons. Paper labels: the FDR-only
# rows ran on 30 Oct, the Bonferroni rows in the 1019 run. hpp_1102: the
# diagnosis-only osteoporosis run first, since its cases match scheme 1.
PAPER_TABLES=(
    "${OUT_ROOT}/assoc_results_fmgwas_89aba56_all60_oct30/ukbb_replication_fdr_1026.tsv"
    "${OUT_ROOT}/assoc_results_fmgwas_89aba56_all60/ukbb_replication.tsv"
)
HPP1102_TABLES=(
    "${OUT_ROOT}/assoc_results_fmgwas_89aba56_all60_hpp1102_osteo_dx/ukbb_replication.tsv"
    "${OUT_ROOT}/assoc_results_fmgwas_89aba56_all60_hpp1102/ukbb_replication.tsv"
)

log() { printf '[%s] [ning_nov03] %s\n' "$(date +'%F %T')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# Settings of one run: $1 = icd_selfrep | icd_only. Both share the targets.
use_scheme() {
    export UKBV_LABEL_DEFS=hpp_1103
    export UKBV_CASE_SCHEME="$1"
    export UKBV_TARGET_GENES_TSV="$TARGETS_TSV"
    export UKBV_TARGET_GENE_DIR="$TARGET_GENE_DIR"
    export UKBV_COHORT_DIR="${OUT_ROOT}/cohorts_hpp1103_$1"
    export UKBV_ASSOC_DIR="${OUT_ROOT}/assoc_results_fmgwas_89aba56_all60_hpp1103_$1"
}

stage_targets() {
    use_scheme icd_selfrep
    "${PHENO_PY}" build_target_genes.py --source "$SOURCE" --keep-validated \
        --traits "${TRAITS[@]}" --out "$TARGETS_TSV" \
        || die "build_target_genes.py failed"
    local n=$(( $(wc -l < "$TARGETS_TSV") - 1 ))
    (( n == EXPECTED_TARGETS )) || die "$n targets, expected $EXPECTED_TARGETS"
    # Every target must be one the 2 Nov features check already covered.
    [[ -f "$NOV02_TARGETS" ]] || die "missing $NOV02_TARGETS (run ning_nov02.sh targets)"
    "${PHENO_PY}" - "$TARGETS_TSV" "$NOV02_TARGETS" <<'PY' || die "targets outside the 2 Nov list need the features check"
import sys
import pandas as pd
key = lambda t: set(zip(t.gene_id.astype(str).str.split(".").str[0], t.trait, t.sex))
new, old = (pd.read_csv(p, sep="\t") for p in sys.argv[1:])
extra = key(new) - key(old)
for row in sorted(extra):
    print(f"  not in the 2 Nov targets: {row}", file=sys.stderr)
sys.exit(1 if extra else 0)
PY
    "${PHENO_PY}" target_genes.py || die "target_genes.py failed"
}

stage_cohorts() {
    local s
    for s in "${SCHEMES[@]}"; do
        use_scheme "$s"
        "${PHENO_PY}" build_cohorts.py || die "build_cohorts.py failed ($s)"
    done
}

stage_cases() {
    local s
    for s in "${SCHEMES[@]}"; do
        use_scheme "$s"
        "${PHENO_PY}" case_breakdown.py || die "case_breakdown.py failed ($s)"
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
        --n-tests "${N_TESTS[@]}" --out "${UKBV_ASSOC_DIR}/ukbb_replication_$2" \
        || die "replication summary for $2 failed"
}

# $1 = scheme, $2 = name of the earlier run, $3 = output, rest = earlier tables.
compare_with() {
    local scheme="$1" old_name="$2" out="$3"
    shift 3
    "${PHENO_PY}" compare_label_runs.py \
        --new "${UKBV_ASSOC_DIR}/ukbb_replication.tsv" --old "$@" \
        --export "$SOURCE" --new-name "$scheme" --old-name "$old_name" \
        --out "${UKBV_ASSOC_DIR}/${out}" \
        || die "comparison with $old_name failed ($scheme)"
}

assoc_scheme() {
    use_scheme "$1"
    local cores jobs
    cores=$(nproc 2>/dev/null || echo 8)
    jobs=$(( (cores - ASSOC_SPARE_CORES) / ASSOC_THREADS ))
    (( jobs < 1 )) && jobs=1
    log "assoc ($1): $jobs rank(s) x $ASSOC_THREADS threads -> $UKBV_ASSOC_DIR"
    bash run_assoc.sh --jobs "$jobs" --threads "$ASSOC_THREADS"
    case $? in
        0) ;;
        3) die "assoc deferred: feature tables not readable here" ;;
        *) die "some cells failed ($1) -- rerun 'bash ning_nov03.sh assoc'" ;;
    esac
    score_scheme "$1"
}

# Replication tables and comparisons of one finished scheme: $1 = scheme.
score_scheme() {
    use_scheme "$1"
    "${PHENO_PY}" summarize_replication.py --targets "$UKBV_TARGET_GENES_TSV" \
        --n-tests "${N_TESTS[@]}" --out "${UKBV_ASSOC_DIR}/ukbb_replication" \
        || die "replication summary failed ($1)"
    summarize_group True bonferroni
    summarize_group False fdr_only
    compare_with "$1" paper paper_vs_hpp1103.tsv "${PAPER_TABLES[@]}"
    compare_with "$1" hpp1102 hpp1102_vs_hpp1103.tsv "${HPP1102_TABLES[@]}"
}

stage_score() {
    local s
    for s in "${SCHEMES[@]}"; do
        score_scheme "$s"
    done
}

stage_assoc() {
    local s
    for s in "${SCHEMES[@]}"; do
        assoc_scheme "$s"
    done
    log "done: ukbb_replication*.tsv and *_vs_hpp1103.tsv in"
    for s in "${SCHEMES[@]}"; do
        log "  ${OUT_ROOT}/assoc_results_fmgwas_89aba56_all60_hpp1103_$s"
    done
}

STAGE="${1:-all}"
log "stage: $STAGE"
case "$STAGE" in
    targets)  stage_targets ;;
    cohorts)  stage_cohorts ;;
    cases)    stage_cases ;;
    assoc)    stage_assoc ;;
    score)    stage_score ;;
    all)      stage_targets; stage_cohorts; stage_cases; stage_assoc ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown stage '$STAGE' (targets | cohorts | cases | assoc | score | all)" ;;
esac
