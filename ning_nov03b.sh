#!/usr/bin/env bash
# Ning's 2026-11-03 request: the 170 genes of novel_genes_id_1103.tsv against
# every UKB trait the phenotype directory offers, in All, F and M, on all 60
# embeddings. F test for quantitative traits, logistic LRT for binary ones.
#
# Traits come from a deep scan of UKBV_PHENO_DIR (trait_scan.py): every table
# with a participant id, every field that is quantitative or binary after the
# rules in trait_rules.py. The 8 FM-GWAS traits are included with their
# curated labels (hpp_1103, scheme 1 for the diseases).
#
# As Ning suggested, each gene's PCA (variance 0.95) is fitted once and saved
# (gene_pca_cache.py) -- separately for All, F and M -- then reused for every
# trait of that sex (gene_scan.py).
#
#   bash ning_nov03b.sh genes     # gene list + features check (no GPU stage)
#   bash ning_nov03b.sh traits    # deep scan -> trait catalog + values
#   bash ning_nov03b.sh pca       # per-gene PCA scores, saved once
#   bash ning_nov03b.sh scan      # all tests
#   bash ning_nov03b.sh summary   # tables, top hits, and the zip for Ning
#                                 #   (UKBV_OUT_ROOT/ukbb_gene_scan_1103.zip)
#   bash ning_nov03b.sh eta       # progress and time left of a running scan
#   bash ning_nov03b.sh           # genes, traits, pca, scan, summary
#
# Sensitivity analysis (not in Ning's method): the same scan with genetic PCs
# 1..GENETIC_PCS as extra covariates, to flag hits driven by ancestry.
#   bash ning_nov03b.sh scan_gpc      # -> results_gpc10
#   bash ning_nov03b.sh summary_gpc
#
# CPU host with the cluster paths (paths.env). Output under
# UKBV_OUT_ROOT/gene_scan_1103. Every stage resumes.
set -uo pipefail
export PYTHONUNBUFFERED=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
source "${HERE}/paths.sh"
PHENO_PY="${CONDA_ROOT}/envs/microbiome/bin/python3"
ASSOC_PY="${CONDA_ROOT}/envs/fmgwas_assoc/bin/python3"

# Another gene list can reuse this scan (ning_nov11.sh does): SCAN_* override
# the list, the run directory, the trait catalog and the zip.
GENES="${SCAN_GENES:-${HERE}/novel_genes_id_1103.tsv}"
EXPECTED_GENES="${SCAN_EXPECTED_GENES:-170}"
RUN="${SCAN_RUN:-${OUT_ROOT}/gene_scan_1103}"
SCAN_DIR="${SCAN_TRAITS:-${RUN}/traits}"
ZIP_NAME="${SCAN_ZIP:-ukbb_gene_scan_1103}"
ZIP_TITLE="${SCAN_TITLE:-UKB scan: 170 genes (novel_genes_id_1103) x every UKB trait, All / F / M}"
CACHE_DIR="${RUN}/gene_pca"
RESULTS="${RUN}/results"
GENETIC_PCS=10
RESULTS_GPC="${RUN}/results_gpc${GENETIC_PCS}"
FEATURE_TABLE_ROOT="${OUT_ROOT}/feature_tables"
COHORT_DIR="${OUT_ROOT}/cohorts_hpp1103_icd_selfrep"   # built by ning_nov03.sh
SPARE_CORES=2
PCA_THREADS=8        # each PCA rank holds one gene's 60 full feature matrices
SCAN_THREADS=2       # scan matrices are N x k, small: more ranks, fewer threads
TRAIT_WORKERS=16

log() { printf '[%s] [ning_nov03b] %s\n' "$(date +'%F %T')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

cores() { nproc 2>/dev/null || echo 8; }

PROGRESS_SECONDS=60   # how often the live ETA line is printed

fmt_hours() { printf '%dh%02dm' $(( $1 / 3600 )) $(( $1 % 3600 / 60 )); }

# One progress line: $1 name, $2 done, $3 total, $4 done at start, $5 start time.
# The rate counts only work done in this run, so a resumed run is not
# credited with what earlier runs did.
progress_line() {
    local name="$1" done="$2" total="$3" done0="$4" t0="$5"
    local now elapsed pct rate_done left eta finish
    now=$(date +%s); elapsed=$(( now - t0 ))
    pct=$(( total > 0 ? 100 * done / total : 100 ))
    rate_done=$(( done - done0 ))
    if (( rate_done > 0 )); then
        left=$(( total - done ))
        eta=$(( elapsed * left / rate_done ))
        finish=$(date -d "@$(( now + eta ))" +'%a %H:%M' 2>/dev/null \
                 || date -r "$(( now + eta ))" +'%a %H:%M')
        log "[ETA] $name: $done/$total ($pct%), elapsed $(fmt_hours "$elapsed")," \
            "ETA $(fmt_hours "$eta") -> done ~$finish"
    else
        log "[ETA] $name: $done/$total ($pct%), elapsed $(fmt_hours "$elapsed")," \
            "ETA after the first unit finishes"
    fi
}

# run_ranks THREADS NAME TOTAL COUNT_FN CMD...: runs "CMD --rank k --world-size n"
# for k = 0..n-1, n = usable cores / THREADS, and prints the overall ETA every
# PROGRESS_SECONDS from COUNT_FN (units done so far) against TOTAL.
run_ranks() {
    local threads="$1" name="$2" total="$3" count_fn="$4"; shift 4
    local n=$(( ($(cores) - SPARE_CORES) / threads )) k failed=0 pid alive
    local -a pids=()
    (( n < 1 )) && n=1
    mkdir -p "${RUN}/logs"
    log "$name: $n rank(s) x $threads thread(s); logs in ${RUN}/logs/${name}.rank*.log"
    local t0 done0
    t0=$(date +%s); done0=$("$count_fn")
    # For `bash ning_nov03b.sh eta` from another shell: this run's start and
    # how much was already done, so its rate counts only this run's work.
    printf '%s %s %s\n' "$t0" "$done0" "$total" > "${RUN}/.${name}_progress"
    for (( k = 0; k < n; k++ )); do
        OMP_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads" \
        MKL_NUM_THREADS="$threads" NUMEXPR_NUM_THREADS="$threads" \
            "$@" --rank "$k" --world-size "$n" > "${RUN}/logs/${name}.rank${k}.log" 2>&1 &
        pids+=("$!")
    done
    progress_line "$name" "$done0" "$total" "$done0" "$t0"
    while :; do
        alive=0
        for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1; done
        (( alive )) || break
        sleep "$PROGRESS_SECONDS"
        progress_line "$name" "$("$count_fn")" "$total" "$done0" "$t0"
    done
    # Each pid on its own: the cluster's older bash has no `wait -n`.
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=$(( failed + 1 ))
    done
    progress_line "$name" "$("$count_fn")" "$total" "$done0" "$t0"
    (( failed == 0 )) || die "$name: $failed rank(s) failed -- see ${RUN}/logs; rerun resumes"
}

n_genes() { echo $(( $(wc -l < "$GENES") - 1 )); }
n_traits() { echo $(( $(wc -l < "${SCAN_DIR}/trait_catalog.tsv") - 1 )); }
# gene_scan.py writes one part per gene and block of TRAITS_PER_PART traits.
TRAITS_PER_PART=50
# Genes without a feature table (stage_genes) are never scanned: not counted.
n_untested() { [[ -f "${RUN}/genes_without_features.txt" ]] && grep -c . "${RUN}/genes_without_features.txt" || echo 0; }
n_parts() { echo $(( ($(n_genes) - $(n_untested)) * ( ($(n_traits) + TRAITS_PER_PART - 1) / TRAITS_PER_PART ) )); }
count_pca() { echo $(( $(find "$CACHE_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l) )); }
count_parts() { echo $(( $(find "$1/per_gene" -name 'part*.tsv.gz' 2>/dev/null | wc -l) )); }
count_scan() { count_parts "$RESULTS"; }
count_scan_gpc() { count_parts "$RESULTS_GPC"; }

stage_genes() {
    [[ -f "$GENES" ]] || die "missing $GENES"
    local n=$(( $(wc -l < "$GENES") - 1 ))
    (( n == EXPECTED_GENES )) || die "$n genes in $GENES, expected $EXPECTED_GENES"
    local missing
    missing=$("${PHENO_PY}" cell_features.py "$GENES" "$FEATURE_TABLE_ROOT")
    if [[ -n "$missing" ]]; then
        mkdir -p "$RUN"
        printf '%s\n' "$missing" > "${RUN}/genes_without_features.txt"
        log "$(wc -l < "${RUN}/genes_without_features.txt") gene(s) have no feature table" \
            "(listed in ${RUN}/genes_without_features.txt); they are reported, not tested"
    else
        log "all $n genes have a feature table"
    fi
}

stage_traits() {
    [[ -d "$COHORT_DIR" ]] || die "missing $COHORT_DIR (run ning_nov03.sh cohorts)"
    if [[ -f "${SCAN_DIR}/trait_catalog.tsv" ]]; then
        log "trait catalog exists: $(( $(wc -l < "${SCAN_DIR}/trait_catalog.tsv") - 1 )) traits"
        return 0
    fi
    "${PHENO_PY}" trait_scan.py --cohort-dir "$COHORT_DIR" --out "$SCAN_DIR" \
        --workers "$TRAIT_WORKERS" || die "trait_scan.py failed"
}

stage_pca() {
    [[ -f "${SCAN_DIR}/participants.tsv" ]] || die "run the traits stage first"
    run_ranks "$PCA_THREADS" pca "$(n_genes)" count_pca "${ASSOC_PY}" gene_pca_cache.py --genes "$GENES" \
        --participants "${SCAN_DIR}/participants.tsv" \
        --feature-root "$FEATURE_TABLE_ROOT" --cache "$CACHE_DIR"
}

stage_scan() {
    mkdir -p "$RUN"
    log "scan: $(n_traits) traits x 3 sexes x $(n_genes) genes x 60 embeddings" \
        "= $(n_parts) parts of $TRAITS_PER_PART traits"
    run_ranks "$SCAN_THREADS" scan "$(n_parts)" count_scan "${ASSOC_PY}" gene_scan.py \
        --scan "$SCAN_DIR" --cache "$CACHE_DIR" --genes "$GENES" --out "$RESULTS" \
        --traits-per-part "$TRAITS_PER_PART"
}

stage_scan_gpc() {
    mkdir -p "$RUN"
    log "scan_gpc: as scan, plus genetic PCs 1-${GENETIC_PCS} as covariates -> $RESULTS_GPC"
    run_ranks "$SCAN_THREADS" scan_gpc "$(n_parts)" count_scan_gpc "${ASSOC_PY}" gene_scan.py \
        --scan "$SCAN_DIR" --cache "$CACHE_DIR" --genes "$GENES" --out "$RESULTS_GPC" \
        --traits-per-part "$TRAITS_PER_PART" --genetic-pcs "$GENETIC_PCS"
}

# Re-reading every part is the slow step; reuse the cell table when no part is
# newer than it (the collapse and flags are rebuilt either way).
summarize() {
    local out="$1" table="$1/gene_trait_scan.tsv"
    local -a reuse=()
    if [[ -f "$table" ]] && [[ -z "$(find "$out/per_gene" -name 'part*.tsv.gz' -newer "$table" -print -quit)" ]]; then
        log "summary: no part newer than $table, rebuilding from it"
        reuse=(--from-summary)
    fi
    "${PHENO_PY}" summarize_gene_scan.py --out "$out" ${reuse[@]+"${reuse[@]}"} \
        --catalog "${SCAN_DIR}/trait_catalog.tsv" --scan "$SCAN_DIR" || die "summary failed"
    "${PHENO_PY}" top_hits.py "$out" || die "top_hits failed"
    "${PHENO_PY}" package_gene_scan.py --results "$out" --zip "${OUT_ROOT}/$2.zip" \
        --gtf "$GRCH37_GTF" --title "$3" --genetic-pcs "${4:-0}" || die "packaging failed"
}

stage_summary() {
    summarize "$RESULTS" "$ZIP_NAME" "$ZIP_TITLE"
}
stage_summary_gpc() {
    summarize "$RESULTS_GPC" "${ZIP_NAME}_gpc${GENETIC_PCS}" \
        "UKB scan, sensitivity: + genetic PCs 1-${GENETIC_PCS} as covariates" "$GENETIC_PCS"
}

# Progress of whichever stage is running in another shell, from the files on
# disk and the start marker run_ranks leaves (newest of scan/pca, else traits).
stage_eta() {
    local marker name t0 done0 total
    # Newest marker first: a rerun's pca stage is not hidden by an old scan one.
    for marker in $(ls -t "${RUN}/.scan_gpc_progress" "${RUN}/.scan_progress" \
                          "${RUN}/.pca_progress" 2>/dev/null); do
        name="$(basename "$marker" _progress)"; name="${name#.}"
        read -r t0 done0 total < "$marker"
        progress_line "$name" "$("count_$name")" "$total" "$done0" "$t0"
        return 0
    done
    if [[ -f "${SCAN_DIR}/trait_catalog.tsv" ]]; then
        log "traits done ($(n_traits)); pca not started yet"
    else
        log "traits stage running (or not started): its [ETA] lines are in its own terminal"
    fi
}

STAGE="${1:-all}"
log "stage: $STAGE"
case "$STAGE" in
    genes)   stage_genes ;;
    traits)  stage_traits ;;
    pca)     stage_pca ;;
    scan)    stage_scan ;;
    summary) stage_summary ;;
    scan_gpc)    stage_scan_gpc ;;
    summary_gpc) stage_summary_gpc ;;
    eta)     stage_eta ;;
    all)     stage_genes; stage_traits; stage_pca; stage_scan; stage_summary ;;
    -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown stage '$STAGE' (genes | traits | pca | scan | summary | scan_gpc |" \
           "summary_gpc | eta | all)" ;;
esac
