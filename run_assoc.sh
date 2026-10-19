#!/usr/bin/env bash
# FM-GWAS association tests, one job per trait/sex -- STANDALONE.
#
# Runs Ning's script, fmgwas/insample_hpp_AG.py (FM-GWAS 89aba56, vendored
# byte for byte, see fmgwas/UPSTREAM.md), unmodified. The settings per trait
# (test, embeddings, gene PCA, covariates) come from assoc_settings.py:
# disease traits get the logistic LRT on the concat_tss* embeddings,
# quantitative traits the partial F test on all 60 views.
#
# Split out of run_pipeline.sh because it needs torch to read the .pt
# embeddings, and the only env that has torch (deep_learning) is compiled
# against the GPU cluster's glibc. On the prep cluster it dies inside numpy
# with "GLIBC_2.27 not found" before reaching any FM-GWAS code, so in practice
# this stage belongs next to the embeddings, on the GPU node.
#
# Both callers use this same file, so there is one definition of what an assoc
# job is:
#   run_embeddings.sh  runs it automatically once every gene is embedded
#   run_pipeline.sh    runs it where the env allows, and defers otherwise
#
# Direct use:
#   bash run_assoc.sh              # every trait/sex cell that is ready
#   bash run_assoc.sh --force      # ignore stage markers and redo them
#   bash run_assoc.sh --jobs 20    # pin how many ranks run at once
#
# Ranks run in parallel (see "Parallelism" below); each rank's output goes to
# ASSOC_DIR/<trait>_<sex>/assoc.rank<k>.log rather than the terminal. Per-gene
# results land in ASSOC_DIR/<trait>_<sex>/per_gene/, and the script's own
# merge writes ftest_results_merged.tsv (with BH q-values) once a cell is done.
set -uo pipefail
export PYTHONUNBUFFERED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
ASSOC_PY_ENV="${CONDA_ROOT}/envs/fmgwas_assoc/bin/python3"
GPU_PY="${CONDA_ROOT}/envs/deep_learning/bin/python3"
PHENO_PY="${CONDA_ROOT}/envs/microbiome/bin/python3"
PREP_PY="${CONDA_ROOT}/envs/wasp_env/bin/python3"

# Only HPP_union_gene (paths.sh) is still read from the FM-GWAS checkout; the
# association script itself is the vendored copy next to this file.

COHORT_DIR="${OUT_ROOT}/cohorts"
FEATURE_ROOT="${OUT_ROOT}/embeddings"
FEATURE_TABLE_ROOT="${OUT_ROOT}/feature_tables"
# Versioned with the vendored script: mirrors config.ASSOC_DIR. The
# d17eae23a6 F-test results stay in assoc_results_pca95.
ASSOC_DIR="${OUT_ROOT}/assoc_results_fmgwas_89aba56"
TARGET_GENE_DIR="${OUT_ROOT}/target_gene_lists"
MARKER_DIR="${OUT_ROOT}/.stage_markers"

FORCE=0
JOBS=0          # 0 = pick from the core count, see below
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --jobs)  JOBS="${2:?--jobs needs a value}"; shift 2 ;;
        -h|--help)
            sed -n '2,29p' "${BASH_SOURCE[0]}" >&2; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[[ "$JOBS" =~ ^[0-9]+$ ]] || { printf -- '--jobs must be a non-negative integer, got %s\n' "$JOBS" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
ASSOC_SCRIPT="${HERE}/fmgwas/insample_hpp_AG.py"

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*" >&2; }

# The assoc stage needs an env that actually RUNS ON THIS HOST: see the header
# for why deep_learning is host-specific. Never install into deep_learning to
# fix a miss -- it is precompiled for the model. $1 is a python snippet a
# usable interpreter must run: the script's imports, scipy new enough for
# eigh(subset_by_index=...), and torch only when reading .pt files.
pick_assoc_python() {
    local probe="$1" py
    for py in "$ASSOC_PY_ENV" "$GPU_PY" "$PHENO_PY" "$PREP_PY"; do
        [[ -x "$py" ]] || continue
        if "$py" -c "$probe" >/dev/null 2>&1; then
            printf '%s' "$py"
            return 0
        fi
    done
    return 1
}

# Mirrors config.RESTRICT_TO_TARGET_GENES -- read from config.py rather than
# hardcoded, so the bash and python halves cannot disagree about which genes
# the run covers. Any interpreter can answer this; config.py imports only os.
read_restrict_flag() {
    local py
    for py in "$PHENO_PY" "$PREP_PY" "$GPU_PY"; do
        [[ -x "$py" ]] || continue
        if "$py" -c 'import config; print("1" if config.RESTRICT_TO_TARGET_GENES else "0")' 2>/dev/null; then
            return 0
        fi
    done
    echo 0
}

# Exported tables are the normal input: export_features.py wrote each .pt's
# fields as numpy on the GPU node, and the script reads <gene>/<gene>.npz in
# place of <gene>/<gene>.pt, so this stage runs anywhere. Without tables it
# falls back to the .pt files, which needs torch.
PROBE_BASE="import numpy, pandas, sklearn.decomposition; from scipy.linalg import eigh; eigh(numpy.eye(2), subset_by_index=[1, 1])"
N_TABLES=$(find "$FEATURE_TABLE_ROOT" -name '*.npz' 2>/dev/null | wc -l)
N_EMBEDDINGS=$(find "$FEATURE_ROOT" -name '*.pt' 2>/dev/null | wc -l)
if (( N_TABLES > 0 )); then
    ASSOC_FEATURE_ROOT="$FEATURE_TABLE_ROOT"
    ASSOC_INPUT="$N_TABLES exported table(s)"
    ASSOC_NEEDS="$PROBE_BASE"
elif (( N_EMBEDDINGS > 0 )); then
    ASSOC_FEATURE_ROOT="$FEATURE_ROOT"
    ASSOC_INPUT="$N_EMBEDDINGS .pt file(s) -- no tables exported yet"
    ASSOC_NEEDS="$PROBE_BASE; import torch"
else
    log "[defer] assoc: nothing under $FEATURE_TABLE_ROOT or $FEATURE_ROOT yet"
    exit 3
fi

if ! ASSOC_PY="$(pick_assoc_python "$ASSOC_NEEDS")"; then
    log "[defer] assoc: no env on this host runs: ${ASSOC_NEEDS}"
    log "        Build the dedicated one: bash setup_assoc_env.sh"
    if [[ "$ASSOC_FEATURE_ROOT" == "$FEATURE_ROOT" ]]; then
        log "        Reading .pt files needs torch. Export the tables on the GPU"
        log "        node instead: python export_features.py"
    fi
    exit 3
fi

RESTRICT_TO_TARGET_GENES="$(read_restrict_flag)"
log "=== FM-GWAS association ==="
log "Script      : $ASSOC_SCRIPT"
log "Interpreter : $ASSOC_PY"
log "Features    : $ASSOC_INPUT"
log "Feature root: $ASSOC_FEATURE_ROOT"
log "Gene lists  : $([[ "$RESTRICT_TO_TARGET_GENES" == "1" ]] \
        && echo "$TARGET_GENE_DIR (key associations only)" \
        || echo "$HPP_UNION_GENE_DIR (full HPP union)")"

mkdir -p "$ASSOC_DIR" "$MARKER_DIR"

# ---------------------------------------------------------------------------
# Which cells have work. Collected first, run second: the count decides how
# many run at once, and the skip/defer lines stay in trait order instead of
# being interleaved with job output.
# ---------------------------------------------------------------------------
CELLS=()
for trait in Cholesterol Height Hyperlipidemia Hypertension Obesity Osteoporosis T2D VAT; do
  for sex in All F M; do
    case "$sex" in
      All) suffix="" ;;
      F)   suffix="_F" ;;
      M)   suffix="_M" ;;
    esac
    stage_name="assoc_fmgwas_${trait}_${sex}"
    marker="${MARKER_DIR}/${stage_name}.done"
    gene_file="${HPP_UNION_GENE_DIR}/${trait}_extreme${suffix}.tsv"
    cohort_file="${COHORT_DIR}/${trait}_${sex}.tsv"

    if (( FORCE == 0 )) && [[ -f "$marker" ]]; then
        log "[skip] ${stage_name} (marker exists)"
        continue
    fi
    # Key-association mode: test only this cell's target genes. A filtered file
    # with just a header means the cell has no target hits -- nothing to run,
    # not a missing input.
    # target_genes.py writes no file for a cell with neither an HPP list nor
    # targets, which is also "nothing to run".
    if [[ "$RESTRICT_TO_TARGET_GENES" == "1" ]]; then
        gene_file="${TARGET_GENE_DIR}/${trait}_extreme${suffix}.tsv"
        if [[ ! -f "$gene_file" ]] || (( $(wc -l < "$gene_file") <= 1 )); then
            log "[skip] ${stage_name}: no target genes for this trait/sex"
            continue
        fi
    fi
    if [[ ! -f "$gene_file" || ! -f "$cohort_file" ]]; then
        log "[defer] ${stage_name}: missing $gene_file or $cohort_file"
        continue
    fi

    CELLS+=("${trait}|${sex}|${gene_file}|${cohort_file}")
  done
done

if (( ${#CELLS[@]} == 0 )); then
    log "=== assoc done (nothing to run) ==="
    exit 0
fi

# ---------------------------------------------------------------------------
# Parallelism. The unit of work is a RANK: the script's own split of a cell's
# gene list (genes[rank::world_size]). Every gene writes its own per_gene file
# and genes with a file are skipped, so the ranks of a cell share its out_dir
# safely and a rerun only does what is left. The out_dir's run_manifest.json
# pins the gene list and settings; a changed target list needs a new ASSOC_DIR.
#
# Each worker is mostly BLAS (a Gram eigendecomposition per embedding), and
# BLAS grabs every core by default, so threads are pinned per worker.
# ---------------------------------------------------------------------------
CORES=$(nproc 2>/dev/null || echo 1)
if (( JOBS == 0 )); then
    # Leave each worker a real BLAS width; each holds a gene's feature
    # matrices in memory while it runs.
    JOBS=$(( CORES / 8 ))
    (( JOBS < 1 )) && JOBS=1
fi

# Missing-feature failures from an earlier run, for genes that have features
# now, are removed first so they count as pending (assoc_pending.py).
PENDING=(); total_pending=0
for cell in "${CELLS[@]}"; do
    IFS='|' read -r trait sex gene_file cohort_file <<<"$cell"
    out_dir="${ASSOC_DIR}/${trait}_${sex}"
    while IFS= read -r gene; do
        [[ -n "$gene" ]] && log "[retry] assoc_fmgwas_${trait}_${sex}: ${gene} has features now"
    done < <("${ASSOC_PY}" "${HERE}/assoc_pending.py" clear_stale "$gene_file" \
                 "$out_dir" "$ASSOC_FEATURE_ROOT")
    n=$("${ASSOC_PY}" "${HERE}/assoc_pending.py" count "$gene_file" "$out_dir") || n=1
    PENDING+=("$n"); total_pending=$(( total_pending + n ))
done
# Ranks in proportion to pending genes: about total_pending / JOBS genes per
# rank, so a 7-gene cell gets more workers than a 1-gene cell.
RANK_SIZE=$(( (total_pending + JOBS - 1) / JOBS ))
(( RANK_SIZE < 1 )) && RANK_SIZE=1

RANKS=()   # trait|sex|gene_file|cohort_file|rank|world_size
for i in "${!CELLS[@]}"; do
    IFS='|' read -r trait sex gene_file cohort_file <<<"${CELLS[$i]}"
    world=$(( (PENDING[i] + RANK_SIZE - 1) / RANK_SIZE ))
    # A cell with nothing pending still gets one rank: it only skips genes and
    # writes the manifest, and the merge below needs that.
    (( world < 1 )) && world=1
    for (( k = 0; k < world; k++ )); do
        RANKS+=("${trait}|${sex}|${gene_file}|${cohort_file}|${k}|${world}")
    done
    log "[plan] assoc_fmgwas_${trait}_${sex}: ${PENDING[i]} gene(s) to test in ${world} rank(s)"
done

(( JOBS > ${#RANKS[@]} )) && JOBS=${#RANKS[@]}
(( JOBS < 1 )) && JOBS=1
THREADS=$(( CORES / JOBS ))
(( THREADS < 1 )) && THREADS=1

log "Ranks       : ${#RANKS[@]} over ${#CELLS[@]} cell(s), ${JOBS} at a time x ${THREADS} thread(s) of ${CORES}"

POLL_SECONDS=5   # only used where `wait -n` is unavailable, see below

FAIL_DIR="$(mktemp -d)"
trap 'rm -rf "$FAIL_DIR"' EXIT

run_rank() {
    local trait="$1" sex="$2" gene_file="$3" cohort_file="$4" k="$5" world="$6"
    local stage_name="assoc_fmgwas_${trait}_${sex}"
    local out_dir="${ASSOC_DIR}/${trait}_${sex}"
    local rank_log="${out_dir}/assoc.rank${k}.log"
    local settings
    mkdir -p "$out_dir"
    # Test, embeddings, gene PCA and covariates for this trait: one token per
    # line, so the comma-separated embedding list stays one argument.
    if ! settings="$("${ASSOC_PY}" "${HERE}/assoc_settings.py" args "$trait")"; then
        : > "${FAIL_DIR}/${stage_name}.${k}"
        log "[fail] ${stage_name} rank ${k}: assoc_settings.py has no settings for ${trait}"
        return
    fi
    local -a settings_args=()
    while IFS= read -r token; do settings_args+=("$token"); done <<<"$settings"

    # Pinned in the child's environment, before python starts: numpy reads
    # these at import time, so setting them later would not take.
    if OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" \
       VECLIB_MAXIMUM_THREADS="$THREADS" \
       "${ASSOC_PY}" "$ASSOC_SCRIPT" run "$trait" "$sex" \
            --rank "$k" --world_size "$world" \
            --gene_tsv "$gene_file" \
            --feature_root "${ASSOC_FEATURE_ROOT}" \
            --cohort_file "$cohort_file" \
            "${settings_args[@]}" \
            --out_dir "$out_dir" \
            --no_progress >"$rank_log" 2>&1; then
        log "[ok]   ${stage_name} rank ${k}/${world}"
    else
        # One file per failure: the workers are separate processes, so a shared
        # counter variable would not survive back to this shell.
        : > "${FAIL_DIR}/${stage_name}.${k}"
        log "[fail] ${stage_name} rank ${k}/${world}: see ${rank_log}"
    fi
}

# A cell is done once none of its ranks failed AND every gene had features: a
# gene with no features gets an input_failed result, so success alone does not
# mean the cell is complete. Otherwise the next run retries it, cheaply, since
# only those genes are recomputed. Either way the script's merge runs, so
# ftest_results_merged.tsv always reflects what is on disk.
mark_cell() {
    local trait="$1" sex="$2" gene_file="$3"
    local stage_name="assoc_fmgwas_${trait}_${sex}"
    local out_dir="${ASSOC_DIR}/${trait}_${sex}"
    if compgen -G "${FAIL_DIR}/${stage_name}.*" >/dev/null; then
        log "[defer] ${stage_name}: $(ls "${FAIL_DIR}/${stage_name}".* | wc -l)" \
            "rank(s) failed -- not marked done"
        return 1
    fi
    if ! "${ASSOC_PY}" "$ASSOC_SCRIPT" merge --out_dir "$out_dir" \
            >"${out_dir}/assoc.merge.log" 2>&1; then
        log "[defer] ${stage_name}: merge failed -- see ${out_dir}/assoc.merge.log"
        return 1
    fi
    local missing status
    missing=$("${ASSOC_PY}" "${HERE}/cell_features.py" "$gene_file" \
                  "$ASSOC_FEATURE_ROOT")
    status=$?
    if (( status == 0 )); then
        touch "${MARKER_DIR}/${stage_name}.done"
        log "[done] ${stage_name}"
    elif (( status == 1 )); then
        log "[partial] ${stage_name}: $(printf '%s\n' "$missing" | wc -l)" \
            "gene(s) have no features yet -- not marked done, the next" \
            "run retries them"
    else
        log "[partial] ${stage_name}: completeness check failed -- not" \
            "marked done, the next run retries it"
    fi
}

# `wait -n` (wait for ANY child) is bash 4.3+; the cluster login nodes run an
# older bash where it fails instantly, turning the throttle below into a hot
# spin that floods the log and never blocks. Decided once, by version: running
# `wait -n` as a probe proves nothing, because with no children it fails on
# every bash.
if (( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) )); then
    HAVE_WAIT_N=1
else
    HAVE_WAIT_N=0
fi
(( HAVE_WAIT_N )) || log "Note: bash ${BASH_VERSION} has no 'wait -n'; throttling by poll"

await_slot() {
    # Block until a worker slot frees. The poll costs at most POLL_SECONDS of
    # idle per finished rank.
    while (( $(jobs -rp | wc -l) >= JOBS )); do
        if (( HAVE_WAIT_N )); then
            wait -n
        else
            sleep "${POLL_SECONDS}"
        fi
    done
}

for job in "${RANKS[@]}"; do
    IFS='|' read -r trait sex gene_file cohort_file k world <<<"$job"
    await_slot
    log "[run]  assoc_fmgwas_${trait}_${sex} rank ${k}/${world}"
    run_rank "$trait" "$sex" "$gene_file" "$cohort_file" "$k" "$world" &
done
wait

failed=0
for cell in "${CELLS[@]}"; do
    IFS='|' read -r trait sex gene_file cohort_file <<<"$cell"
    mark_cell "$trait" "$sex" "$gene_file" || failed=$(( failed + 1 ))
done

log "=== assoc done ($failed cell(s) failed) ==="
log "Next: run_pipeline.sh (either cluster) scores recovery over ${ASSOC_DIR}"
(( failed == 0 ))
