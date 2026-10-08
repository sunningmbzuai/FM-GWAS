#!/usr/bin/env bash
# FM-GWAS association tests, one job per trait/sex -- STANDALONE.
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
#   bash run_assoc.sh --jobs 20    # pin how many gene shards run at once
#
# Gene shards run in parallel (see "Parallelism" below); each shard's output goes to
# ASSOC_DIR/<trait>_<sex>/assoc.shard<k>.log rather than the terminal.
set -uo pipefail
export PYTHONUNBUFFERED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
ASSOC_PY_ENV="${CONDA_ROOT}/envs/fmgwas_assoc/bin/python3"
GPU_PY="${CONDA_ROOT}/envs/deep_learning/bin/python3"
PHENO_PY="${CONDA_ROOT}/envs/microbiome/bin/python3"
PREP_PY="${CONDA_ROOT}/envs/wasp_env/bin/python3"


COHORT_DIR="${OUT_ROOT}/cohorts"
FEATURE_ROOT="${OUT_ROOT}/embeddings"
FEATURE_TABLE_ROOT="${OUT_ROOT}/feature_tables"
# Versioned: FM-GWAS d17eae23a6 (95%-variance PCA, raw covariates) changes
# every p-value, and insample_hpp_AG.py skips genes whose .tsv exists, so
# the old 50-PC results must not share a directory with the new ones.
ASSOC_DIR="${OUT_ROOT}/assoc_results_pca95"
TARGET_GENE_DIR="${OUT_ROOT}/target_gene_lists"
MARKER_DIR="${OUT_ROOT}/.stage_markers"

FORCE=0
JOBS=0          # 0 = pick from the core count, see below
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --jobs)  JOBS="${2:?--jobs needs a value}"; shift 2 ;;
        -h|--help)
            sed -n '2,21p' "${BASH_SOURCE[0]}" >&2; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

[[ "$JOBS" =~ ^[0-9]+$ ]] || { printf -- '--jobs must be a non-negative integer, got %s\n' "$JOBS" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*" >&2; }

# The assoc stage torch.load()s the .pt embeddings, so it needs an env with
# torch that actually RUNS ON THIS HOST. Probe rather than assume: see the
# header for why deep_learning is host-specific. Never install into
# deep_learning to fix a miss -- it is precompiled for the model.
# $1 is the import list a usable interpreter must satisfy: with exported
# feature tables that is numpy and statsmodels, and torch only when falling
# back to the .pt files. statsmodels is always in it because deep_learning
# imports numpy and torch fine but dies in statsmodels.api on the scipy
# _lazywhere removal -- which used to fail once per trait/sex cell instead of
# being caught here.
pick_assoc_python() {
    local needs="$1" py
    for py in "$ASSOC_PY_ENV" "$GPU_PY" "$PHENO_PY" "$PREP_PY"; do
        [[ -x "$py" ]] || continue
        if "$py" -c "import ${needs}" >/dev/null 2>&1; then
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

# Exported tables are the normal input: export_features.py did the tensor work
# on the GPU node, so this stage is numpy and runs anywhere. Genes that have no
# table yet still work from their .pt, which is what needs torch.
N_TABLES=$(find "$FEATURE_TABLE_ROOT" -name '*.npz' 2>/dev/null | wc -l)
N_EMBEDDINGS=$(find "$FEATURE_ROOT" -name '*.pt' 2>/dev/null | wc -l)
if (( N_TABLES > 0 )); then
    ASSOC_FEATURE_ROOT="$FEATURE_TABLE_ROOT"
    ASSOC_INPUT="$N_TABLES exported table(s)"
    ASSOC_NEEDS="numpy, sklearn, statsmodels.api"
elif (( N_EMBEDDINGS > 0 )); then
    ASSOC_FEATURE_ROOT="$FEATURE_ROOT"
    ASSOC_INPUT="$N_EMBEDDINGS .pt file(s) -- no tables exported yet"
    ASSOC_NEEDS="numpy, sklearn, torch, statsmodels.api"
else
    log "[defer] assoc: nothing under $FEATURE_TABLE_ROOT or $FEATURE_ROOT yet"
    exit 3
fi

if ! ASSOC_PY="$(pick_assoc_python "$ASSOC_NEEDS")"; then
    log "[defer] assoc: no env on this host imports ${ASSOC_NEEDS}."
    log "        Build the dedicated one: bash setup_assoc_env.sh"
    if [[ "$ASSOC_FEATURE_ROOT" == "$FEATURE_ROOT" ]]; then
        log "        Reading .pt files needs torch. Export the tables on the GPU"
        log "        node instead: python export_features.py"
    fi
    exit 3
fi

# The assoc script reads .pt by default; this teaches it the exported tables.
# Idempotent, so it runs every time rather than being a step to remember.
"${ASSOC_PY}" "${HERE}/fix_assoc_feature_tables.py" \
    "${FMGWAS_ROOT}/insample_hpp_AG.py" \
    || { log "[defer] assoc: could not patch insample_hpp_AG.py"; exit 3; }
# Ning's 2026-10-03 update (95%-variance gene PCA, raw covariates), replayed
# onto the staged July copy; see fix_assoc_pca95.py. Also idempotent.
"${ASSOC_PY}" "${HERE}/fix_assoc_pca95.py" \
    "${FMGWAS_ROOT}/insample_hpp_AG.py" \
    || { log "[defer] assoc: could not apply the pca95 update"; exit 3; }

RESTRICT_TO_TARGET_GENES="$(read_restrict_flag)"
log "=== FM-GWAS association ==="
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
    stage_name="assoc_pca95_${trait}_${sex}"
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
    if [[ "$RESTRICT_TO_TARGET_GENES" == "1" && -f "${TARGET_GENE_DIR}/${trait}_extreme${suffix}.tsv" ]]; then
        gene_file="${TARGET_GENE_DIR}/${trait}_extreme${suffix}.tsv"
        if (( $(wc -l < "$gene_file") <= 1 )); then
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
# Parallelism. The unit of work is a SHARD: a disjoint subset of one cell's
# genes still lacking a result (shard_genes.py). A cell used to be one process
# testing gene after gene, ~45 min each at 43,680 participants, so a 70-gene
# cell took two days whatever the core count. Every gene writes its own
# <gene_id>.tsv, so a cell's shards share its save_path safely.
#
# Each worker is mostly BLAS (PCA on a 40k x ~3k matrix), and BLAS grabs every
# core by default, so threads are pinned per worker: N workers each spawning
# 80 threads spend their time fighting over cores instead of using them.
# ---------------------------------------------------------------------------
CORES=$(nproc 2>/dev/null || echo 1)
if (( JOBS == 0 )); then
    # Leave each worker a real BLAS width: the PCA parallelises, and each
    # worker holds a gene's feature matrix in memory while it runs.
    JOBS=$(( CORES / 8 ))
    (( JOBS < 1 )) && JOBS=1
fi

SHARD_ROOT="${OUT_ROOT}/.assoc_shards"

# Pending genes per cell, then shards in proportion: every shard gets about
# total_pending / JOBS genes, so a 70-gene cell gets most of the workers and a
# 3-gene cell gets few.
PENDING=(); total_pending=0
for cell in "${CELLS[@]}"; do
    IFS='|' read -r trait sex gene_file cohort_file <<<"$cell"
    n=$("${ASSOC_PY}" "${HERE}/shard_genes.py" count "$gene_file" \
            "${ASSOC_DIR}/${trait}_${sex}") || n=1
    PENDING+=("$n"); total_pending=$(( total_pending + n ))
done
SHARD_SIZE=$(( (total_pending + JOBS - 1) / JOBS ))
(( SHARD_SIZE < 1 )) && SHARD_SIZE=1

SHARDS=()   # trait|sex|shard_file|cohort_file|index
for i in "${!CELLS[@]}"; do
    IFS='|' read -r trait sex gene_file cohort_file <<<"${CELLS[$i]}"
    n_shards=$(( (PENDING[i] + SHARD_SIZE - 1) / SHARD_SIZE ))
    (( n_shards < 1 )) && continue
    k=0
    while IFS= read -r shard_file; do
        [[ -n "$shard_file" ]] || continue
        SHARDS+=("${trait}|${sex}|${shard_file}|${cohort_file}|${k}")
        k=$(( k + 1 ))
    done < <("${ASSOC_PY}" "${HERE}/shard_genes.py" split "$gene_file" \
                 "${ASSOC_DIR}/${trait}_${sex}" "$n_shards" \
                 "${SHARD_ROOT}/${trait}_${sex}")
    log "[plan] assoc_pca95_${trait}_${sex}: ${PENDING[i]} gene(s) to test in ${k} shard(s)"
done

(( JOBS > ${#SHARDS[@]} )) && JOBS=${#SHARDS[@]}
(( JOBS < 1 )) && JOBS=1
THREADS=$(( CORES / JOBS ))
(( THREADS < 1 )) && THREADS=1

log "Shards      : ${#SHARDS[@]} over ${#CELLS[@]} cell(s), ${JOBS} at a time x ${THREADS} thread(s) of ${CORES}"

POLL_SECONDS=5   # only used where `wait -n` is unavailable, see below

FAIL_DIR="$(mktemp -d)"
trap 'rm -rf "$FAIL_DIR"' EXIT

run_shard() {
    local trait="$1" sex="$2" gene_file="$3" cohort_file="$4" k="$5"
    local stage_name="assoc_pca95_${trait}_${sex}"
    local save_path="${ASSOC_DIR}/${trait}_${sex}"
    local shard_log="${save_path}/assoc.shard${k}.log"
    mkdir -p "$save_path"

    # assoc_rank_capped.py runs the upstream script with PCA capped at the
    # matrix rank (see rank_capped_pca.py). Since the pca95 update the request is
    # a variance fraction, which already stops at the rank, so the cap is now a
    # guard that only acts if an integer request ever comes back.
    # Pinned in the child's environment, before python starts: numpy reads
    # these at import time, so setting them later would not take.
    if OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
       MKL_NUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" \
       "${ASSOC_PY}" "${HERE}/assoc_rank_capped.py" "${FMGWAS_ROOT}/insample_hpp_AG.py" "$trait" "$sex" 0 \
            --world_size 1 \
            --feature_root "${ASSOC_FEATURE_ROOT}" \
            --gene_file "$gene_file" \
            --cohort_file "$cohort_file" \
            --save_path "$save_path" >"$shard_log" 2>&1; then
        log "[ok]   ${stage_name} shard ${k}"
    else
        # One file per failure: the workers are separate processes, so a shared
        # counter variable would not survive back to this shell.
        : > "${FAIL_DIR}/${stage_name}.${k}"
        log "[fail] ${stage_name} shard ${k}: see ${shard_log}"
    fi
}

# A cell is done once none of its shards failed AND every gene had features:
# a gene with no features yet is skipped silently upstream, so success alone
# does not mean the cell is complete. Otherwise the next run retries it,
# cheaply, since genes whose .tsv exists are not re-sharded.
mark_cell() {
    local trait="$1" sex="$2" gene_file="$3"
    local stage_name="assoc_pca95_${trait}_${sex}"
    if compgen -G "${FAIL_DIR}/${stage_name}.*" >/dev/null; then
        log "[defer] ${stage_name}: $(ls "${FAIL_DIR}/${stage_name}".* | wc -l)" \
            "shard(s) failed -- not marked done"
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
            "run retries it"
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
    # idle per finished shard, against shards that run for hours.
    while (( $(jobs -rp | wc -l) >= JOBS )); do
        if (( HAVE_WAIT_N )); then
            wait -n
        else
            sleep "${POLL_SECONDS}"
        fi
    done
}

for shard in "${SHARDS[@]}"; do
    IFS='|' read -r trait sex gene_file cohort_file k <<<"$shard"
    await_slot
    log "[run]  assoc_pca95_${trait}_${sex} shard ${k}"
    run_shard "$trait" "$sex" "$gene_file" "$cohort_file" "$k" &
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
