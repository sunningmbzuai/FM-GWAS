#!/usr/bin/env bash
# FM-GWAS embedding extraction -- STANDALONE, for the GPU cluster.
#
# Split out of run_pipeline.sh because this is the only stage that needs an
# A100, and it runs on a different cluster from the rest of the pipeline. It
# shares the same UKBV_OUT_ROOT, so no data has to be copied: run the
# pipeline here, run this there, come back and re-run the pipeline.
#
#   # on the GPU cluster
#   bash run_embeddings.sh                     # single GPU, whole gene set
#   bash run_embeddings.sh --world_size 4 --rank 0   # one shard of four
#
# The GPU trip is embed -> expand -> export feature tables, and stops there.
# The exported tables are plain numpy, so the association tests run back in
# run_pipeline.sh on the prep cluster rather than holding a GPU while they fit
# regressions.
#
# Sharding: each rank handles a disjoint slice of the gene set (the FM-GWAS
# script does the splitting off --world_size/--rank), so launch one process per
# GPU with the same --world_size and a distinct --rank. Ranks are independent;
# they can run on different nodes or at different times.
#
# GPU_ENV (deep_learning) was compiled specifically for this model -- NEVER
# pip/conda install into it. If a package is missing there this script fails
# loudly rather than touching the env.
set -uo pipefail

# Unbuffered stdout for every stage: python buffers when redirected to a log,
# so a long stage's progress lines would only appear once it exits -- which
# makes a slow run indistinguishable from a hung one.
export PYTHONUNBUFFERED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
GPU_PY="${CONDA_ROOT}/envs/deep_learning/bin/python3"

: "${COMPILED_BINARY:?set UKBV_COMPILED_BINARY (the portable AIDO.DNA3-AG .pt2) in the environment or paths.env}"

# The portable wrapper (aido_dna3ag_portable.py) sits in the same dir as the
# .pt2 in ukbb_validation/. The FM-GWAS script adds dirname(--compiled_binary)
# to sys.path, making the wrapper directly importable -- no extra PYTHONPATH
# needed, but defined anyway for safety.
COMPILED_WRAPPER_DIR="$(dirname "$COMPILED_BINARY")"
export PYTHONPATH="${COMPILED_WRAPPER_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# The embedding script only embeds genes listed here. target_genes.py writes a
# copy under target_gene_lists/ with the target genes the HPP list lacks; use
# it when present, so UKB-only targets are not silently skipped.
GENE_ANNOTATION="${OUT_ROOT}/target_gene_lists/candidate_genes.tsv"
[[ -f "$GENE_ANNOTATION" ]] || GENE_ANNOTATION="${HPP_UNION_GENE_DIR}/candidate_genes.tsv"
GENE_SEQ_DIR="${OUT_ROOT}/gene_sequences"
FEATURE_ROOT="${OUT_ROOT}/embeddings"
MARKER_DIR="${OUT_ROOT}/.stage_markers"
# Deduplicated path: distinct SEQUENCES only. A gene has 43,706 rows, ~37
# distinct (ref, hap1, hap2) triples, and fewer distinct haplotypes still --
# the triples are pairs drawn from that smaller pool. Sequences are packed two
# per row, so S sequences cost S forward passes. The model never sees a
# participant id and never mixes the two haplotypes, so reassembling
# afterwards is exact, not an approximation (see dedup_gene_sequences.py).
UNIQUE_GENE_SEQ_DIR="${OUT_ROOT}/gene_sequences_unique"
UNIQUE_FEATURE_ROOT="${OUT_ROOT}/embeddings_unique"
FEATURE_TABLE_ROOT="${OUT_ROOT}/feature_tables"

# Must match config.MAX_GENE_LEN. build_gene_sequences.py excludes genes over
# 128 kb, so the script's own 524288 default would pad every sequence to 4x the
# length it needs. Padded positions are masked out of pooling, so this costs
# only compute -- but 4x of it.
PADDING_LENGTH=131072

# dynamic_batch_size() in the FM-GWAS script doubles its batch for "A100",
# assuming the 80 GB part. This node reports 44.39 GiB, where a real gene at
# L~31k asks for a single 60.5 GiB activation and OOMs on the first batch.
# FMGWAS_BATCH_SIZE pins it (see fix_batch_size_override.py); empty restores
# the script's own length-derived sizing.
BATCH_SIZE=4

WORLD_SIZE=1
RANK=0
GPU="A100"
# Dedup is on by default; --no_dedup embeds every participant row, which is the
# old behaviour and ~3,500x more forward passes.
DEDUP=1
# seqs = distinct sequences (the floor); rows = distinct (ref, hap1, hap2)
# triples, the older and ~2-10x more expensive unit, kept for A/B checks.
DEDUP_MODE="seqs"
# Genes per GPU batch when deduplicating. The next batch is deduplicated on the
# CPU while the GPU embeds the current one, so dedup (minutes per gene, bound
# by the FUSE read) overlaps the model instead of running before it. Each
# batch reloads the model, so too small a batch spends its time loading.
# 0 = old behaviour: dedup everything, then one GPU call.
PREFETCH_BATCH=8

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

usage() {
    cat >&2 <<USAGE
Usage: bash run_embeddings.sh [--world_size N] [--rank R] [--gpu NAME]

  --world_size N   number of shards the gene set is split into (default 1)
  --rank R         which shard this process handles, 0 <= R < N (default 0)
  --gpu NAME       GPU name passed through to the FM-GWAS script (default A100)
  --batch_size N   pin the batch size (default 4); empty string uses the
                   script's own length-derived sizing
  --dedup_mode M   seqs (default, distinct sequences) or rows (distinct
                   (ref, hap1, hap2) triples)
  --no_dedup       embed every participant row instead of the distinct ones
  --prefetch_batch N  genes per GPU batch; the next batch is deduplicated on
                   the CPU while the GPU embeds this one (default 8; 0 =
                   dedup everything first, then one GPU call)

Several GPUs: one process per GPU, device picked by CUDA_VISIBLE_DEVICES
(--gpu is the model NAME, not a device index):
  CUDA_VISIBLE_DEVICES=1 bash run_embeddings.sh --world_size 4 --rank 1

Reads  ${GENE_SEQ_DIR}
Writes ${FEATURE_ROOT}/<gene_id>/<gene_id>.pt
       ${FEATURE_TABLE_ROOT}/<gene_id>/<gene_id>.npz  (what assoc reads)
Then run_pipeline.sh on the prep cluster runs assoc off those tables.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --world_size) WORLD_SIZE="${2:?--world_size needs a value}"; shift 2 ;;
        --rank)       RANK="${2:?--rank needs a value}"; shift 2 ;;
        --gpu)        GPU="${2:?--gpu needs a value}"; shift 2 ;;
        --batch_size) BATCH_SIZE="${2:?--batch_size needs a value}"; shift 2 ;;
        --dedup_mode) DEDUP_MODE="${2:?--dedup_mode needs a value}"; shift 2 ;;
        --no_dedup)   DEDUP=0; shift ;;
        --prefetch_batch) PREFETCH_BATCH="${2:?--prefetch_batch needs a value}"; shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *)            usage; die "unknown argument: $1" ;;
    esac
done

[[ "$WORLD_SIZE" =~ ^[0-9]+$ ]] && (( WORLD_SIZE >= 1 )) \
    || die "--world_size must be a positive integer, got '$WORLD_SIZE'"
[[ "$RANK" =~ ^[0-9]+$ ]] && (( RANK < WORLD_SIZE )) \
    || die "--rank must satisfy 0 <= rank < world_size ($WORLD_SIZE), got '$RANK'"
[[ "$PREFETCH_BATCH" =~ ^[0-9]+$ ]] \
    || die "--prefetch_batch must be a non-negative integer, got '$PREFETCH_BATCH'"

# ---------------------------------------------------------------------------
# Preflight. Fail before claiming a GPU, not after.
# ---------------------------------------------------------------------------
[[ -x "$GPU_PY" ]] || die "no python at $GPU_PY"
[[ -d "$GENE_SEQ_DIR" ]] && [[ -n "$(ls -A "$GENE_SEQ_DIR" 2>/dev/null)" ]] \
    || die "$GENE_SEQ_DIR is empty or missing -- run the gene_sequences stage of
  run_pipeline.sh on the prep cluster first."
[[ -f "$COMPILED_BINARY" ]] || die "compiled binary not found: $COMPILED_BINARY"
[[ -d "$MODEL_PATH" ]] || die "model dir not found: $MODEL_PATH"
[[ -f "${COMPILED_WRAPPER_DIR}/aido_dna3ag_portable.py" ]] \
    || die "portable wrapper not found: ${COMPILED_WRAPPER_DIR}/aido_dna3ag_portable.py"
[[ -f "$GENE_ANNOTATION" ]] || die "gene annotation not found: $GENE_ANNOTATION"
"${GPU_PY}" -c "import sklearn, torch" 2>/dev/null || die \
"sklearn/torch missing from the deep_learning env ($GPU_PY).
  Not auto-installing -- it is a precompiled env for this model. Fix it
  out-of-band."

# Deduplicate, then embed. Two schedules:
#   PREFETCH_BATCH > 0 (default, dedup on): this rank's genes go in batches;
#     the CPU deduplicates batch b+1 while the GPU embeds batch b. Sharding is
#     done here (gene i -> rank i % WORLD_SIZE over the sorted gene list), so
#     each rank only deduplicates its own genes and the FM-GWAS script is
#     called unsharded on each batch.
#   otherwise: dedup every gene, then one GPU call, sharded by the script.
# expand_embeddings.py gathers the result back to one row per participant.
EMBED_OUT="$FEATURE_ROOT"
(( DEDUP == 1 )) && EMBED_OUT="$UNIQUE_FEATURE_ROOT"
mkdir -p "$EMBED_OUT" "$MARKER_DIR"

dedup_genes() {   # $1 = directory of gene TSVs (or links to them)
    "${GPU_PY}" "${HERE}/dedup_gene_sequences.py" prepare --mode "$DEDUP_MODE" \
        --gene_dir "$1" --out_dir "$UNIQUE_GENE_SEQ_DIR"
}

# Directory of links to the given files, so a script that takes a directory
# sees only this subset -- nothing is copied. Index files ride along with
# their TSV so a deduplicated subset is still a complete input set.
link_dir() {      # $1 = dir to (re)create, rest = TSVs
    local dir="$1" tsv index; shift
    rm -rf "$dir"; mkdir -p "$dir"
    for tsv in "$@"; do
        ln -s "$tsv" "${dir}/$(basename "$tsv")"
        index="${tsv%.tsv}.index.json"
        [[ -f "$index" ]] && ln -s "$index" "${dir}/$(basename "$index")"
    done
}

embed_dir() {     # $1 = input dir, $2 = world size, $3 = rank
    FMGWAS_BATCH_SIZE="${BATCH_SIZE}" \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${GPU_PY}" \
        "${FMGWAS_ROOT}/get_hpp_embedding-AG_clean.py" \
        --model_path "${MODEL_PATH}" \
        --input_path "$1" \
        --output_path "${EMBED_OUT}" \
        --compiled_binary "${COMPILED_BINARY}" \
        --compiled_binary_max_length 128000 \
        --padding_length "${PADDING_LENGTH}" \
        --gene_annotation_file "${GENE_ANNOTATION}" \
        --gpu "${GPU}" \
        --world_size "$2" \
        --rank "$3"
}

is_embedded() {   # $1 = gene id
    [[ -f "${EMBED_OUT}/$1/$1.pt" ]]
}

N_GENES=$(ls -1 "$GENE_SEQ_DIR"/*.tsv 2>/dev/null | wc -l)
SCRATCH="${OUT_ROOT}/.embed_rank${RANK}_of${WORLD_SIZE}"
PREFETCH_PID=""
# A failed batch must not leave a dedup running on its own in the background.
trap '[[ -n "$PREFETCH_PID" ]] && kill "$PREFETCH_PID" 2>/dev/null' EXIT

log_header() {    # $1 = n to embed, $2 = n done, $3 = schedule
    log "=== FM-GWAS embedding extraction ==="
    log "Genes available : $N_GENES  ($GENE_SEQ_DIR)"
    log "To embed        : $1 gene(s) ($2 already have a .pt)"
    log "Shard           : rank $RANK of $WORLD_SIZE"
    log "Schedule        : $3"
    log "Padding length  : $PADDING_LENGTH"
    log "Batch size      : ${BATCH_SIZE:-<script default>}"
    log "Output          : $EMBED_OUT/<gene_id>/<gene_id>.pt"
}

check_done() {    # $1 = n already embedded
    # Vet what an earlier run left behind before spending a GPU on the rest: a
    # constant-embedding gene is a bug, and finding it now beats finding it
    # after the remaining genes finish. Non-fatal here -- the post-pass turns
    # it into a hard failure.
    (( DEDUP == 1 && $1 > 0 )) || return 0
    log "Checking $1 already-embedded gene(s)"
    "${GPU_PY}" "${HERE}/expand_embeddings.py" --check_only \
        --unique_root "$UNIQUE_FEATURE_ROOT" \
        --index_dir "$UNIQUE_GENE_SEQ_DIR" \
        || log "[warn] some already-embedded genes look degenerate (see above)"
}

if (( DEDUP == 1 && PREFETCH_BATCH > 0 )); then
    # This rank's outstanding genes. Sharded over the FULL sorted list, so a
    # gene's rank does not change as others finish.
    PENDING=(); n_done=0; i=0
    for tsv in "$GENE_SEQ_DIR"/*.tsv; do
        [[ -e "$tsv" ]] || continue
        (( i++ % WORLD_SIZE == RANK )) || continue
        if is_embedded "$(basename "$tsv" .tsv)"; then
            n_done=$((n_done + 1))
        else
            PENDING+=("$tsv")
        fi
    done
    n_todo=${#PENDING[@]}
    n_batches=$(( (n_todo + PREFETCH_BATCH - 1) / PREFETCH_BATCH ))
    check_done "$n_done"
    log_header "$n_todo" "$n_done" \
        "$n_batches batch(es) of <= $PREFETCH_BATCH, next batch deduplicated during each GPU pass"

    batch_raw() { link_dir "${SCRATCH}/raw$1" "${PENDING[@]:$(( $1 * PREFETCH_BATCH )):$PREFETCH_BATCH}"; }
    batch_unique() {
        local tsv unique=()
        for tsv in "${PENDING[@]:$(( $1 * PREFETCH_BATCH )):$PREFETCH_BATCH}"; do
            unique+=("${UNIQUE_GENE_SEQ_DIR}/$(basename "$tsv")")
        done
        link_dir "${SCRATCH}/unique$1" "${unique[@]}"
    }

    if (( n_batches > 0 )); then
        log "Deduplicating batch 1/$n_batches"
        batch_raw 0
        dedup_genes "${SCRATCH}/raw0" || die "deduplication failed (batch 1)"
    fi
    for (( b = 0; b < n_batches; b++ )); do
        if (( b + 1 < n_batches )); then
            batch_raw $(( b + 1 ))
            log "Deduplicating batch $(( b + 2 ))/$n_batches in the background"
            dedup_genes "${SCRATCH}/raw$(( b + 1 ))" \
                > "${SCRATCH}/dedup$(( b + 1 )).log" 2>&1 &
            PREFETCH_PID=$!
        fi
        log "Embedding batch $(( b + 1 ))/$n_batches"
        batch_unique "$b"
        embed_dir "${SCRATCH}/unique$b" 1 0 \
            || die "embedding extraction failed (batch $(( b + 1 )), rank $RANK of $WORLD_SIZE)"
        if [[ -n "$PREFETCH_PID" ]]; then
            wait "$PREFETCH_PID" \
                || die "deduplication failed (batch $(( b + 2 ))), see ${SCRATCH}/dedup$(( b + 1 )).log"
            PREFETCH_PID=""
        fi
    done
else
    INPUT_DIR="$GENE_SEQ_DIR"
    if (( DEDUP == 1 )); then
        log "Deduplicating distinct sequences -> ${UNIQUE_GENE_SEQ_DIR}"
        dedup_genes "$GENE_SEQ_DIR" || die "deduplication failed"
        INPUT_DIR="$UNIQUE_GENE_SEQ_DIR"
    fi
    # Resume: hide genes that already have a .pt from the embedding script,
    # which otherwise re-embeds them from scratch.
    TODO=(); n_done=0
    for tsv in "$INPUT_DIR"/*.tsv; do
        [[ -e "$tsv" ]] || continue
        if is_embedded "$(basename "$tsv" .tsv)"; then
            n_done=$((n_done + 1))
        else
            TODO+=("$tsv")
        fi
    done
    n_todo=${#TODO[@]}
    link_dir "${SCRATCH}/todo" "${TODO[@]}"
    check_done "$n_done"
    log_header "$n_todo" "$n_done" "one GPU call, sharded by the FM-GWAS script"
    if (( n_todo == 0 )); then
        log "Every gene already embedded -- skipping the model entirely."
    else
        embed_dir "${SCRATCH}/todo" "$WORLD_SIZE" "$RANK" \
            || die "embedding extraction failed for rank $RANK of $WORLD_SIZE"
    fi
fi
rm -rf "$SCRATCH"

# Per-rank marker: a sharded run is only complete once every rank has one, so
# run_pipeline.sh counts .pt files rather than trusting a single marker.
touch "${MARKER_DIR}/embeddings.rank${RANK}_of${WORLD_SIZE}.done"

# Fan the distinct-row embeddings back out to per-participant rows, which is
# the layout assoc reads. Done per rank: a rank's genes are complete once its
# own shard finishes, so this does not wait for the others.
if (( DEDUP == 1 )); then
    log "Expanding distinct-row embeddings -> $FEATURE_ROOT"
    "${GPU_PY}" "${HERE}/expand_embeddings.py" \
        --unique_root "$UNIQUE_FEATURE_ROOT" \
        --index_dir "$UNIQUE_GENE_SEQ_DIR" \
        --out_root "$FEATURE_ROOT" \
        || die "expanding the deduplicated embeddings failed"
fi

# The last piece of work that needs torch: turn the .pt files into plain
# arrays of the views assoc tests. Everything after this point is numpy, so
# the association stage can run on the prep cluster where the CPUs are.
log "Exporting torch-free feature tables -> $FEATURE_TABLE_ROOT"
"${GPU_PY}" "${HERE}/export_features.py" \
    --feature_root "$FEATURE_ROOT" \
    --out_root "$FEATURE_TABLE_ROOT" \
    || die "exporting the feature tables failed"

N_EMB=$(find "$FEATURE_ROOT" -name '*.pt' 2>/dev/null | wc -l)
log "=== Done (rank $RANK of $WORLD_SIZE) ==="
log "Embeddings present now: $N_EMB / $N_GENES genes"

# Assoc is deliberately NOT chained on here. It reads the tables just
# exported, which need numpy and nothing else, so it runs in run_pipeline.sh on
# the prep cluster -- no GPU held while regressions fit, and no second env with
# torch to keep working there.
if (( N_EMB < N_GENES )); then
    log "Still short of the full gene set -- expected while other ranks run."
    log "Re-run this script once the last rank finishes, to export the rest."
fi
log "Next: run_pipeline.sh on the prep cluster runs assoc off $FEATURE_TABLE_ROOT."
