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
#   gpu      run_embeddings.sh     -> embeddings + feature tables (GPU)
#   assoc    run_assoc.sh          -> association, recovery, replication (CPU),
#            then the replication tables again for each list on its own
#
#   bash ning_oct30.sh             # every stage in order, on one host
#   bash ning_oct30.sh cases       # CPU host, any time (reads phenotypes only)
#   bash ning_oct30.sh prep        # CPU host
#   bash ning_oct30.sh gpu --session S --sessions N [--gpus K]
#   bash ning_oct30.sh gpu --session S --layout K0,K1,...
#        GPU host. N sessions (nodes or allocations) of K GPUs each split the
#        genes N*K ways, one rank per GPU. K defaults to the GPUs this session
#        sees. Sessions with different GPU counts list every session's count
#        in --layout instead (same list on every session). The last session to
#        finish expands and exports every gene. With no flags: one session,
#        all its GPUs. Examples:
#          two sessions of two GPUs:
#            session A:  bash ning_oct30.sh gpu --session 0 --sessions 2
#            session B:  bash ning_oct30.sh gpu --session 1 --sessions 2
#          two sessions of two GPUs and one of one:
#            session A:  bash ning_oct30.sh gpu --session 0 --layout 2,2,1
#            session B:  bash ning_oct30.sh gpu --session 1 --layout 2,2,1
#            session C:  bash ning_oct30.sh gpu --session 2 --layout 2,2,1
#   bash ning_oct30.sh assoc       # CPU host, after every gpu session finished
#        Uses all but 2 cores, 8 BLAS threads per rank as in earlier runs, then
#        checks that genes tested in the 1019 run give byte-identical results.
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
PREVIOUS_ASSOC_DIR="${UKBV_OUT_ROOT}/assoc_results_fmgwas_89aba56_all60"
ASSOC_THREADS=8
ASSOC_SPARE_CORES=2
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

# The GPUs this session may use, as device ids: CUDA_VISIBLE_DEVICES when the
# scheduler set it, otherwise every GPU nvidia-smi lists.
session_devices() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES"
    else
        nvidia-smi -L | awk '{print NR - 1}'
    fi
}

stage_gpu() {
    local session=0 sessions=1 gpus=0 layout=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --session)  session="${2:?--session needs a value}"; shift 2 ;;
            --sessions) sessions="${2:?--sessions needs a value}"; shift 2 ;;
            --gpus)     gpus="${2:?--gpus needs a value}"; shift 2 ;;
            --layout)   layout="${2:?--layout needs a value}"; shift 2 ;;
            *) die "unknown gpu option '$1' (--session S --sessions N [--gpus K] | --session S --layout K0,K1,...)" ;;
        esac
    done
    command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 \
        || die "no GPU visible -- run 'bash ning_oct30.sh gpu' on a GPU node"
    local devices=() device
    while read -r device; do
        [[ -n "$device" ]] && devices+=("$device")
    done < <(session_devices)
    (( ${#devices[@]} > 0 )) || die "no GPU device ids found"

    # GPUs per session, one entry per session. Without --layout every session
    # has --gpus K (default: what this one sees).
    local counts=() count
    if [[ -n "$layout" ]]; then
        [[ "$layout" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] \
            || die "--layout wants GPU counts like 2,2,1, got '$layout'"
        IFS=',' read -r -a counts <<<"$layout"
        sessions=${#counts[@]}
    else
        (( gpus == 0 )) && gpus=${#devices[@]}
        for (( i = 0; i < sessions; i++ )); do counts+=("$gpus"); done
    fi
    (( sessions >= 1 && session >= 0 && session < sessions )) \
        || die "need 0 <= --session < $sessions sessions, got session $session"
    gpus=${counts[session]}
    (( gpus <= ${#devices[@]} )) \
        || die "session $session needs $gpus GPU(s), but sees ${#devices[@]}: ${devices[*]}"

    # Ranks are numbered session by session: this one starts after every GPU
    # of the sessions before it.
    local world=0 first=0 i
    for (( i = 0; i < sessions; i++ )); do
        (( i < session )) && first=$(( first + counts[i] ))
        world=$(( world + counts[i] ))
    done

    local done_dir="${UKBV_ASSOC_DIR}/.gpu_done"
    local log_dir="${UKBV_ASSOC_DIR}/gpu_logs" g rank pids=() ranks=()
    mkdir -p "$done_dir" "$log_dir"
    for (( g = 0; g < gpus; g++ )); do
        rank=$(( first + g ))
        rm -f "${done_dir}/rank${rank}_of${world}"
        log "rank $rank of $world on GPU ${devices[g]} -> ${log_dir}/rank${rank}_of${world}.log"
        CUDA_VISIBLE_DEVICES="${devices[g]}" bash run_embeddings.sh \
            --world_size "$world" --rank "$rank" --embed_only \
            > "${log_dir}/rank${rank}_of${world}.log" 2>&1 &
        pids+=("$!"); ranks+=("$rank")
    done
    local failed=0
    for i in "${!pids[@]}"; do
        if wait "${pids[i]}"; then
            touch "${done_dir}/rank${ranks[i]}_of${world}"
            log "rank ${ranks[i]} of $world done"
        else
            failed=1
            log "rank ${ranks[i]} of $world FAILED -- see ${log_dir}/rank${ranks[i]}_of${world}.log"
        fi
    done
    (( failed == 0 )) || die "rerun this same command to retry; finished genes are kept"

    # Every rank of every session done? Then this session is the last one and
    # expands and exports for all of them. The lock stops two sessions that
    # finish together from exporting at once.
    for (( rank = 0; rank < world; rank++ )); do
        if [[ ! -f "${done_dir}/rank${rank}_of${world}" ]]; then
            log "this session's ranks are done; rank $rank is still running elsewhere,"
            log "and the last session to finish expands and exports every gene"
            return 0
        fi
    done
    mkdir "${done_dir}/export.lock" 2>/dev/null \
        || { log "another session is exporting (${done_dir}/export.lock)"; return 0; }
    log "all $world ranks done: expanding and exporting every gene"
    CUDA_VISIBLE_DEVICES="${devices[0]}" bash run_embeddings.sh \
        > "${log_dir}/export.log" 2>&1
    local status=$?
    rmdir "${done_dir}/export.lock"
    (( status == 0 )) || die "expand/export failed -- see ${log_dir}/export.log"
    log "GPU stage complete; run 'bash ning_oct30.sh assoc' on the CPU host"
}

# Genes this run and the 1019 run both tested in the same cell must give
# byte-identical per-gene results: same script, settings, cohort, features
# and BLAS threads. Proves the faster schedule changed no number.
check_against_1019() {
    local same=0 differ=0 new old
    for new in "${UKBV_ASSOC_DIR}"/*_*/per_gene/*.tsv; do
        [[ -e "$new" ]] || continue
        old="${PREVIOUS_ASSOC_DIR}/${new#"${UKBV_ASSOC_DIR}"/}"
        [[ -f "$old" ]] || continue
        if cmp -s "$new" "$old"; then
            same=$(( same + 1 ))
        else
            differ=$(( differ + 1 ))
            log "[diff] ${new#"${UKBV_ASSOC_DIR}"/} differs from the 1019 run"
        fi
    done
    log "check vs 1019 run: $same per-gene result(s) identical, $differ different"
}

stage_assoc() {
    # All but ASSOC_SPARE_CORES cores, at the 8 BLAS threads per rank earlier
    # runs used; more ranks at once, the same arithmetic in each.
    local cores jobs
    cores=$(nproc 2>/dev/null || echo 8)
    jobs=$(( (cores - ASSOC_SPARE_CORES) / ASSOC_THREADS ))
    (( jobs < 1 )) && jobs=1
    log "assoc: $jobs rank(s) at once x $ASSOC_THREADS threads on $cores cores"
    # Exit 3 means the feature tables cannot be read here (not exported yet).
    bash run_assoc.sh --jobs "$jobs" --threads "$ASSOC_THREADS"
    case $? in
        0) summarize_one novel_genes_fdr_1026.tsv fdr_1026
           summarize_one known_genes_1023.tsv known_1023
           check_against_1019
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
    all)     stage_targets; stage_cases; stage_prep; stage_gpu; stage_assoc ;;
    -h|--help) sed -n '2,45p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) die "unknown stage '$STAGE' (targets | cases | prep | gpu | assoc | all)" ;;
esac
