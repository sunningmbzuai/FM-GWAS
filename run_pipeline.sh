#!/usr/bin/env bash
# Full UKB replication run: GTF -> Eagle phasing -> cohorts -> gene sequences
# -> FM-GWAS embeddings -> FM-GWAS association test -> recovery rate, for all
# 8 traits x {All,F,M} (22 cells, not 24: HPP has no significant male genes for
# Hypertension or Obesity, so those two have no gene list upstream).
#
# The embedding stage is NOT run here -- it needs an A100 on a different
# cluster. Run run_embeddings.sh there, then re-run this script. That GPU pass
# also exports the feature tables assoc reads (export_features.py), so the GPU
# side ends at "tensors -> plain arrays" and every statistical stage, assoc
# included, happens here. Assoc therefore runs early in the second pass, right
# after the cohorts and gene lists it needs.
#
# The validation endpoint is the recovery rate: the fraction of the
# HPP-significant genes (HPP_union_gene/<trait>_extreme[_F|_M].tsv) that come
# out significant at p_F_analytic < 2.5e-6 in UKB. MAGMA, GSEA and Open Targets
# are paper-only analyses with no code in the FM-GWAS repo and are out of scope.
#
# Resumable: each stage is guarded by a marker file under
# OUT_ROOT/.stage_markers/. Re-running this script skips any stage whose
# marker exists, so a killed/partial run (or a deliberate re-run after fixing
# one stage) does not redo finished work. Force a stage to redo by deleting
# its marker (see bottom of file).
#
# Trait data (UKB phenotype parquet, HPP_union_gene/*_extreme*.tsv) may not be
# staged yet. Phasing has no dependency on it, so it runs first (it's also
# the long pole). Everything downstream of it -- cohorts, gene sequences,
# embeddings, association -- DEFERS (logs and moves on, no marker written,
# no hard failure) if its trait inputs aren't there yet. Re-running this
# script later picks deferred stages back up automatically once the data
# shows up.
#
# One conda env per stage, resolved off UKBV_CONDA_ROOT.
#
# GPU_ENV (deep_learning) was compiled specifically for this model -- NEVER
# pip/conda install into it. If a required package is missing there, this
# script fails loudly instead of touching the env.
set -uo pipefail  # NOTE: no -e -- deferrable stages rely on handling their own failures

# Unbuffered stdout for every stage: python buffers when redirected to a log,
# so a long stage's progress lines would only appear once it exits -- which
# makes a slow run indistinguishable from a hung one.
export PYTHONUNBUFFERED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
PREP_PY="${CONDA_ROOT}/envs/wasp_env/bin/python3"
# wasp_env has no pyarrow; the cohort stage reads UKB parquet, so it runs in
# microbiome instead. Neither env is modified to accommodate the other.
PHENO_PY="${CONDA_ROOT}/envs/microbiome/bin/python3"
GPU_PY="${CONDA_ROOT}/envs/deep_learning/bin/python3"

# The model checkpoint, compiled binary and padding length are used only by
# the embedding stage, which lives in run_embeddings.sh on the GPU cluster.
# Deliberately not duplicated here, so the two cannot drift apart.

COHORT_DIR="${OUT_ROOT}/cohorts"
GENE_SEQ_DIR="${OUT_ROOT}/gene_sequences"
FEATURE_ROOT="${OUT_ROOT}/embeddings"
FEATURE_TABLE_ROOT="${OUT_ROOT}/feature_tables"
# Versioned: FM-GWAS d17eae23a6 (95%-variance PCA, raw covariates) changes
# every p-value, and insample_hpp_AG.py skips genes whose .tsv exists, so
# the old 50-PC results must not share a directory with the new ones.
ASSOC_DIR="${OUT_ROOT}/assoc_results_pca95"
TARGET_GENE_DIR="${OUT_ROOT}/target_gene_lists"
# Mirrors config.RESTRICT_TO_TARGET_GENES: read it from config.py rather than
# hardcoding it here, so the bash and python halves cannot disagree about which
# genes the run covers. run_assoc.sh reads the same flag the same way.
RESTRICT_TO_TARGET_GENES=$("${PHENO_PY}" -c \
    'import config; print("1" if config.RESTRICT_TO_TARGET_GENES else "0")' 2>/dev/null || echo 0)
PHASED_VCF="${OUT_ROOT}/phasing/all.norm.phased.vcf.gz"

MARKER_DIR="${OUT_ROOT}/.stage_markers"
mkdir -p "$MARKER_DIR"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*" >&2; }

# run STAGE_NAME "description" cmd...   -- skips if MARKER_DIR/STAGE_NAME.done
# exists; hard-fails the whole script on error (use for required stages).
run_stage() {
    local name="$1" desc="$2"; shift 2
    local marker="${MARKER_DIR}/${name}.done"
    if [[ -f "$marker" ]]; then
        log "[skip] ${name}: ${desc} (marker exists: $marker)"
        return 0
    fi
    log "[run]  ${name}: ${desc}"
    "$@" || { log "ERROR: ${name} failed"; exit 1; }
    touch "$marker"
}

# run_stage_optional STAGE_NAME "description" cmd...   -- same, but a failure
# only logs + defers (no marker written, no exit); next run retries it.
run_stage_optional() {
    local name="$1" desc="$2"; shift 2
    local marker="${MARKER_DIR}/${name}.done"
    if [[ -f "$marker" ]]; then
        log "[skip] ${name}: ${desc} (marker exists: $marker)"
        return 0
    fi
    log "[run]  ${name}: ${desc}"
    if "$@"; then
        touch "$marker"
    else
        log "[defer] ${name}: ${desc} -- failed, will retry on next run (no marker written)"
    fi
}

# ---------------------------------------------------------------------------
run_stage gtf "GRCh37 GTF" \
    bash setup_grch37_gtf.sh

run_stage validate_reference "check fasta/GTF/UKB array genotypes agree on build (wasp_env, pysam)" \
    "${PREP_PY}" validate_ukb_build.py

run_stage eagle_setup "stage Eagle binary + genetic map" \
    bash setup_eagle.sh

run_stage eagle_phase "phase QC-passed ukb_snp_all samples" \
    bash run_eagle_phasing_ukb.sh

# ---------------------------------------------------------------------------
# Everything below here needs trait data (UKB phenotypes, HPP_union_gene/).
# Defer, don't fail, if it isn't staged yet.
# ---------------------------------------------------------------------------
if [[ ! -f "$UKB_MAIN_PARQUET" ]]; then
    log "[defer] cohorts: $UKB_MAIN_PARQUET not readable -- skipping, will retry next run"
elif [[ ! -d "$HPP_UNION_GENE_DIR" ]] || ! ls "${HPP_UNION_GENE_DIR}"/*_extreme*.tsv >/dev/null 2>&1; then
    log "[defer] cohorts: no *_extreme*.tsv under $HPP_UNION_GENE_DIR yet -- skipping, will retry next run"
elif ! "${PHENO_PY}" -c "import pyarrow" 2>/dev/null; then
    log "ERROR: pyarrow missing from the ${PHENO_PY} env -- build_cohorts.py cannot read UKB parquet."
else
    run_stage_optional cohorts "cohorts from UKB 676772 phenotypes (microbiome env, pyarrow)" \
        "${PHENO_PY}" build_cohorts.py
fi

# Key-association mode: filter the HPP gene lists down to the target genes once,
# up front, so both the gene-sequence stage and the assoc stage see the same
# short list. Re-run every time (cheap, and the target list can change).
if [[ "$RESTRICT_TO_TARGET_GENES" == "1" ]]; then
    if [[ ! -d "$HPP_UNION_GENE_DIR" ]] || ! ls "${HPP_UNION_GENE_DIR}"/*_extreme*.tsv >/dev/null 2>&1; then
        log "[defer] target_gene_lists: no *_extreme*.tsv under $HPP_UNION_GENE_DIR yet"
    else
        log "[run]  target_gene_lists: filter HPP gene lists to hpp_top_novel_genes.tsv"
        "${PHENO_PY}" target_genes.py \
            || log "[warn] target_gene_lists: failed -- assoc will fall back to the full lists"
    fi
fi

# ---------------------------------------------------------------------------
# Assoc. First thing after the GPU trip, because by now everything it needs is
# on disk: the cohorts and gene lists above, and the feature tables the GPU
# pass exported. It reads those tables with numpy, so it runs HERE rather than
# holding a GPU -- the .pt files stay on the GPU cluster's side of the split.
# ---------------------------------------------------------------------------
N_TABLES=$(find "$FEATURE_TABLE_ROOT" -name '*.npz' 2>/dev/null | wc -l)
N_GENE_SEQS=$(ls -1 "$GENE_SEQ_DIR" 2>/dev/null | wc -l)

if (( N_TABLES == 0 )); then
    if (( N_GENE_SEQS == 0 )); then
        log "[defer] assoc: no gene sequences yet, so nothing to embed or test"
    else
        log "[defer] assoc: no feature tables under $FEATURE_TABLE_ROOT"
        log "        $N_GENE_SEQS gene sequences are ready. On the GPU cluster run:"
        log "            bash run_embeddings.sh                    # single GPU"
        log "            bash run_embeddings.sh --world_size 4 --rank 0   # sharded"
        log "        That embeds and exports the tables; then re-run this script."
    fi
else
    if (( N_TABLES < N_GENE_SEQS )); then
        log "[warn] features: $N_TABLES/$N_GENE_SEQS genes exported -- assoc will"
        log "       run on the genes present and skip the rest. Finish the"
        log "       remaining shards on the GPU cluster for a complete result."
    else
        log "[ok]   features: $N_TABLES/$N_GENE_SEQS genes exported"
    fi

    # One definition of an assoc job, shared with run_assoc.sh's direct users.
    # It exits 3 when the features on disk cannot be read here at all, which
    # means the GPU pass never exported the tables.
    bash run_assoc.sh
    case $? in
        0) : ;;
        3) log "[defer] assoc: deferred (see above) -- export the tables on the" \
               "GPU node: python export_features.py" ;;
        *) log "[warn] assoc: some trait/sex cells failed -- re-run to retry" ;;
    esac
fi

if [[ ! -s "$PHASED_VCF" ]]; then
    log "[defer] gene_sequences: $PHASED_VCF not ready yet (eagle_phase incomplete) -- skipping, will retry next run"
elif [[ ! -d "$HPP_UNION_GENE_DIR" ]] || ! ls "${HPP_UNION_GENE_DIR}"/*_extreme*.tsv >/dev/null 2>&1; then
    log "[defer] gene_sequences: no *_extreme*.tsv under $HPP_UNION_GENE_DIR yet -- skipping, will retry next run"
elif ! "${PREP_PY}" -c "import pysam" 2>/dev/null; then
    log "ERROR: pysam missing from wasp_env ($PREP_PY). Install it there manually:"
    log "  ${CONDA_ROOT}/envs/wasp_env/bin/pip install pysam"
else
    run_stage_optional gene_sequences "gene sequences from phased VCF (wasp_env, pysam)" \
        "${PREP_PY}" build_gene_sequences.py
fi

# ---------------------------------------------------------------------------
# Validation: recovery rate of the HPP-significant genes in UKB. Re-run every
# time (no marker) -- it is seconds of work and its answer changes as more
# trait/sex assoc jobs land.
# ---------------------------------------------------------------------------
if [[ -d "$ASSOC_DIR" ]] && ls "$ASSOC_DIR"/*/*.tsv >/dev/null 2>&1; then
    log "[run]  recovery: HPP-significant gene recovery rate in UKB"
    "${PHENO_PY}" compute_recovery.py \
        || log "[warn] recovery: failed -- association results may be partial"
else
    log "[defer] recovery: no association results under $ASSOC_DIR yet"
fi

log "Done for this run. Results (so far) under ${ASSOC_DIR}/<trait>_<sex>/<gene_id>.tsv"
log "Re-run this script any time to pick up deferred stages once trait data lands."
log "To force a completed stage to redo: rm ${MARKER_DIR}/<stage_name>.done"
