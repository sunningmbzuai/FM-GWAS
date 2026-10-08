# Shared path setup for the shell drivers. Source it; do not run it.
#
# Reads the same UKBV_* variables as config.py: from the environment, or from
# paths.env next to this file (environment wins). See paths.env.example.
# UKBV_OUT_ROOT, UKBV_CONDA_ROOT and UKBV_FMGWAS_ROOT are needed by every driver; the rest
# have defaults derived from them or are checked by the stage that uses them.

UKBV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${UKBV_DIR}/paths.env" ]]; then
    # Keep anything already exported: the environment overrides the file.
    while IFS='=' read -r _key _value; do
        _key="${_key#export }"
        _key="${_key//[[:space:]]/}"
        [[ -z "$_key" || "$_key" == \#* ]] && continue
        if [[ -z "${!_key:-}" ]]; then
            _value="${_value%% #*}"
            _value="${_value%"${_value##*[![:space:]]}"}"
            _value="${_value%\"}"; _value="${_value#\"}"
            _value="${_value%\'}"; _value="${_value#\'}"
            eval "export ${_key}=\"${_value}\""
        fi
    done < "${UKBV_DIR}/paths.env"
    unset _key _value
fi

: "${UKBV_OUT_ROOT:?set UKBV_OUT_ROOT (output directory) in the environment or paths.env}"
: "${UKBV_CONDA_ROOT:?set UKBV_CONDA_ROOT (conda root holding the pipeline envs) in the environment or paths.env}"

CONDA_ROOT="$UKBV_CONDA_ROOT"
OUT_ROOT="$UKBV_OUT_ROOT"
: "${UKBV_FMGWAS_ROOT:?set UKBV_FMGWAS_ROOT (checkout of the FM-GWAS repository) in the environment or paths.env}"
FMGWAS_ROOT="$UKBV_FMGWAS_ROOT"
HPP_UNION_GENE_DIR="${FMGWAS_ROOT}/HPP_union_gene"
MODEL_PATH="${FMGWAS_ROOT}/TransUNetWithTracks"
UKB_MAIN_PARQUET="${UKBV_PHENO_DIR:+${UKBV_PHENO_DIR}/ukb676772.parquet}"
GRCH37_GTF="${UKBV_GRCH37_GTF:-${OUT_ROOT}/reference/gencode.v19.annotation.gtf}"
EAGLE_LOCAL_DIR="${UKBV_EAGLE_DIR:-${OUT_ROOT}/eagle_local}"
COMPILED_BINARY="${UKBV_COMPILED_BINARY:-}"
