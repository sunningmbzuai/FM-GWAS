#!/usr/bin/env bash
# Create the conda env the association stage runs in. Run once.
#
# Why a separate env: insample_hpp_AG.py needs torch (to load the .pt
# embeddings) AND statsmodels. deep_learning has both, but its statsmodels is
# older than the SciPy beside it and dies on import:
#     ImportError: cannot import name '_lazywhere' from 'scipy._lib._util'
# deep_learning is precompiled for the embedding model and must not be touched,
# and microbiome/wasp_env have no torch. So the assoc stage gets its own env.
#
# CPU-only torch on purpose: assoc does statistics, the GPU work is already
# finished and saved, and a CPU env runs on either cluster.
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
ENV_NAME="fmgwas_assoc"
ENV_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

if [[ -x "${ENV_PREFIX}/bin/python3" ]]; then
    log "${ENV_PREFIX} already exists -- verifying it instead of rebuilding"
else
    CONDA_EXE="${CONDA_ROOT}/bin/mamba"
    [[ -x "$CONDA_EXE" ]] || CONDA_EXE="${CONDA_ROOT}/bin/conda"
    [[ -x "$CONDA_EXE" ]] || die "no conda/mamba at ${CONDA_ROOT}/bin"

    log "Creating ${ENV_NAME} with $CONDA_EXE (a few minutes)"
    # conda supplies the interpreter and nothing else. Two reasons:
    #   - this host's conda-forge mirror has no statsmodels >= 0.14.2, which is
    #     the first release that dropped the private scipy._lib._util._lazywhere
    #     import -- the exact break above;
    #   - mixing -c pytorch with conda-forge made the solve unsatisfiable on a
    #     CUDA-13 host (cpuonly/pytorch-mutex vs __cuda).
    # PyPI has both, and pip resolves numpy/scipy/statsmodels together.
    "$CONDA_EXE" create -y -p "$ENV_PREFIX" \
        --override-channels -c conda-forge python=3.11 pip \
        || die "env creation failed"

    log "Installing torch (CPU wheel) and the stats stack with pip"
    "${ENV_PREFIX}/bin/pip" install --no-input --upgrade pip \
        || die "pip self-upgrade failed"
    "${ENV_PREFIX}/bin/pip" install --no-input \
        --index-url https://download.pytorch.org/whl/cpu torch \
        || die "pip torch install failed"
    "${ENV_PREFIX}/bin/pip" install --no-input \
        numpy scipy pandas "statsmodels>=0.14.2" scikit-learn \
        || die "pip stats stack install failed"
fi

log "Verifying the imports insample_hpp_AG.py makes"
"${ENV_PREFIX}/bin/python3" - <<'PY' || die "verification failed -- the env is not usable for assoc"
import numpy, pandas, scipy, sklearn, torch
import statsmodels.api as sm
print(f"numpy {numpy.__version__}  scipy {scipy.__version__}  "
      f"pandas {pandas.__version__}  statsmodels {sm.version.version}  "
      f"torch {torch.__version__}  sklearn {sklearn.__version__}")
PY

log "Done. run_assoc.sh picks this env up automatically."
