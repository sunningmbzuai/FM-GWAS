#!/usr/bin/env bash
# Build the hpp_dynlen_a100 conda env from scratch.
#
# This is the ONLY environment that can load aido_dna3ag_1bp4_dyn_a100.pt2
# (the genuinely dynamic-length A100 build). It requires torch==2.13.0
# (needed to avoid a torch 2.6.0 AOTInductor codegen bug -- see README.md's
# "Known limitations" -> "Fixed" section) and a from-source flash-attn build
# for that torch/CUDA pairing (no prebuilt wheel exists for it).
#
# Usage:
#   bash setup_hpp_dynlen_a100_env.sh
#
# Expect ~15-30 minutes total, almost all of it the flash-attn build. Safe to
# re-run (each step is idempotent / conda-create will just complain if the
# env already exists).
set -euo pipefail

ENV_NAME="hpp_dynlen_a100"

echo "=== [1/4] Creating conda env '${ENV_NAME}' (python 3.10) ==="
conda create -y -n "${ENV_NAME}" python=3.10

echo "=== [2/4] Installing torch==2.13.0 + tokenizer/data deps ==="
conda run -n "${ENV_NAME}" pip install torch==2.13.0 einops "transformers==4.57.1" pandas ninja

echo "=== [3/4] Verifying torch sees the GPU before the long flash-attn build ==="
conda run -n "${ENV_NAME}" python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not visible to torch -- check drivers/CUDA toolkit before continuing'
cap = torch.cuda.get_device_capability(0)
print(f'torch {torch.__version__}, CUDA {torch.version.cuda}, GPU compute capability {cap}')
assert cap == (8, 0), f'Expected an A100 (sm_80), got compute capability {cap}'
"

echo "=== [4/4] Building flash-attn from source (scoped to sm_80 only -- this is the long step) ==="
echo "    Without TORCH_CUDA_ARCH_LIST=8.0 this also builds sm_90/100/120 kernels (4x the work)."
echo "    ninja must already be installed (done in step 2) or the build silently falls back to a"
echo "    fully serial compile, turning ~15-30 min into ~2.5 hours."
TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS="$(( $(nproc) > 24 ? 24 : $(nproc) ))" \
  conda run -n "${ENV_NAME}" pip install flash-attn --no-cache-dir --no-binary flash-attn --no-build-isolation

echo "=== Verifying flash-attn actually runs on the GPU ==="
conda run -n "${ENV_NAME}" python -c "
import torch
from flash_attn import flash_attn_func
x = torch.randn(1, 128, 4, 32, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(x, x, x)
print('flash_attn OK, output shape', out.shape)
"

echo "=== Done. Env '${ENV_NAME}' is ready. ==="
echo "Use it as:  conda run -n ${ENV_NAME} python get_hpp_embedding-AG_clean.py --compiled_binary .../aido_dna3ag_1bp4_dyn_a100.pt2 ..."
