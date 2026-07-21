# AIDO.DNA3-AG (TransUNetWithTracks) — compiled binary for A100

A compiled **AOTInductor** build of the `TransUNetWithTracks` model in the
parent directory, recompiled specifically for **NVIDIA A100 (compute
capability 8.0 / sm_80)**. The existing `../compiled/*.pt2` binaries are baked
for sm_90 (H100/H200) only and will not run here — AOTInductor embeds
arch-specific GPU kernels with no PTX fallback (see `../compiled/README.md`).
Weights are embedded in **compiled form**, same as the H100 build.

## Files

**Recommended:**
- `aido_dna3ag_1bp4_dyn_a100.pt2` — **genuinely dynamic**: batch 1–32,
  length any multiple of 128 in [256, 128000]. One binary covers everything
  from short SNP-flanking windows to whole genes, at whatever the real input
  length actually is (no fixed-`L` padding waste). Built with a *different*
  environment than everything else in this directory — see "Two environments"
  below.

**Fixed-shape fallbacks** (built on the more battle-tested torch 2.6.0 /
flash-attn 2.5.6 combo; kept in case the dynamic build's newer env is ever a
problem):
- `aido_dna3ag_1bp4_L8192_a100.pt2` — static build, batch=1, L=8192 exactly.
- `aido_dna3ag_1bp4_L8192_dynbatch_a100.pt2` — L=8192 fixed, batch dynamic 1–32.
- `aido_dna3ag_1bp4_L128000_dynbatch_a100.pt2` — L=128000 fixed, batch dynamic 1–8.

**Support code:**
- `aido_dna3ag_binary.py` — drop-in wrapper (same interface as
  `../compiled/aido_dna3ag_binary.py`) for all four `.pt2` files above.
  `load(path)` auto-detects the dynamic build from its filename; for the
  fixed-shape builds pass `max_length=<L>`.
- `export_patches.py`, `build_static_L8192.py`, `build_dynbatch_L8192.py`,
  `build_dynbatch_L128000.py`, `build_dynamic_full.py` — the exact code used
  to produce the `.pt2` files above, kept for reproducibility (not needed to
  just *use* the binaries, except that `aido_dna3ag_binary.py` inlines the one
  part of `export_patches.py` that **is** needed at runtime — see
  "Requirements" below).

## Usage
```python
import aido_dna3ag_binary

model = aido_dna3ag_binary.load("aido_dna3ag_1bp4_dyn_a100.pt2")
out = model.predict_tracks(input_ids, is_human=True)   # input_ids: (B, L) int, B<=32, L<=128000
out.cage.values      # (B, L, 546)
out.rna_seq.values   # (B, L, 667)
out.atac.values      # (B, L, 167)
out.dnase.values     # (B, L, 305)
```
`load()` returns an object with a `.predict_tracks(input_ids, is_human=True)`
method and no-op `.eval()/.cuda()/.to()` so it drops into existing
model-loading code, e.g. in place of the `AutoModel.from_pretrained(...)` call
in `get_hpp_embedding-AG_clean.py`'s `initialize_model()` — which does exactly
this when called with `--compiled_binary <path>` (shape support is inferred
from the filename automatically; `dyn` → dynamic, `_L<length>_` → fixed).

For the fixed-shape builds, pass `max_length` explicitly, e.g.
`load("aido_dna3ag_1bp4_L128000_dynbatch_a100.pt2", max_length=128000)`. For
the plain static package (`aido_dna3ag_1bp4_L8192_a100.pt2`), leave
`max_length=None` (the default) and pass inputs shaped exactly `(1, 8192)` —
no padding is applied.

## Two environments
This directory's `.pt2` files were built on two different environments,
because getting the dynamic build to compile at all required a newer PyTorch
(see "Known limitations" → fixed below):

|                          | dynamic build | fixed-shape builds |
|--------------------------|---------------|---------------------|
| conda env                | `hpp_dynlen_a100` | `hpp_lora_env` |
| PyTorch                  | 2.13.0+cu130 | 2.6.0+cu124 |
| flash-attn               | 2.8.3.post1 (built from source, see below) | 2.5.6 |
| transformers (tokenizer) | 4.57.1 | 4.57.1 |

**Whichever `.pt2` you load, the same runtime rule applies**: `flash-attn`
must be importable and match the torch/CUDA build it was compiled against —
`import aido_dna3ag_binary` in the matching env before calling
`aoti_load_package`. Mixing them (e.g. loading `..._dyn_a100.pt2` in
`hpp_lora_env`) will not work — flash-attn 2.5.6 wasn't built for torch 2.13.

`flash-attn==2.8.3.post1` for `torch==2.13.0+cu130` has no prebuilt wheel
(pip's cache serves a mismatched cu12-linked wheel that fails to import —
`libcudart.so.12: cannot open shared object file`); it must be built from
source:
```bash
conda create -n hpp_dynlen_a100 python=3.10
conda run -n hpp_dynlen_a100 pip install torch==2.13.0 einops transformers==4.57.1 pandas ninja
TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=24 conda run -n hpp_dynlen_a100 \
  pip install flash-attn --no-cache-dir --no-binary flash-attn --no-build-isolation
```
`TORCH_CUDA_ARCH_LIST="8.0"` matters a lot: without it, flash-attn compiles
kernels for sm_80/90/100/120 (4x the work) when only sm_80/A100 is needed.
Make sure `ninja` is installed *before* starting the build — without it,
flash-attn's setup.py silently falls back to a fully serial one-file-at-a-time
compile (ignoring `MAX_JOBS` entirely), turning a ~15 minute build into a
~2.5 hour one. Even correctly parallelized, expect ~15-30 minutes.

## Requirements
- **PyTorch 2.6 / CUDA 12.4** for the fixed-shape builds; **PyTorch 2.13.0 /
  CUDA 13.0** for the dynamic build (see "Two environments" above for exact
  versions and why they differ).
- **NVIDIA A100 (compute capability 8.0 / sm_80).** Will not run on sm_90
  (H100/H200) — use `../compiled/` for that.
- **flash-attn must be importable at runtime**, matching the torch build (see
  above). This is the one way this differs from the H100 build: the attention
  kernel itself was kept as the *real* flash-attn CUDA kernel rather than
  reimplemented, wrapped as a torch custom op purely so `torch.export` could
  trace through it (flash-attn has no export-visible custom-op registration
  of its own, and naively tracing through it silently produces an *incorrect*
  graph under non-strict export — see "How this was built" below).
  `aido_dna3ag_binary.py` registers that custom op's schema/implementation at
  import time; it must be imported (`import aido_dna3ag_binary`) before
  `aoti_load_package` is called on any of these `.pt2` files, or loading
  fails with `RuntimeError: Could not find schema for aido::flash_attn_kv`.
- Model source code / `transformers` are **not** needed at runtime for
  inference itself (same as the H100 build) — only `torch` and `flash_attn`.
  (`transformers` is only needed for the *tokenizer*, unrelated to the
  compiled binary.)

## Validation
Compared against the eager model (weights cast to bf16, matching how
`get_hpp_embedding-AG_clean.py` runs the model under `torch.amp.autocast(...,
dtype=torch.bfloat16)`), using real tokenized DNA (not random tokens):

**Dynamic build** (`aido_dna3ag_1bp4_dyn_a100.pt2`), across lengths spanning
the practical range including odd, non-round values (i.e. not tied to any
tier boundary — proof this is genuine dynamism, not a fixed-bucket fallback):

| L | batch | raw corr | pooled corr | TSS-500 corr |
|---|---|---|---|---|
| 256 | 8 | 0.999608 | 0.999838 | 0.999838 |
| 4096 | 8 | 0.999624 | 0.999864 | 0.999780 |
| 8192 | 8 | 0.999771 | 0.999943 | 0.999894 |
| 8320 | 8 | 0.999753 | 0.999918 | 0.999883 |
| 25344 | 8 | 0.999787 | 0.999978 | 0.999928 |
| 51712 | 8 | 0.999703 | 0.999970 | 0.999891 |
| 65536 | 4 | 0.999797 | 0.999985 | 0.999954 |
| 100096 | 4 | 0.999826 | 0.999986 | 0.999956 |
| 128000 | 2 | 0.999814 | 0.999993 | 0.999792 |

Also re-validated with random tokens across (batch, length) pairs spanning
the whole declared dynamic_shapes range (batch 1–32, L 256–128000): all exact
(max abs diff 0.0) against eager **before** the AOTInductor compile step
(proving the traced `ExportedProgram` itself is exact), and corr > 0.999 for
every pair **after** compiling to the `.pt2` binary.

**Fixed-shape builds:**
- `aido_dna3ag_1bp4_L8192_a100.pt2`, real tokenizer input, L=8192: raw tracks
  corr **0.999735**, masked-mean pooled corr **0.999937**, TSS-500 mean corr
  **0.999856**.
- `aido_dna3ag_1bp4_L8192_dynbatch_a100.pt2` re-validated at batch sizes
  1/2/4/8/16/32 (random tokens): corr 0.9993–0.9995 at every batch size.
- `aido_dna3ag_1bp4_L128000_dynbatch_a100.pt2`, real tokenizer input,
  L=128000: raw tracks corr **0.999220**, masked-mean pooled corr
  **0.999575**, TSS-500 mean corr **0.999686**; re-validated at batch sizes
  1/2/4/8 (random tokens): corr 0.9996–0.9998 at every batch size.

These numbers are in the same range as the H100 build's own reported
"downstream embeddings vs eager bf16: corr 0.9999" — the small residual gap
comes from computing everything in bf16 throughout (see below), vs. the H100
build's fp32-internal-precision design.

## Known limitations
- **Internal precision is bf16 throughout** (not fp32 like the H100 build).
  Autocast-traced fp32-weights-plus-bf16-compute was tried first and
  produced a subtly broken compiled binary (the custom attention op received
  fp32 tensors at actual AOTInductor runtime despite the exported graph
  showing bf16 — likely a constant-folding interaction between AOTInductor
  and the opaque custom op). Casting the whole model to bf16 before export
  sidesteps that fragility and matches the precision
  `get_hpp_embedding-AG_clean.py` already runs the eager model at, at the
  cost of the H100 build's extra fp32 headroom. A bf16-vs-fp32 tradeoff is
  the same one the H100 `README.md` calls out for its own optional bf16
  variant.
- **The dynamic build's L range is [256, 128000], not [128, 128000].** At the
  absolute minimum (L=128, where the bottleneck after 7 stride-2 poolings is
  a single token), `torch.export` inserts a guard that specializes tracing
  away from `k==1`, and the compiled package then rejects L=128 at runtime
  with `Guard failed: input_ids.size()[1] // 128 != 1`. Not a practically
  useful window size anyway; `k_dim = Dim("k", min=2, ...)` in
  `build_dynamic_full.py` sidesteps it by simply not claiming support for it.
- Human head only; four 1bp track types only (CAGE, RNA_SEQ, ATAC, DNASE) —
  same scope as the H100 build.
- The three fixed-shape builds always cost their full traced `L`'s compute
  even on a shorter real input (the wrapper pads up to `L`, then crops the
  output back down) — this is exactly what the dynamic build avoids, which is
  why it's the recommended default now that it exists.

### Fixed (previously a limitation of this doc): general dynamic-length build
An earlier version of this build only shipped fixed-`L` binaries, because
`torch._inductor.aoti_compile_and_package` on a graph with *both* a dynamic
batch dim and a *derived* dynamic length dim (`L = 128*k`, from
`torch.export.Dim("k") * 128` — required since the model's 7 stride-2
poolings force `L` to be a multiple of 128) failed with a C++ codegen bug in
PyTorch 2.6.0+cu124: `int64_t int_array_124[] = {s0, 128L*s2, 512L};` where
`s2` (standing in for `k`) is referenced but never declared anywhere in the
generated file — traced to `codegen_shape_tuple` → `codegen_sizevar` not
calling `ensure_size_computed` for derived symbols the way other codegen paths
do. A monkeypatch forcing that call didn't fix it (the actual declaration
site is `codegen_int_array_var`, which is `lru_cache`d on the *already
stringified* size expression, decoupled from which symbols the caching key's
string actually depends on — a deeper interaction with the "two-pass memory
planning" codegen than a one-line patch could safely fix). Reproduced in a
minimal repro (`Embedding` → `RMSNorm` → `Linear`, no flash-attn) and
confirmed **fixed in torch 2.13.0** — hence the second environment.
`build_dynamic_full.py` is the resulting script; run it in `hpp_dynlen_a100`.

## How this was built
`TransUNetWithTracks`'s attention layers (`mha.py`) use flash-attn (FA2 on
A100, since FA3/`flash_attn_interface` is Hopper-only) for both the rotary
positional embedding (a Triton kernel) and the attention kernel itself,
neither of which `torch.export` can trace through directly:
1. **Rotary embedding** — `RotaryEmbedding.forward` caches `cos`/`sin` on
   `self._cos_cached` behind a Python-level `if seqlen > self._seq_len_cached`
   check. Under export this either fails outright (strict mode: FakeTensor
   can't call `.stride()` inside the Triton kernel) or, worse, silently
   produces an *incorrect* graph (non-strict mode: the actual rotary
   computation is dropped, replaced by an uninitialized `empty_like` — this
   was caught by comparing exported vs. eager output and seeing a CUDA
   illegal-memory-access instead of matching numbers). Fix: this model always
   calls rotary with `interleaved=False`, no xpos (`scale_base=None`), and
   `seqlen_offset=0` (no KV-cache decoding) — under those conditions,
   recomputing `cos`/`sin` fresh from `torch.arange(L)` each call
   (`apply_rotary_emb_torch`, already shipped in `flash_attn.layers.rotary`
   as a reference implementation) is exact and fully traceable, including
   with a symbolic `L`.
2. **Attention kernel** — `flash_attn_kvpacked_func` (the GQA/cross-attention
   path, always used here since `num_heads_kv != num_heads` everywhere in
   this model) is wrapped as a torch custom op (`aido::flash_attn_kv`) with a
   `register_fake` meta kernel, so export treats it as an atomic,
   correctly-shaped node while still dispatching to the *real* CUDA kernel at
   runtime — same kernel, same numerics, just export-visible.
3. Two unrelated `torch.export` tracer issues in the surrounding model code
   were also patched (monkeypatched, not edited in the parent directory):
   slicing an `nn.ModuleList` with `up_to=None` in
   `TransUNetModel.get_embs` (a `torch.fx` proxy-tracer bug with `ModuleList.
   __getitem__` + slice), and a data-dependent `if mask.any():` branch in
   `BorzoiSimpleAdapter._apply_track_specific_activation` (rewritten as one
   vectorized, branch-free expression — the three masks are mutually
   exclusive and exhaustive over the channel dim, so this is exact, not an
   approximation).
4. Getting the *dynamic-length* build to actually compile (not just export)
   additionally required the newer torch — see "Fixed" above.

`export_patches.py` implements (1)-(3) as monkeypatches applied to the
already-imported model modules (no fork of the parent directory's `.py`
files) and is shared, unmodified, by both environments (confirmed
API-compatible with both flash-attn 2.5.6 and 2.8.3.post1). See
`build_static_L8192.py` / `build_dynbatch_L8192.py` / `build_dynbatch_L128000.py`
/ `build_dynamic_full.py` for the full load → patch → export →
`aoti_compile_and_package` pipelines.
