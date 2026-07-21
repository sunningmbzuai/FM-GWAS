"""Drop-in, A100-native binary wrapper for AIDO.DNA3-AG (TransUNetWithTracks).

Runs a compiled AOTInductor package (.pt2) built for NVIDIA A100 (sm_80),
recompiled from the parent `TransUNetWithTracks` checkpoint specifically for
this GPU generation -- the pre-existing `compiled/*.pt2` binaries in this repo
are baked for sm_90 (H100/H200) only and cannot run on A100 (see
`compiled/README.md`).

Exposes the same surface the HPP embedding script uses:

    model = aido_dna3ag_binary.load("aido_dna3ag_1bp4_dyn_a100.pt2")
    out = model.predict_tracks(input_ids, is_human=True)
    out.cage.values     # (B, L, 546)
    out.rna_seq.values  # (B, L, 667)
    out.atac.values     # (B, L, 167)
    out.dnase.values    # (B, L, 305)

Only the human head and the 4 track types the HPP script consumes are
provided; output is full-length (no crop), matching the script's length-L
masking. See README.md in this directory for how these packages were built,
their validated accuracy, and their shape support.

`aido_dna3ag_1bp4_dyn_a100.pt2` is a genuinely dynamic build: batch 1-32,
length any multiple of 128 in [256, 128000] -- built with torch==2.13.0 +
flash-attn compiled from source for that torch/CUDA pairing (see
"Requirements" below; this is a DIFFERENT env than the other, fixed-shape
`.pt2` files in this directory, which run on the more battle-tested
torch==2.6.0 / flash-attn==2.5.6 combo). `load()` auto-pads any input length
up to the nearest multiple of 128 (no more -- unlike the fixed-L packages,
which always cost the full traced L regardless of real input length) and
crops the output back down.

Runtime requirement note (differs from the H100 build): flash-attn must be
importable in this process. The attention kernel itself was kept as the real
flash-attn CUDA kernel (not reimplemented), wrapped as a torch custom op
(`aido::flash_attn_kv`) purely so torch.export could trace through it -- the
op's schema/implementation must be registered (this module does it at import
time) before the .pt2 package is loaded, or `aoti_load_package` raises
"Could not find schema for aido::flash_attn_kv".
"""
import torch

from flash_attn import flash_attn_kvpacked_func

# Channel layout of the compiled output (human, 1bp), in this order.
TRACK_ORDER = ("CAGE", "RNA_SEQ", "ATAC", "DNASE")
SPLITS = (546, 667, 167, 305)

# Shape support of the truly-dynamic build (aido_dna3ag_1bp4_dyn_a100.pt2).
DYN_LENGTH_MULTIPLE = 128
DYN_MIN_LENGTH = 256
DYN_MAX_LENGTH = 128000
DYN_MAX_BATCH = 32


def _register_flash_attn_kv_op():
    if hasattr(torch.ops, "aido") and hasattr(torch.ops.aido, "flash_attn_kv"):
        return

    torch.library.define(
        "aido::flash_attn_kv",
        "(Tensor q, Tensor kv, float softmax_scale, int window_left, int window_right) -> Tensor",
    )

    @torch.library.impl("aido::flash_attn_kv", "cuda")
    def _impl(q, kv, softmax_scale, window_left, window_right):
        return flash_attn_kvpacked_func(
            q, kv, dropout_p=0.0, causal=False, softmax_scale=softmax_scale,
            window_size=(window_left, window_right),
        )

    @torch.library.register_fake("aido::flash_attn_kv")
    def _meta(q, kv, softmax_scale, window_left, window_right):
        b, s, h, d = q.shape
        return q.new_empty(b, s, h, d)


_register_flash_attn_kv_op()


class _TrackOutput:
    __slots__ = ("values",)
    def __init__(self, values):
        self.values = values


class _Prediction:
    def __init__(self, parts):
        self.cage = _TrackOutput(parts[0])
        self.rna_seq = _TrackOutput(parts[1])
        self.atac = _TrackOutput(parts[2])
        self.dnase = _TrackOutput(parts[3])


class AidoDna3AgBinaryA100:
    def __init__(
        self, package_path, device="cuda", splits=SPLITS,
        max_length=None, length_multiple=None, min_length=None, max_batch=None,
    ):
        self._runner = torch._inductor.aoti_load_package(package_path)
        self.device = device
        self.splits = tuple(splits)
        # max_length=None, length_multiple=None: strictly static shape
        #   (whatever the package was traced with, e.g. batch=1, L=8192).
        # max_length=<int>, length_multiple=None: fixed-L package, batch
        #   dynamic up to some max -- right-pad shorter inputs up to
        #   max_length exactly (always costs the full max_length compute).
        # max_length=<int>, length_multiple=<int>: truly dynamic-length
        #   package -- right-pad only up to the next multiple of
        #   length_multiple (much less wasted compute on short inputs).
        self.max_length = max_length
        self.length_multiple = length_multiple
        self.min_length = min_length
        self.max_batch = max_batch

    def predict_tracks(self, input_ids, is_human=True):
        assert is_human, "This binary is compiled for the human head (is_human=True) only."
        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids)
        input_ids = input_ids.to(self.device)
        if input_ids.dtype != torch.int64:
            input_ids = input_ids.long()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        b, l = input_ids.shape
        if self.max_batch is not None and b > self.max_batch:
            raise ValueError(f"batch size {b} exceeds this package's max supported batch {self.max_batch}.")

        pad = 0
        if self.max_length is not None:
            if l > self.max_length:
                raise ValueError(
                    f"input length {l} exceeds this package's max supported length "
                    f"{self.max_length}. Use a build with a larger L, or crop/window "
                    f"the input upstream -- see README.md's 'Known limitations' section."
                )
            if self.length_multiple is not None:
                target = max(self.min_length or self.length_multiple, l)
                target = -(-target // self.length_multiple) * self.length_multiple  # ceil to multiple
            else:
                target = self.max_length
            pad = target - l
            if pad > 0:
                input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=0)

        out = self._runner(input_ids)  # (B, L', sum(splits))
        if pad > 0:
            out = out[:, :l, :]
        parts = torch.split(out, self.splits, dim=-1)
        return _Prediction(parts)

    # HF-model-like no-ops so `initialize_model`-style code keeps working.
    def eval(self):
        return self

    def cuda(self, *a, **k):
        return self

    def to(self, *a, **k):
        return self

    def __call__(self, input_ids, is_human=True, **kw):
        return self.predict_tracks(input_ids, is_human=is_human)


def load(package_path, device="cuda", splits=SPLITS, max_length=None, length_multiple=None,
         min_length=None, max_batch=None):
    """Load a compiled A100 binary and return a drop-in model object.

    For `aido_dna3ag_1bp4_dyn_a100.pt2` (the truly dynamic build), all shape
    parameters are auto-detected from the filename -- just call
    `load(path)`. For the fixed-shape packages, pass `max_length` (the
    package's traced L, e.g. 8192 or 128000) to enable automatic
    right-padding for shorter inputs. Leave everything at the defaults for
    the plain static package, where the input must already match the traced
    (batch, L) exactly.
    """
    if length_multiple is None and max_length is None and "_dyn_a100" in package_path:
        max_length = DYN_MAX_LENGTH
        length_multiple = DYN_LENGTH_MULTIPLE
        min_length = DYN_MIN_LENGTH if min_length is None else min_length
        max_batch = DYN_MAX_BATCH if max_batch is None else max_batch
    return AidoDna3AgBinaryA100(
        package_path, device=device, splits=splits, max_length=max_length,
        length_multiple=length_multiple, min_length=min_length, max_batch=max_batch,
    )
