"""Drop-in wrapper for the architecture-independent AIDO.DNA3-AG package.

Same call surface as ../compiled_A100_v1/aido_dna3ag_binary.py and
../compiled_A100/aido_dna3ag_binary.py:

    import aido_dna3ag_portable as A
    model = A.load()                                  # or load("/path/to/x.pt2")
    out = model.predict_tracks(input_ids, is_human=True)   # input_ids: (B, L) int
    out.cage.values      # (B, L, 546)
    out.rna_seq.values   # (B, L, 667)
    out.atac.values      # (B, L, 167)
    out.dnase.values     # (B, L, 305)

Differences from the v1 binary, all of them removals:
  - **no flash-attn needed.** Attention is pure ATen in the graph.
  - **no custom operator to register.** Nothing has to be imported before load.
  - **no C++ compiler, no link step, no .so cache.** Loading is deserialization.
  - **no GPU-architecture constraint.** No compiled kernels are shipped; the
    local torch dispatches the graph, so any GPU torch supports will run it.

Shape support: batch 1-32, length any multiple of 128 in [256, 128000]. Inputs
are right-padded up to the next multiple of 128 and the output cropped back, so
a short input costs only its own compute.

Speed
-----
This trades throughput for portability: the graph runs op-by-op instead of as
fused AOTInductor kernels. To get most of that back, compile it locally for
whatever GPU you actually have -- one line, done once per process:

    model = A.load(compile=True)      # wraps the graph in torch.compile

The first call then pays a JIT cost (tens of seconds) and later calls run
fused, native kernels for your architecture. Note torch.compile will re-trace
when the input length changes; with `dynamic=True` (the default here) it
should specialize once over the dynamic range rather than per length.
"""
import json
import os
import zipfile

import torch

TRACK_ORDER = ("CAGE", "RNA_SEQ", "ATAC", "DNASE")
SPLITS = (546, 667, 167, 305)

LENGTH_MULTIPLE = 128
MIN_LENGTH = 256
MAX_LENGTH = 128000
MAX_BATCH = 32

DEFAULT_PACKAGE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aido_dna3ag_1bp4_dyn_portable.pt2"
)


def package_info(path=DEFAULT_PACKAGE):
    """Return the build metadata recorded inside the package (or {})."""
    try:
        with zipfile.ZipFile(path) as zf:
            if "aido_build_meta.json" in zf.namelist():
                return json.loads(zf.read("aido_build_meta.json"))
    except Exception:
        pass
    return {}


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


class AidoDna3AgPortable:
    def __init__(self, package_path=DEFAULT_PACKAGE, device="cuda", splits=SPLITS,
                 compile=False, verbose=False):
        if not os.path.isfile(package_path):
            raise FileNotFoundError(package_path)
        self.meta = package_info(package_path)

        bt = self.meta.get("torch_version")
        if bt and bt.split("+")[0] != torch.__version__.split("+")[0] and verbose:
            print(f"[aido] note: package was exported with torch {bt}, running "
                  f"{torch.__version__}. torch.export artifacts are usually "
                  f"forward-compatible; if loading fails, match the version.")

        if verbose:
            print(f"[aido] loading {os.path.basename(package_path)} ...")
        ep = torch.export.load(package_path)
        self._mod = ep.module()
        # Note: no .eval() / .to() here. The object returned by
        # ExportedProgram.module() is not a normal nn.Module in torch 2.5.1 --
        # .eval() raises NotImplementedError -- and it needs neither: the graph
        # was traced from an already-eval()ed model (no dropout/batchnorm nodes
        # exist in it to toggle), and the weights deserialize onto the device
        # they were exported from.
        self.device = self._infer_device(device)
        if verbose:
            print(f"[aido] weights are on {self.device}")

        if compile:
            if verbose:
                print("[aido] wrapping in torch.compile (first call will JIT) ...")
            self._mod = torch.compile(self._mod, dynamic=True)

        self.splits = tuple(splits)

    def _infer_device(self, requested):
        """Where to put inputs: wherever the deserialized weights actually live.

        Moving the graph's weights is avoided -- the module returned by
        torch.export is not reliably movable in torch 2.5.1 -- so inputs follow
        the weights instead. If a specific device was requested and the weights
        are elsewhere, try to move them and fall back with a clear warning.
        """
        found = None
        for getter in ("parameters", "buffers"):
            try:
                for t in getattr(self._mod, getter)():
                    found = t.device
                    break
            except Exception:
                pass
            if found is not None:
                break
        if found is None:
            found = torch.device(requested or "cuda")

        if requested is not None and torch.device(requested).type != found.type:
            try:
                self._mod = self._mod.to(requested)
                return torch.device(requested)
            except Exception as e:
                print(f"[aido] warning: could not move the graph to {requested} "
                      f"({type(e).__name__}); using {found} instead")
        return found

    def predict_tracks(self, input_ids, is_human=True):
        assert is_human, "This package is exported for the human head (is_human=True) only."
        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids)
        input_ids = input_ids.to(self.device)
        if input_ids.dtype != torch.int64:
            input_ids = input_ids.long()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        b, l = input_ids.shape
        if b > MAX_BATCH:
            raise ValueError(f"batch size {b} exceeds the exported maximum {MAX_BATCH}.")
        if l > MAX_LENGTH:
            raise ValueError(
                f"input length {l} exceeds the exported maximum {MAX_LENGTH}. "
                f"Crop or window the input upstream."
            )

        target = max(MIN_LENGTH, l)
        target = -(-target // LENGTH_MULTIPLE) * LENGTH_MULTIPLE
        pad = target - l
        if pad > 0:
            input_ids = torch.nn.functional.pad(input_ids, (0, pad), value=0)

        # no_grad is enforced here rather than left to the caller. This package
        # is inference-only, and with autograd live the graph retains every
        # intermediate for a backward pass that will never happen: at
        # B=2, L=128000 that is ~78 GB instead of ~4.8 GB, i.e. an OOM on any
        # GPU. Forgetting `with torch.no_grad():` is the single easiest way to
        # make this model look broken, so the wrapper does not allow it.
        # SDPA auto-picks a backend (flash / mem-efficient / math) per GPU SM
        # capability. Flash's output isn't contiguous the same way across
        # architectures, and this graph's baked-in .view() was traced
        # assuming one layout -- it silently breaks on a different GPU
        # (e.g. works on L40S/sm_89, RuntimeError on A100/sm_80). Force one
        # backend so behavior doesn't depend on which GPU this runs on.
        with torch.no_grad(), torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=True
        ):
            out = self._mod(input_ids)
        if pad > 0:
            out = out[:, :l, :]
        return _Prediction(torch.split(out, self.splits, dim=-1))

    # HF-model-like no-ops so existing model-loading code keeps working.
    def eval(self):
        return self

    def cuda(self, *a, **k):
        return self

    def to(self, *a, **k):
        return self

    def __call__(self, input_ids, is_human=True, **kw):
        return self.predict_tracks(input_ids, is_human=is_human)


def load(package_path=DEFAULT_PACKAGE, device="cuda", splits=SPLITS,
         compile=False, verbose=False, max_length=None):
    """Load the portable package and return a drop-in model object.

    Pass compile=True to wrap the graph in torch.compile, which JITs fused
    kernels for the local GPU -- recommended for throughput-sensitive runs.

    max_length is accepted for backward compatibility with older loaders but
    has no effect -- the portable model supports [256, 128000] by construction.
    """
    return AidoDna3AgPortable(package_path, device=device, splits=splits,
                              compile=compile, verbose=verbose)
