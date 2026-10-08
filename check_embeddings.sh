#!/usr/bin/env bash
# FM-GWAS embedding progress + integrity check. Run on the GPU cluster.
#
#   bash check_embeddings.sh              # world_size 4 assumed
#   bash check_embeddings.sh --world_size 4 --n_samples 43706
#
# Reports: how many genes are done overall and per rank, which .pt files are
# unreadable / truncated / degenerate, and which are safe to keep on resume.
set -uo pipefail
export PYTHONUNBUFFERED=1

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
GPU_PY="${CONDA_ROOT}/envs/deep_learning/bin/python3"

EMB_SCRIPT="${FMGWAS_ROOT}/get_hpp_embedding-AG_clean.py"
GENE_SEQ_DIR="${OUT_ROOT}/gene_sequences"
FEATURE_ROOT="${OUT_ROOT}/embeddings"
MARKER_DIR="${OUT_ROOT}/.stage_markers"

WORLD_SIZE=4
N_SAMPLES=43706          # rows expected per gene (dataset size from the logs)
DEEP=0                   # --deep loads every .pt instead of a sample
SAMPLE_N=200             # files to fully load when not --deep

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --world_size) WORLD_SIZE="${2:?}"; shift 2 ;;
        --n_samples)  N_SAMPLES="${2:?}";  shift 2 ;;
        --sample_n)   SAMPLE_N="${2:?}";   shift 2 ;;
        --deep)       DEEP=1; shift ;;
        -h|--help)    sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -x "$GPU_PY" ]] || { echo "no python at $GPU_PY" >&2; exit 1; }
[[ -d "$GENE_SEQ_DIR" ]] || { echo "missing $GENE_SEQ_DIR" >&2; exit 1; }

echo "=============================================================="
echo " sharding logic in $(basename "$EMB_SCRIPT")"
echo "=============================================================="
grep -n "world_size\|rank" "$EMB_SCRIPT" 2>/dev/null | head -20 \
    || echo "(script not readable from here)"

echo
echo "=============================================================="
echo " rank markers"
echo "=============================================================="
ls -l "$MARKER_DIR" 2>/dev/null | grep embeddings || echo "(no rank markers -- no rank finished cleanly)"

echo
echo "=============================================================="
echo " files written in the last 30 min (possibly mid-write)"
echo "=============================================================="
find "$FEATURE_ROOT" -name '*.pt' -newermt '-30 minutes' -printf '%T+ %10s %p\n' 2>/dev/null | sort | tail -20 \
    || echo "(none)"

WORLD_SIZE="$WORLD_SIZE" N_SAMPLES="$N_SAMPLES" DEEP="$DEEP" SAMPLE_N="$SAMPLE_N" \
GENE_SEQ_DIR="$GENE_SEQ_DIR" FEATURE_ROOT="$FEATURE_ROOT" \
"$GPU_PY" - <<'PY'
import os, glob, random, torch

W          = int(os.environ["WORLD_SIZE"])
N_SAMPLES  = int(os.environ["N_SAMPLES"])
DEEP       = os.environ["DEEP"] == "1"
SAMPLE_N   = int(os.environ["SAMPLE_N"])
SEQ_DIR    = os.environ["GENE_SEQ_DIR"]
EMB_DIR    = os.environ["FEATURE_ROOT"]

genes = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{SEQ_DIR}/*.tsv"))
pt    = {os.path.basename(p)[:-3]: p for p in glob.glob(f"{EMB_DIR}/*/*.pt")}
done  = set(pt)

print()
print("=" * 62)
print(" progress")
print("=" * 62)
print(f"genes with sequences : {len(genes)}")
print(f".pt files present    : {len(done)}  ({100*len(done)/max(len(genes),1):.1f}%)")
orphan = done - set(genes)
if orphan:
    print(f"WARNING: {len(orphan)} .pt with no matching .tsv, e.g. {sorted(orphan)[:3]}")

# The FM-GWAS script's split is not visible from the wrapper, so report both
# conventions. The grep above tells you which one is real.
schemes = {
    "strided   genes[r::W]": lambda r: genes[r::W],
    "contiguous chunk r/W ": lambda r: genes[r*len(genes)//W:(r+1)*len(genes)//W],
}
for name, f in schemes.items():
    print(f"\n-- assuming {name}")
    for r in range(W):
        mine = f(r)
        d    = [g for g in mine if g in done]
        todo = [g for g in mine if g not in done]
        nxt  = todo[0] if todo else "-"
        print(f"   rank {r}: {len(d):5d}/{len(mine):5d} "
              f"({100*len(d)/max(len(mine),1):5.1f}%)  next={nxt}")

# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------
files = sorted(pt.values())
if not DEEP and len(files) > SAMPLE_N:
    random.seed(0)
    check = sorted(random.sample(files, SAMPLE_N))
    mode  = f"random sample of {SAMPLE_N}"
else:
    check = files
    mode  = "all files"

print()
print("=" * 62)
print(f" integrity ({mode}; --deep loads everything)")
print("=" * 62)

def tensor_of(o):
    if torch.is_tensor(o):
        return o
    if isinstance(o, dict):
        for k in ("embeddings", "embedding", "features", "feature", "x"):
            if k in o and torch.is_tensor(o[k]):
                return o[k]
        for v in o.values():
            if torch.is_tensor(v):
                return v
    return None

bad, shapes, keys_seen = [], {}, set()
for f in check:
    try:
        o = torch.load(f, map_location="cpu")
    except Exception as e:
        bad.append((f, f"load failed: {type(e).__name__}: {e}"))
        continue
    if isinstance(o, dict):
        keys_seen.add(tuple(sorted(o.keys()))[:8])
    x = tensor_of(o)
    if x is None:
        bad.append((f, f"no tensor (type={type(o).__name__})"))
        continue
    shapes.setdefault(tuple(x.shape), []).append(f)
    if x.shape[0] != N_SAMPLES:
        bad.append((f, f"n_rows={x.shape[0]} != {N_SAMPLES} -- TRUNCATED"))
        continue
    xf = x.float()
    if not torch.isfinite(xf).all():
        bad.append((f, "nan/inf"))
    elif float(xf.std()) == 0.0:
        bad.append((f, "constant tensor"))

if keys_seen:
    print("dict key sets:", list(keys_seen)[:3])
print("distinct shapes:")
for s, fs in sorted(shapes.items(), key=lambda kv: -len(kv[1]))[:8]:
    print(f"   {s}  x{len(fs)}   e.g. {os.path.basename(fs[0])}")
print(f"\nBAD: {len(bad)} / {len(check)}")
for f, why in bad[:25]:
    print(f"   {os.path.basename(f)}: {why}")
if len(bad) > 25:
    print(f"   ... and {len(bad)-25} more")

if check:
    f = check[len(check)//2]
    x = tensor_of(torch.load(f, map_location="cpu"))
    if x is not None:
        xf = x.float()
        print(f"\nspot check {os.path.basename(f)}: shape={tuple(x.shape)} "
              f"dtype={x.dtype} mean={xf.mean():.4f} std={xf.std():.4f} "
              f"min={xf.min():.3f} max={xf.max():.3f}")

if bad:
    out = "/tmp/bad_embeddings.txt"
    with open(out, "w") as fh:
        for f, why in bad:
            fh.write(f"{f}\t{why}\n")
    print(f"\nbad file list -> {out}")
    print("delete them before resuming:  cut -f1 /tmp/bad_embeddings.txt | xargs -r rm -v")
PY
