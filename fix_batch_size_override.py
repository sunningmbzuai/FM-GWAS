#!/usr/bin/env python
r"""Let the embedding script's batch size be pinned from the environment.

get_hpp_embedding-AG_clean.py sizes each batch with

    def dynamic_batch_size(length, gpu="A100"):
        multiple = 2 if gpu == "A100" else 1
        ...

The x2 for "A100" assumes the 80 GB part. On a 40 GB A100 (44.39 GiB total)
a real gene at L=30976 lands in the `<= 51712` branch and asks for 4*2 = 8,
whose activations want a single 60.5 GiB allocation -- OOM before the first
batch completes. The compiled binary's own validation table used batch 2-8
across this length range on the 80 GB card.

Rather than retune the whole table blind, this adds one env-var escape hatch:

    FMGWAS_BATCH_SIZE=4 python get_hpp_embedding-AG_clean.py ...

Unset, behaviour is exactly as before. dynamic_batch_size is the only place a
batch size is chosen, so this covers every call site.

Idempotent. Backs the original up to <name>.orig on first run.

Usage:
    python fix_batch_size_override.py <path to get_hpp_embedding-AG_clean.py>
"""
import argparse
import pathlib

ANCHOR = '''def dynamic_batch_size(length, gpu="A100"):
    multiple = 2 if gpu == "A100" else 1'''

PATCHED = '''def dynamic_batch_size(length, gpu="A100"):
    # FMGWAS_BATCH_SIZE pins the batch size regardless of length or GPU. The
    # x2 below assumes an 80 GB A100; on the 40 GB part the derived sizes OOM
    # at real gene lengths, and there is no CLI flag to say so.
    override = os.environ.get("FMGWAS_BATCH_SIZE")
    if override:
        return int(override)
    multiple = 2 if gpu == "A100" else 1'''

MARKER = 'os.environ.get("FMGWAS_BATCH_SIZE")'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = pathlib.Path(args.script)
    if not target.is_file():
        raise SystemExit("not a file: %s" % target)

    source = target.read_text()
    if MARKER in source:
        print("already patched: %s" % target.name)
        return
    if ANCHOR not in source:
        raise SystemExit(
            "dynamic_batch_size does not look as expected in %s -- patch by hand"
            % target.name
        )
    if "import os" not in source:
        raise SystemExit(
            "%s does not import os, which the override needs" % target.name
        )

    if args.dry_run:
        print("would patch %s" % target.name)
        return

    backup = target.with_suffix(target.suffix + ".orig")
    if not backup.exists():
        backup.write_text(source)
        print("backup -> %s" % backup.name)
    target.write_text(source.replace(ANCHOR, PATCHED, 1))
    print("patched %s" % target.name)
    print("set FMGWAS_BATCH_SIZE to pin the batch size; unset restores the old behaviour")


if __name__ == "__main__":
    main()
