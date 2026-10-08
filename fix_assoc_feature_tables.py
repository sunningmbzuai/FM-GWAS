#!/usr/bin/env python
r"""Teach FM-GWAS's assoc script to read the exported feature tables.

insample_hpp_AG.py torch.load()s <gene_id>.pt and then spends its first
minutes turning tensors into the 16 numpy views it tests. That work is a
deterministic reshape, so export_features.py now does it once on the GPU node
and writes <gene_id>.npz. This patch makes assoc prefer that file:

  * the per-gene glob looks for .npz first and falls back to .pt, so a gene
    that has not been exported yet still runs the old way;
  * the loader gains an .npz branch that is numpy only;
  * `import torch` becomes optional, which is the point -- with the tables in
    place the association tests run on the prep cluster, where the CPUs are,
    instead of being pinned to the GPU node by the only torch that loads there.

Idempotent. Backs the original up to <name>.orig on first run.

Usage:
    python fix_assoc_feature_tables.py \
        <FM-GWAS checkout>/insample_hpp_AG.py
"""
import argparse
import pathlib

REPLACEMENTS = [
    # 1. Prefer the exported table. Same directory layout, different suffix.
    (
        "win_paths = glob.glob(join(feature_root, gene_id, f'{gene_id}.pt'))",
        "win_paths = (glob.glob(join(feature_root, gene_id, f'{gene_id}.npz'))\n"
        "                     or glob.glob(join(feature_root, gene_id, f'{gene_id}.pt')))",
    ),
    # 2. Read it. The views are stored under the names this script gives them,
    #    so the dict goes straight into pt_data; subject ids keep the
    #    "<participant>_<...>" spelling the .pt used, parsed the same way.
    (
        '        if path.endswith(".pt"):\n'
        '            data = torch.load(path, map_location="cpu")\n',
        '        if path.endswith(".npz"):\n'
        '            # Built by export_features.py on the GPU node: numpy only.\n'
        '            with np.load(path, allow_pickle=False) as handle:\n'
        '                pt_ids = np.array([int(str(s).split("_")[0])\n'
        '                                   for s in handle["subject_id"]])\n'
        '                pt_data = {k: handle[k] for k in handle.files\n'
        '                           if k != "subject_id"}\n'
        '            continue\n'
        '        if path.endswith(".pt"):\n'
        '            data = torch.load(path, map_location="cpu")\n',
    ),
    # 3. torch is now only needed for the .pt fallback.
    (
        "import torch\n",
        "try:\n"
        "    import torch\n"
        "except ImportError:  # only the .pt fallback needs it; .npz is numpy\n"
        "    torch = None\n",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    target = pathlib.Path(args.script)
    if not target.is_file():
        raise SystemExit("not a file: %s" % target)

    source = target.read_text()
    patched = source
    applied = 0
    for old, new in REPLACEMENTS:
        if new in patched:
            continue
        if old not in patched:
            raise SystemExit(
                "expected code not found in %s -- inspect by hand:\n  %s"
                % (target.name, old)
            )
        patched = patched.replace(old, new, 1)
        applied += 1

    if applied == 0:
        print("already patched: %s" % target.name)
        return
    if args.dry_run:
        print("would apply %d replacement(s) to %s" % (applied, target.name))
        return

    backup = target.with_suffix(target.suffix + ".orig")
    if not backup.exists():
        backup.write_text(source)
        print("backup -> %s" % backup.name)
    target.write_text(patched)
    print("patched %s (%d replacement(s))" % (target.name, applied))


if __name__ == "__main__":
    main()
