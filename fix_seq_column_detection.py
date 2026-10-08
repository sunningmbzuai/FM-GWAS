#!/usr/bin/env python
r"""Fix the sequence-column autodetection in FM-GWAS's embedding script.

get_hpp_embedding-AG_clean.py's PairedSequenceDataset picks the haplotype
column prefix like this:

    base_key = (
        "mutation_seq" if "mutation_seq" in self.df.columns
        else "mutated_seq" if "mutated_seq" in self.df.columns
        else "variant_seq"
    )
    self.mut1_key = f"{base_key}_1"

but the columns are always the *suffixed* pair (mutation_seq_1 /
mutation_seq_2) -- the bare base name is never itself a column. So both tests
fail and it falls through to "variant_seq" regardless, then dies with

    KeyError: 'variant_seq_1'

on gene_sequences TSVs whose header is
participant_id / ref_seq / mutation_seq_1 / mutation_seq_2 (see COLUMNS in
gene_seq_writer.py). Testing the _1 column fixes the detection.

Idempotent. Backs the original up to <name>.orig on first run.

Usage:
    python fix_seq_column_detection.py <path to get_hpp_embedding-AG_clean.py>
"""
import argparse
import pathlib

REPLACEMENTS = [
    (
        '"mutation_seq" if "mutation_seq" in self.df.columns',
        '"mutation_seq" if "mutation_seq_1" in self.df.columns',
    ),
    (
        '"mutated_seq" if "mutated_seq" in self.df.columns',
        '"mutated_seq" if "mutated_seq_1" in self.df.columns',
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
