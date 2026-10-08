"""Preflight: confirm GRCH37_FASTA / GRCH37_GTF / UKB_BFILE all agree on genome
build and contig naming before spending compute on phasing or gene extraction.
A build mismatch (e.g. a GRCh38 fasta paired with GRCh37 array positions)
would silently corrupt every downstream base call, so this gates the pipeline
rather than warning.

Checks:
  1. fasta contig lengths match known GRCh37 (not GRCh38) chromosome lengths
  2. GTF chromosome naming resolves onto the fasta's contigs (chr-prefix or not)
  3. direct build/strand check: sample N random UKB array SNPs from the .bim
     and confirm the fasta base at that position is one of the two bim
     alleles -- a build or strand mismatch shows up here directly, not just
     via contig naming

Run with wasp_env (needs pysam). Exits nonzero on any hard mismatch.
"""
import os
import sys

import pandas as pd
import pysam

import config

N_SAMPLE = 5000
MIN_REF_MATCH_RATE = 0.97  # correct build/strand: ~100% except rare A/T,C/G strand-ambiguous SNPs

# GRCh37/hg19 primary-assembly chromosome lengths (human_g1k_v37 / b37 convention,
# no 'chr' prefix). Used to catch an accidental GRCh38 fasta.
GRCH37_LENGTHS = {
    "1": 249250621, "2": 243199373, "3": 198022430, "4": 191154276,
    "5": 180915260, "6": 171115067, "7": 159138663, "8": 146364022,
    "9": 141213431, "10": 135534747, "11": 135006516, "12": 133851895,
    "13": 115169878, "14": 107349540, "15": 102531392, "16": 90354753,
    "17": 81195210, "18": 78077248, "19": 59128983, "20": 63025520,
    "21": 48129895, "22": 51304566,
}


def log(msg):
    print(f"[validate_ukb_build] {msg}", file=sys.stderr, flush=True)


def fail(msg):
    log(f"FAIL: {msg}")
    sys.exit(1)


def strip_chr(c):
    return c[3:] if c.startswith("chr") else c


def main():
    bim_path = f"{config.UKB_BFILE}.bim"
    for path in (config.GRCH37_FASTA, config.GRCH37_GTF, bim_path):
        os.path.exists(path) or fail(f"not found: {path}")

    fasta = pysam.FastaFile(config.GRCH37_FASTA)
    fa_contigs = {strip_chr(c): c for c in fasta.references}

    # ---- 1: fasta build check via known GRCh37 chromosome lengths ----------
    mismatched_len = []
    for chrom, expected_len in GRCH37_LENGTHS.items():
        raw = fa_contigs.get(chrom)
        if raw is None:
            mismatched_len.append((chrom, "MISSING", expected_len))
            continue
        actual_len = fasta.get_reference_length(raw)
        if actual_len != expected_len:
            mismatched_len.append((chrom, actual_len, expected_len))
    if mismatched_len:
        for chrom, actual, expected in mismatched_len[:5]:
            log(f"  chrom {chrom}: fasta length {actual} != GRCh37 {expected}")
        fail(f"{config.GRCH37_FASTA} does not match GRCh37 chromosome lengths "
             f"({len(mismatched_len)}/22 mismatched) -- likely GRCh38 or a different assembly")
    log(f"fasta lengths match GRCh37 for all 22 autosomes: {config.GRCH37_FASTA}")

    # ---- 2: GTF chromosome set resolves onto the fasta ----------------------
    gtf_chroms = set()
    with open(config.GRCH37_GTF) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            gtf_chroms.add(strip_chr(line.split("\t", 1)[0]))
            if len(gtf_chroms) >= 25:  # autosomes + X/Y/M is plenty to confirm naming
                break
    missing = sorted(c for c in gtf_chroms if c in GRCH37_LENGTHS and c not in fa_contigs)
    if missing:
        fail(f"GTF contigs not found in fasta: {missing}")
    log(f"GTF contig naming resolves onto fasta contigs: {config.GRCH37_GTF}")

    # ---- 3: direct build/strand check against the UKB array bim ------------
    # pandas' C parser here instead of a python per-line loop -- the bim can be
    # large enough (esp. over NFS) that a manual split()-per-line loop is the
    # slow part of this whole check.
    log(f"reading {bim_path} ...")
    bim = pd.read_csv(
        bim_path, sep=r"\s+", header=None, usecols=[0, 3, 4, 5],
        names=["chrom", "pos", "a1", "a2"], dtype=str,
    )
    bim["chrom"] = bim["chrom"].str.replace(r"^chr", "", regex=True)
    keep = (
        bim["chrom"].isin(GRCH37_LENGTHS)
        & bim["a1"].str.len().eq(1) & bim["a2"].str.len().eq(1)
        & bim["a1"].isin(list("ACGT")) & bim["a2"].isin(list("ACGT"))
    )
    bim = bim.loc[keep]
    if bim.empty:
        fail(f"no usable biallelic SNP rows read from {bim_path}")
    log(f"{len(bim):,} usable biallelic SNP rows; sampling {min(N_SAMPLE, len(bim)):,}")

    sample = bim.sample(n=min(N_SAMPLE, len(bim)), random_state=0)
    # sorted by (chrom, pos) so fasta.fetch hits go roughly sequentially --
    # unsorted, this is 5000 random-access seeks against an NFS-hosted fasta.
    sample["pos_int"] = sample["pos"].astype(int)
    sample = sample.sort_values(["chrom", "pos_int"])
    log("checking sample against fasta ...")
    mismatches = []
    for i, (chrom, _pos, a1, a2, pos) in enumerate(sample.itertuples(index=False), 1):
        base = fasta.fetch(fa_contigs[chrom], pos - 1, pos).upper()
        if base not in (a1, a2):
            mismatches.append((chrom, pos, a1, a2, base))
        if i % 1000 == 0:
            log(f"  {i}/{len(sample)}")

    rate = 1 - len(mismatches) / len(sample)
    log(f"UKB bim vs fasta base match: {len(sample) - len(mismatches)}/{len(sample)} = {rate:.4f}")
    if rate < MIN_REF_MATCH_RATE:
        for chrom, pos, a1, a2, base in mismatches[:10]:
            log(f"  mismatch {chrom}:{pos} bim=({a1}/{a2}) fasta={base}")
        fail(f"match rate {rate:.4f} < {MIN_REF_MATCH_RATE} -- {config.UKB_BFILE} and "
             f"{config.GRCH37_FASTA} likely disagree on build or strand")

    log("All checks passed: fasta / GTF / UKB array genotypes agree on GRCh37.")


if __name__ == "__main__":
    main()
