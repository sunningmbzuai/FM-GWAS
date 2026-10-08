"""Build FM-GWAS gene raw-sequence TSVs: <gene_id>.tsv with columns
participant_id, ref_seq, mutation_seq_1, mutation_seq_2 (space-separated bases).

Window building is strand-aware and GTF-driven. Variant source is the Eagle-phased VCF
produced by run_eagle_phasing_ukb.sh (config.PHASED_VCF) -- population-based
phasing of UKB's QC-passed array samples, the same method HPP used. hap1/hap2 therefore reflect real
statistically-phased haplotypes, not the fixed ref/alt convention this script
used before phasing existed. The usual limits of statistical phasing still
apply (phase-switch errors over distance, statistical not experimental phase).

Run with "wasp_env" + pysam on PATH. Requires run_eagle_phasing_ukb.sh to
have produced config.PHASED_VCF already.
"""
import os
import re
import time

import numpy as np
import pandas as pd
import pysam

import config
import gene_seq_writer as W
import target_genes
from ukb_sequence import orient_to_tss

# Complementing an alt allele when a minus-strand window is reverse-complemented.
COMPLEMENT_BASE = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}

# How often to print a progress line. The per-gene work is minutes at cohort
# scale, so a silent run gives no way to tell a slow stage from a hung one.
# Scaled to the gene set: a fixed 10 means a key-association run (18 genes)
# prints nothing for its first ~half, which reads exactly like a hang.
PROGRESS_EVERY_MAX = 10
PROGRESS_TARGET_LINES = 20


def progress_every(n_genes: int) -> int:
    return max(1, min(PROGRESS_EVERY_MAX, n_genes // PROGRESS_TARGET_LINES))

CHROMS = [str(c) for c in range(1, 23)]  # autosomes only


def load_blindspots(bed_path):
    """{chrom: set(0-based positions)} of multi-allelic positions Eagle-phasing
    dropped -- present in the reference/consensus but genuinely unknown."""
    spots = {}
    if not bed_path or not os.path.exists(bed_path):
        return spots
    with open(bed_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chrom, start, end = line.split("\t")[:3]
            spots.setdefault(chrom, set()).update(range(int(start), int(end)))
    return spots


def build_gene_model() -> dict:
    """gene_id (versioned) -> (chrom, start, end, strand), 1-based inclusive.

    Restricted to config.GENE_TYPE ("protein_coding") per the paper's Methods.
    The GTF "gene" feature spans TSS to TES on both strands, so start/end are
    already the paper's window; strand matters only if a flank is configured.
    """
    gm = {}
    n_wrong_type = 0
    with open(config.GRCH37_GTF) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            chrom = fields[0].replace("chr", "")
            if chrom not in CHROMS:
                continue
            start, end, strand, attrs = int(fields[3]), int(fields[4]), fields[6], fields[8]
            m = re.search(r'gene_id "([^"]+)"', attrs)
            if not m:
                continue
            gene_type = re.search(r'gene_type "([^"]+)"', attrs)
            if gene_type is None or gene_type.group(1) != config.GENE_TYPE:
                n_wrong_type += 1
                continue
            gm[m.group(1)] = (chrom, start, end, strand)
    print(f"Gene model: {len(gm):,} {config.GENE_TYPE} genes on chr1-22 "
          f"({n_wrong_type:,} genes of other types excluded)")
    return gm


def gene_window(chrom: str, start: int, end: int, strand: str) -> tuple[int, int]:
    """TSS -> TES, plus any configured flank (both flanks are 0 per the paper)."""
    if strand == "+":
        return max(1, start - config.GENE_UPSTREAM_BP), end + config.GENE_DOWNSTREAM_BP
    return max(1, start - config.GENE_DOWNSTREAM_BP), end + config.GENE_UPSTREAM_BP


def union_gene_ids() -> list[str]:
    seen = []
    for fname in os.listdir(config.HPP_UNION_GENE_DIR):
        if not fname.endswith(".tsv"):
            continue
        col = pd.read_csv(os.path.join(config.HPP_UNION_GENE_DIR, fname), sep="\t")["gene_id"]
        seen.extend(col.tolist())
    genes = sorted(set(seen))
    print(f"Union gene list across all trait files: {len(genes):,} genes")
    if config.RESTRICT_TO_TARGET_GENES:
        genes = restrict_to_targets(genes)
    return genes


def restrict_to_targets(gene_ids: list[str]) -> list[str]:
    """Keep only the key associations named in config.TARGET_GENES_TSV.

    Matching is unversioned because the target list and the HPP lists carry
    independent gene-id versions. A target absent from the HPP union is added
    under the target list's id rather than dropped: this is a UKB run, and the
    union only gates which genes to process.
    """
    kept, added = target_genes.extend_with_targets(gene_ids)
    print(f"Restricted to {len(kept)} target genes from "
          f"{config.TARGET_GENES_TSV}")
    if added:
        print(f"  {len(added)} of them are not in the HPP union and were added "
              f"from the target list")
    return kept


def collect_variants(vf: pysam.VariantFile, blindspots: dict, chrom: str,
                     start: int, end: int, ref_len: int):
    """One gene's variants as (positions, alts, genotype matrix).

    The matrix is int8 of shape (n_variants, n_participants), so a gene costs
    ~13 MB for 300 variants instead of one full sequence per participant. Only
    biallelic SNPs are kept -- PLINK bim rows are inherently biallelic SNPs, so
    anything else is a data problem, not a case to handle.
    """
    samples = list(vf.header.samples)
    spots = blindspots.get(chrom, set())
    positions, alts, columns = [], [], []

    for rec in vf.fetch(chrom, start - 1, end):
        if rec.ref is None or len(rec.ref) != 1 or len(rec.alts or ()) != 1 \
                or len(rec.alts[0]) != 1:
            continue
        idx = rec.pos - start
        if not (0 <= idx < ref_len):
            continue

        if (rec.pos - 1) in spots:
            codes = np.full(len(samples), W.CODE_MISSING, dtype=np.int8)
        else:
            codes = np.fromiter(
                (W.code_genotype(*rec.samples[s].get("GT", (None, None))[:2])
                 for s in samples),
                dtype=np.int8, count=len(samples),
            )

        positions.append(idx)
        alts.append(rec.alts[0].upper())
        columns.append(codes)

    if not positions:
        return samples, np.empty(0, dtype=np.int64), [], np.empty((0, len(samples)),
                                                                  dtype=np.int8)
    return (samples, np.asarray(positions, dtype=np.int64), alts,
            np.vstack(columns))


def main() -> None:
    if not os.path.exists(config.PHASED_VCF):
        raise SystemExit(
            f"{config.PHASED_VCF} not found -- run ./run_eagle_phasing_ukb.sh first."
        )
    os.makedirs(config.GENE_SEQ_DIR, exist_ok=True)

    gene_model = build_gene_model()
    gene_ids = union_gene_ids()
    fa = pysam.FastaFile(config.GRCH37_FASTA)
    vf = pysam.VariantFile(config.PHASED_VCF)
    blindspots = load_blindspots(config.PHASING_BLINDSPOTS_BED)

    skipped, too_long, no_variants = [], [], []
    written, started = 0, time.time()
    every = progress_every(len(gene_ids))
    print(f"Building {len(gene_ids):,} genes, progress every {every} gene(s)",
          flush=True)
    for i, gene_id in enumerate(gene_ids, start=1):
        if i % every == 0 or i == len(gene_ids):
            elapsed = time.time() - started
            rate = written / elapsed if elapsed > 0 and written else 0.0
            remaining = (len(gene_ids) - i) / rate if rate > 0 else float("nan")
            print(f"  [{i}/{len(gene_ids)}] {written} written, "
                  f"{len(skipped) + len(too_long) + len(no_variants)} skipped, "
                  f"{rate * 60:.1f} genes/min, ~{remaining / 3600:.1f} h left",
                  flush=True)
        out_path = os.path.join(config.GENE_SEQ_DIR, f"{gene_id}.tsv")
        if os.path.exists(out_path):
            continue

        bare_id = gene_id.split(".")[0]
        coords = gene_model.get(gene_id) or next(
            (v for k, v in gene_model.items() if k.split(".")[0] == bare_id), None
        )
        if coords is None:
            skipped.append(gene_id)
            continue
        chrom, gstart, gend, strand = coords
        wstart, wend = gene_window(chrom, gstart, gend, strand)
        # Excluded, not truncated: a truncated gene is a different sequence, and
        # the paper's gene set comes from excluding genes over 128 kb outright.
        if wend - wstart + 1 > config.MAX_GENE_LEN:
            too_long.append(gene_id)
            continue

        ref_seq = fa.fetch(chrom, wstart - 1, wend).upper()
        samples, positions, alts, codes = collect_variants(
            vf, blindspots, chrom, wstart, wend, len(ref_seq)
        )
        if len(positions) == 0:
            no_variants.append(gene_id)
            continue

        # Orient 5'->3' along the gene AFTER collecting variants: reversing the
        # sequence moves every variant, so positions have to be remapped too.
        if strand == "-":
            ref_seq = orient_to_tss(ref_seq, strand)
            positions = (len(ref_seq) - 1) - positions
            alts = [COMPLEMENT_BASE[a] for a in alts]

        # Temp path + rename: a killed run must not leave a half-written TSV
        # that the resume logic would treat as a finished gene.
        tmp_path = f"{out_path}.partial"
        W.write_gene_tsv(tmp_path, samples, ref_seq, positions, alts, codes)
        os.replace(tmp_path, out_path)
        written += 1

    dropped = len(skipped) + len(too_long) + len(no_variants)
    print(f"Wrote {len(gene_ids) - dropped:,} gene TSVs to {config.GENE_SEQ_DIR}")
    if skipped:
        print(f"Skipped {len(skipped)} genes not protein-coding on chr1-22 in "
              f"the GRCh37 GTF: {skipped[:20]}...")
    if too_long:
        print(f"Skipped {len(too_long)} genes longer than "
              f"{config.MAX_GENE_LEN // 1024} kb: {too_long[:20]}...")
    if no_variants:
        print(f"Skipped {len(no_variants)} genes with no observed variants: "
              f"{no_variants[:20]}...")


if __name__ == "__main__":
    main()
