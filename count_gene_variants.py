"""Per-gene variant counts in the UKBB replication input, for the HPP comparison.

For every target gene this reports the window the pipeline actually used
(GRCh37, GENCODE v19, TSS -> TES, no flanks) and how many variants fall in it:

    n_variants     biallelic SNP records in the phased VCF inside the window,
                   i.e. exactly what build_gene_sequences.py put into the
                   gene sequences
    n_polymorphic  of those, the ones with at least one alt allele and at
                   least one ref allele among the QC-passed participants

The UKBB input is ukb_snp_all, the directly genotyped array (~800K SNPs
genome-wide), not imputed or sequenced data, so these counts are expected to
sit far below a WGS cohort's for the same gene.

Genes the pipeline never sequenced keep a row with a status saying why
(not in the GRCh37 GTF as protein-coding, or over the length cap).

It also writes <out>.bins.tsv, the same table Ning sent for HPP
(num_variants binned 0, 1, 2, 3, 4, 5+ against num_genes), with UKBB's
significant genes per bin from compute_recovery.py's output and HPP's
num_genes (hpp_variant_bins.tsv) as the last column.

Run with "wasp_env" (pysam):
    python count_gene_variants.py --out ukbb_variants_per_gene.tsv
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import pysam

import config
import target_genes
from build_gene_sequences import build_gene_model, gene_window

STATUS_OK = "ok"
STATUS_NOT_IN_GTF = "not_protein_coding_in_grch37_gtf"
STATUS_TOO_LONG = "longer_than_cap"

BIN_LABELS = ["0", "1", "2", "3", "4", "5+"]
TOP_BIN = 5
HPP_BINS_TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "hpp_variant_bins.tsv")
UKB_SIGNIFICANT_NAME = "ukb_significant_genes.tsv"  # under config.ASSOC_DIR


def allele_counts(genotypes) -> tuple[int, int]:
    """(ref allele count, alt allele count) over called alleles, missing skipped."""
    called = [a for gt in genotypes for a in gt[:2] if a is not None]
    n_alt = sum(1 for a in called if a > 0)
    return len(called) - n_alt, n_alt


def is_biallelic_snp(ref: str | None, alts) -> bool:
    """Same record filter as build_gene_sequences.collect_variants."""
    return (ref is not None and len(ref) == 1 and alts is not None
            and len(alts) == 1 and len(alts[0]) == 1)


def count_window(vf: pysam.VariantFile, chrom: str, start: int,
                 end: int) -> tuple[int, int]:
    """(n_variants, n_polymorphic) for 1-based inclusive [start, end]."""
    n_variants, n_poly = 0, 0
    for rec in vf.fetch(chrom, start - 1, end):
        if not (start <= rec.pos <= end) or not is_biallelic_snp(rec.ref, rec.alts):
            continue
        n_variants += 1
        n_ref, n_alt = allele_counts(s.get("GT", (None, None))
                                     for s in rec.samples.values())
        n_poly += int(n_ref > 0 and n_alt > 0)
    return n_variants, n_poly


def lookup(gene_model: dict, bare_id: str):
    """GTF coords by unversioned id; HPP and v19 versions differ."""
    return next((v for k, v in gene_model.items()
                 if target_genes.strip_version(k) == bare_id), None)


def gene_row(gene_id: str, gene_model: dict, vf: pysam.VariantFile) -> dict:
    bare_id = target_genes.strip_version(gene_id)
    row = {"gene_id": gene_id, "chrom": None, "start": None, "end": None,
           "length_bp": None, "n_variants": 0, "n_polymorphic": 0}
    coords = lookup(gene_model, bare_id)
    if coords is None:
        return {**row, "status": STATUS_NOT_IN_GTF}
    chrom, gstart, gend, strand = coords
    wstart, wend = gene_window(chrom, gstart, gend, strand)
    length = wend - wstart + 1
    row = {**row, "chrom": chrom, "start": wstart, "end": wend, "length_bp": length}
    if length > config.MAX_GENE_LEN:
        return {**row, "status": STATUS_TOO_LONG}
    n_variants, n_poly = count_window(vf, chrom, wstart, wend)
    return {**row, "n_variants": n_variants, "n_polymorphic": n_poly,
            "status": STATUS_OK}


def variant_bin(n: int) -> str:
    return BIN_LABELS[min(int(n), TOP_BIN)]


def bin_table(counts: pd.DataFrame, significant: set[str] | None,
              hpp_bins: pd.DataFrame | None) -> pd.DataFrame:
    """Genes per num_variants bin, in the layout of Ning's HPP table.

    Untested genes (not sequenced, or no variant) fall in their bin with
    n_variants; significance is counted only where a gene was tested.
    """
    bare = counts["gene_id"].map(target_genes.strip_version)
    frame = counts.assign(num_variants=counts["n_variants"].map(variant_bin))
    table = (frame.groupby("num_variants").size()
             .reindex(BIN_LABELS, fill_value=0).rename("num_genes").to_frame())
    if significant is not None:
        sig = frame[bare.isin(significant)].groupby("num_variants").size()
        table["significant"] = sig.reindex(BIN_LABELS, fill_value=0)
        table["rate"] = (table["significant"] / table["num_genes"]).where(
            table["num_genes"] > 0)
    if hpp_bins is not None:
        hpp = hpp_bins.astype({"num_variants": str}).set_index("num_variants")
        table["hpp_num_genes"] = hpp["num_genes"].reindex(BIN_LABELS)
    return table.reset_index()


def load_significant(path: str) -> set[str] | None:
    if not os.path.exists(path):
        print(f"[warn] {path} not found, bins table has no significance column")
        return None
    return set(pd.read_csv(path, sep="\t")["gene_id"].map(target_genes.strip_version))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--targets", default=config.TARGET_GENES_TSV)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    targets = target_genes.load_targets(args.targets)
    gene_ids = list(dict.fromkeys(targets[target_genes.GENE_ID_COLUMN]))
    gene_model = build_gene_model()
    vf = pysam.VariantFile(config.PHASED_VCF)
    print(f"{len(gene_ids)} target genes, {len(vf.header.samples):,} participants")

    counts = pd.DataFrame([gene_row(g, gene_model, vf) for g in gene_ids])
    out = targets.drop(columns="bare_gene_id").merge(counts, on="gene_id", how="left")
    out.to_csv(args.out, sep="\t", index=False)

    bins_path = re.sub(r"\.tsv$", "", args.out) + ".bins.tsv"
    hpp_bins = (pd.read_csv(HPP_BINS_TSV, sep="\t")
                if os.path.exists(HPP_BINS_TSV) else None)
    bins = bin_table(counts, load_significant(os.path.join(config.ASSOC_DIR, UKB_SIGNIFICANT_NAME)), hpp_bins)
    bins.to_csv(bins_path, sep="\t", index=False)
    print(f"Wrote {bins_path}")
    print(bins.to_string(index=False))

    ok = counts[counts["status"] == STATUS_OK]
    print(f"Wrote {args.out}")
    print(counts["status"].value_counts().to_string())
    print(f"n_variants among sequenced genes: median {np.median(ok['n_variants']):.0f}, "
          f"zero in {(ok['n_variants'] == 0).sum()} genes")


if __name__ == "__main__":
    main()
