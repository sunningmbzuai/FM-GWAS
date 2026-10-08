"""Strand orientation for gene sequences. No pysam dependency, so it is testable
anywhere -- build_gene_sequences.py imports it.

The downstream embedding step derives its TSS window from the FIRST 500
positions of the sequence (get_hpp_embedding-AG_clean.py's
`mut1_track[:, :tss_window]`), so the sequences this pipeline writes must be
oriented 5'->3' along the gene, not genomic-forward.
"""

COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def orient_to_tss(seq: str, strand: str) -> str:
    """Return the sequence 5'->3' along the gene, so position 0 is the TSS.

    GTF coordinates are genomic-forward, so a minus-strand gene's window STARTS
    at the TES. Left genomic-forward, such a gene's "TSS-500bp" embedding would
    be computed at the wrong end of the gene.
    """
    if strand == "+":
        return seq
    if strand == "-":
        return reverse_complement(seq)
    raise ValueError(f"unknown strand {strand!r}; expected '+' or '-'")
