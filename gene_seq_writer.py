"""Stream per-participant gene sequences to disk without materializing them all.

The previous approach built hap1/hap2 as one Python list per participant, which
for 43,706 participants and a 30 kb gene is ~21 GB of pointers before a single
row is written -- the OOM killer got it. Here a gene is held as the reference
string plus a compact (n_variants x n_participants) int8 genotype matrix, and
each participant's two haplotypes are built, written and discarded one at a
time. Peak memory is one sequence plus the matrix (~13 MB for 300 variants).

Sequences are written UNSPACED. get_hpp_embedding-AG_clean.py:78 does
`" ".join(row[key])` itself, so writing space-separated bases here would make
it join the spaces too -- doubling every sequence length and tokenizing
garbage.

No pysam dependency, so this is unit-testable off-cluster.
"""
import numpy as np

# Genotype codes. Phase is meaningful: Eagle gives a true haplotype assignment,
# so HAP1_ALT and HAP2_ALT are different states, not one heterozygous state.
CODE_HOM_REF = 0
CODE_HAP1_ALT = 1
CODE_HAP2_ALT = 2
CODE_BOTH_ALT = 3
CODE_MISSING = 4   # no genotype, or a multi-allelic blind spot: N on both haps

COLUMNS = ("participant_id", "ref_seq", "mutation_seq_1", "mutation_seq_2")


def code_genotype(allele1: int | None, allele2: int | None) -> int:
    """Phased GT -> genotype code. None on either side means missing."""
    if allele1 is None or allele2 is None:
        return CODE_MISSING
    if allele1 and allele2:
        return CODE_BOTH_ALT
    if allele1:
        return CODE_HAP1_ALT
    if allele2:
        return CODE_HAP2_ALT
    return CODE_HOM_REF


def build_haplotype_pair(ref_seq: str, positions: np.ndarray, alts: list[str],
                         codes: np.ndarray) -> tuple[str, str]:
    """One participant's two haplotypes, from the reference plus their codes.

    `codes` is that participant's column of the genotype matrix, aligned to
    `positions`/`alts`. Every base is either the reference base or a recorded
    alternate (or N where the genotype is missing), so this reproduces exactly
    what a per-participant edit of the reference would have produced.
    """
    if not (len(positions) == len(alts) == len(codes)):
        raise ValueError(
            f"positions/alts/codes length mismatch: {len(positions)}, "
            f"{len(alts)}, {len(codes)}"
        )
    hap1 = bytearray(ref_seq, "ascii")
    hap2 = bytearray(ref_seq, "ascii")

    for idx, alt, code in zip(positions, alts, codes):
        if code == CODE_HOM_REF:
            continue
        alt_byte = ord(alt)
        if code == CODE_MISSING:
            hap1[idx] = hap2[idx] = ord("N")
        elif code == CODE_BOTH_ALT:
            hap1[idx] = hap2[idx] = alt_byte
        elif code == CODE_HAP1_ALT:
            hap1[idx] = alt_byte
        elif code == CODE_HAP2_ALT:
            hap2[idx] = alt_byte
        else:
            raise ValueError(f"unknown genotype code {code}")

    return hap1.decode("ascii"), hap2.decode("ascii")


def write_gene_tsv(path: str, participants: list[str], ref_seq: str,
                   positions: np.ndarray, alts: list[str],
                   codes: np.ndarray) -> int:
    """Write one gene's TSV, one participant per row. Returns rows written.

    Written to `path` directly; the caller is responsible for writing to a temp
    path and renaming if it wants crash-atomicity.
    """
    if codes.shape != (len(positions), len(participants)):
        raise ValueError(
            f"codes shape {codes.shape} does not match "
            f"({len(positions)} variants, {len(participants)} participants)"
        )

    with open(path, "w") as fh:
        fh.write("\t".join(COLUMNS) + "\n")
        for j, participant in enumerate(participants):
            hap1, hap2 = build_haplotype_pair(
                ref_seq, positions, alts, codes[:, j]
            )
            fh.write(f"{participant}\t{ref_seq}\t{hap1}\t{hap2}\n")
    return len(participants)
