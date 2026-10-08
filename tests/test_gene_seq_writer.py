"""Unit tests for streamed haplotype construction.

The key property: building sequences from (reference + genotype matrix) must
give exactly what editing a per-participant copy of the reference would have
given, including on the minus strand where reverse-complementing moves every
variant. A naive reference implementation is included to compare against.
"""
import numpy as np
import pandas as pd
import pytest

import gene_seq_writer as W
from ukb_sequence import orient_to_tss

COMPLEMENT_BASE = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}


def naive_haplotypes(ref: str, positions, alts, codes) -> tuple[str, str]:
    """The previous implementation's logic: edit lists of single characters."""
    hap1, hap2 = list(ref), list(ref)
    for idx, alt, code in zip(positions, alts, codes):
        if code == W.CODE_MISSING:
            hap1[idx] = hap2[idx] = "N"
        elif code == W.CODE_BOTH_ALT:
            hap1[idx] = hap2[idx] = alt
        elif code == W.CODE_HAP1_ALT:
            hap1[idx] = alt
        elif code == W.CODE_HAP2_ALT:
            hap2[idx] = alt
    return "".join(hap1), "".join(hap2)


@pytest.mark.unit
@pytest.mark.parametrize("gt,expected", [
    ((0, 0), W.CODE_HOM_REF),
    ((1, 0), W.CODE_HAP1_ALT),
    ((0, 1), W.CODE_HAP2_ALT),
    ((1, 1), W.CODE_BOTH_ALT),
    ((None, None), W.CODE_MISSING),
    ((None, 1), W.CODE_MISSING),
])
def test_genotype_coding(gt, expected):
    assert W.code_genotype(*gt) == expected


@pytest.mark.unit
def test_phase_is_not_collapsed():
    """1|0 and 0|1 are different haplotype assignments, not one het state."""
    assert W.code_genotype(1, 0) != W.code_genotype(0, 1)


@pytest.mark.unit
def test_haplotype_pair_matches_the_naive_implementation():
    ref = "ACGTACGTAC"
    positions = np.array([1, 4, 7])
    alts = ["T", "G", "C"]
    codes = np.array([W.CODE_HAP1_ALT, W.CODE_BOTH_ALT, W.CODE_MISSING],
                     dtype=np.int8)

    assert W.build_haplotype_pair(ref, positions, alts, codes) == \
        naive_haplotypes(ref, positions, alts, codes)


@pytest.mark.unit
def test_hom_ref_leaves_the_sequence_untouched():
    ref = "ACGTACGT"
    hap1, hap2 = W.build_haplotype_pair(
        ref, np.array([2]), ["T"], np.array([W.CODE_HOM_REF], dtype=np.int8)
    )

    assert hap1 == hap2 == ref


@pytest.mark.unit
def test_minus_strand_remap_matches_reverse_complementing_the_haplotypes():
    """Position remap + allele complement == building then revcomp'ing.

    The old code reverse-complemented finished haplotypes, which moved and
    complemented variants implicitly. The streaming version has to do both
    explicitly, so this pins the two together.
    """
    ref = "ACGTACGTAC"
    positions = np.array([1, 4, 7])
    alts = ["T", "G", "C"]
    codes = np.array([W.CODE_HAP1_ALT, W.CODE_BOTH_ALT, W.CODE_HAP2_ALT],
                     dtype=np.int8)

    # old path: build on the forward strand, then reverse-complement
    fwd1, fwd2 = naive_haplotypes(ref, positions, alts, codes)
    expected = (orient_to_tss(fwd1, "-"), orient_to_tss(fwd2, "-"))

    # new path: reverse-complement the reference, remap positions, complement alts
    rc_ref = orient_to_tss(ref, "-")
    rc_positions = (len(ref) - 1) - positions
    rc_alts = [COMPLEMENT_BASE[a] for a in alts]
    actual = W.build_haplotype_pair(rc_ref, rc_positions, rc_alts, codes)

    assert actual == expected


@pytest.mark.unit
def test_missing_genotypes_survive_the_minus_strand_remap_as_n():
    ref = "ACGTAC"
    codes = np.array([W.CODE_MISSING], dtype=np.int8)
    rc_ref = orient_to_tss(ref, "-")

    hap1, hap2 = W.build_haplotype_pair(rc_ref, np.array([len(ref) - 1 - 2]),
                                        ["N"], codes)

    assert hap1.count("N") == 1 and hap2.count("N") == 1


@pytest.mark.unit
def test_written_tsv_is_unspaced_and_has_the_expected_columns(tmp_path):
    """get_hpp_embedding does `" ".join(seq)` itself -- spaces here would double."""
    out = tmp_path / "ENSG1.tsv"
    ref = "ACGTACGT"
    codes = np.array([[W.CODE_HAP1_ALT, W.CODE_HOM_REF]], dtype=np.int8)

    rows = W.write_gene_tsv(str(out), ["p1", "p2"], ref, np.array([2]), ["T"], codes)

    frame = pd.read_csv(out, sep="\t", dtype=str)
    assert rows == 2
    assert list(frame.columns) == list(W.COLUMNS)
    assert " " not in frame["ref_seq"].iloc[0]
    assert frame["ref_seq"].iloc[0] == ref
    assert frame["mutation_seq_1"].iloc[0] == "ACTTACGT"   # p1 hap1 edited
    assert frame["mutation_seq_2"].iloc[0] == ref          # p1 hap2 untouched
    assert frame["mutation_seq_1"].iloc[1] == ref          # p2 hom ref


@pytest.mark.unit
def test_write_rejects_a_mismatched_matrix(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        W.write_gene_tsv(
            str(tmp_path / "g.tsv"), ["p1", "p2"], "ACGT",
            np.array([1]), ["T"], np.zeros((1, 3), dtype=np.int8),
        )


@pytest.mark.unit
def test_build_rejects_ragged_inputs():
    with pytest.raises(ValueError, match="length mismatch"):
        W.build_haplotype_pair("ACGT", np.array([1, 2]), ["T"],
                               np.array([0, 0], dtype=np.int8))
