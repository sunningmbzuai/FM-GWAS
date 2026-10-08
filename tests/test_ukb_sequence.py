"""Unit tests for strand orientation.

The downstream embedding step reads its TSS window off the first 500 positions
of whatever sequence this pipeline writes, so getting minus-strand orientation
wrong is silent: the run completes and half the genes are embedded at the wrong
end. These tests pin it down.
"""
import pytest

from ukb_sequence import orient_to_tss, reverse_complement


@pytest.mark.unit
def test_reverse_complement_round_trips():
    seq = "ACGTTGCAAN"
    assert reverse_complement(reverse_complement(seq)) == seq


@pytest.mark.unit
def test_reverse_complement_pairs_bases_and_preserves_n():
    assert reverse_complement("ACGT") == "ACGT"
    assert reverse_complement("AAAA") == "TTTT"
    assert reverse_complement("ANG") == "CNT"


@pytest.mark.unit
def test_plus_strand_is_left_genomic_forward():
    assert orient_to_tss("ACGTACGT", "+") == "ACGTACGT"


@pytest.mark.unit
def test_minus_strand_puts_the_tss_at_position_zero():
    """A minus-strand gene's TSS is at the genomic END of its window."""
    #  genomic-forward window: TES ................ TSS
    genomic = "AAAAAAAA" + "GGGCCC"      # last 6 bp abut the TSS
    oriented = orient_to_tss(genomic, "-")

    assert oriented.startswith(reverse_complement("GGGCCC"))
    assert len(oriented) == len(genomic)


@pytest.mark.unit
def test_unknown_strand_is_rejected():
    with pytest.raises(ValueError, match="strand"):
        orient_to_tss("ACGT", ".")
