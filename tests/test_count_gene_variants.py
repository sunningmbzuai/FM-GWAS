import pytest

pysam = pytest.importorskip("pysam")

from count_gene_variants import allele_counts, is_biallelic_snp, lookup


def test_allele_counts_skips_missing():
    gts = [(0, 0), (0, 1), (1, 1), (None, None)]
    assert allele_counts(gts) == (3, 3)


def test_monomorphic_has_no_alt():
    assert allele_counts([(0, 0), (0, 0)]) == (4, 0)


@pytest.mark.parametrize("ref,alts,expected", [
    ("A", ("G",), True),
    ("A", ("G", "T"), False),
    ("AT", ("A",), False),
    ("A", ("AT",), False),
    (None, ("G",), False),
    ("A", None, False),
])
def test_is_biallelic_snp(ref, alts, expected):
    assert is_biallelic_snp(ref, alts) is expected


def test_lookup_ignores_version():
    model = {"ENSG00000174327.2": ("17", 10, 20, "+")}
    assert lookup(model, "ENSG00000174327") == ("17", 10, 20, "+")
    assert lookup(model, "ENSG00000000001") is None


def _counts(ns):
    import pandas as pd
    return pd.DataFrame({"gene_id": [f"ENSG{i:011d}.1" for i in range(len(ns))],
                         "n_variants": ns})


def test_variant_bin_caps_at_five_plus():
    from count_gene_variants import variant_bin
    assert [variant_bin(n) for n in (0, 1, 4, 5, 300)] == ["0", "1", "4", "5+", "5+"]


def test_bin_table_matches_hpp_layout():
    import pandas as pd
    from count_gene_variants import bin_table
    counts = _counts([0, 0, 1, 2, 2, 9])
    hpp = pd.DataFrame({"num_variants": ["0", "1", "2", "3", "4", "5+"],
                        "num_genes": [0, 35, 22, 0, 2, 45]})
    table = bin_table(counts, {"ENSG00000000002", "ENSG00000000005"}, hpp)
    assert table["num_variants"].tolist() == ["0", "1", "2", "3", "4", "5+"]
    assert table["num_genes"].tolist() == [2, 1, 2, 0, 0, 1]
    assert table["significant"].tolist() == [0, 1, 0, 0, 0, 1]
    assert table["hpp_num_genes"].tolist() == [0, 35, 22, 0, 2, 45]
    assert pd.isna(table.loc[3, "rate"])


def test_bin_table_without_significance_or_hpp():
    from count_gene_variants import bin_table
    table = bin_table(_counts([1, 7]), None, None)
    assert list(table.columns) == ["num_variants", "num_genes"]
