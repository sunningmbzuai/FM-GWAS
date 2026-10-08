"""Unit tests for building the target list from the HPP novel-gene exports.

Fixtures are temp files, so nothing here touches cluster paths. The
distinctions pinned down: a CSV export and a TSV export read the same, a
duplicated association is not double-counted, one saturated (trait, sex) cell
cannot crowd out the rest, and the written table is tab-separated with the
p-values in the committed 2-decimal scientific form.
"""
import csv
import os

import pytest

import build_target_genes as B

HEADER = ["gene_id", "p_F_analytic_min", "p_F_analytic_max", "n_rows",
          "trait", "sex"]


def write_source(path, rows, delimiter="\t", bom=False):
    text = delimiter.join(HEADER) + "\n"
    for row in rows:
        text += delimiter.join(str(v) for v in row) + "\n"
    path.write_text(("﻿" if bom else "") + text, encoding="utf-8")
    return str(path)


def cell_rows(trait, sex, n, start_p):
    return [(f"ENSG{i:011d}.1", start_p * (i + 1), start_p * (i + 1) * 10, 5,
             trait, sex) for i in range(n)]


@pytest.mark.unit
def test_csv_and_tsv_sources_read_alike(tmp_path):
    rows = cell_rows("Cholesterol", "M", 2, 1e-11)
    tsv = B.read_source(write_source(tmp_path / "a.tsv", rows))
    csv_ = B.read_source(write_source(tmp_path / "a.csv", rows, delimiter=","))
    assert tsv == csv_


@pytest.mark.unit
def test_bom_does_not_corrupt_the_first_column(tmp_path):
    path = write_source(tmp_path / "bom.tsv",
                        cell_rows("Height", "F", 1, 1e-9), bom=True)
    assert B.read_source(path)[0]["gene_id"] == "ENSG00000000000.1"


@pytest.mark.unit
def test_missing_column_is_an_error(tmp_path):
    path = tmp_path / "bad.tsv"
    path.write_text("gene_id\tsex\nENSG1\tAll\n", encoding="utf-8")
    with pytest.raises(KeyError):
        B.read_source(str(path))


@pytest.mark.unit
def test_union_deduplicates_the_same_association(tmp_path):
    shared = ("ENSG00000174327.6", 4.3e-11, 6.5e-09, 15, "Cholesterol", "M")
    rounded = ("ENSG00000174327.6", 4.27e-11, 6.51e-09, 15, "Cholesterol", "M")
    only_in_second = ("ENSG00000105131.3", 2.2e-11, 7.4e-10, 15,
                      "Cholesterol", "F")
    merged = B.union_sources([
        write_source(tmp_path / "first.tsv", [shared]),
        write_source(tmp_path / "second.csv", [rounded, only_in_second],
                     delimiter=","),
    ])
    assert len(merged) == 2
    # First source wins, so the union never depends on float equality.
    assert merged[0]["p_F_analytic_min"] == "4.3e-11"


@pytest.mark.unit
def test_same_gene_in_two_cells_is_kept_twice(tmp_path):
    rows = [("ENSG1.1", 1e-11, 1e-9, 5, "VAT", "M"),
            ("ENSG1.1", 2e-11, 1e-9, 5, "VAT", "F")]
    assert len(B.union_sources([write_source(tmp_path / "s.tsv", rows)])) == 2


@pytest.mark.unit
def test_selection_caps_each_cell_and_ranks_by_p():
    rows = [dict(zip(HEADER, r)) for r in
            cell_rows("VAT", "M", 10, 1e-9) + cell_rows("T2D", "F", 1, 5e-8)]
    kept = B.select_targets(rows, max_per_cell=4, max_rows=18)
    assert [r["trait"] for r in kept].count("VAT") == 4
    # The crowded cell keeps its four smallest p-values, and T2D survives.
    assert [r["trait"] for r in kept][-1] == "T2D"
    assert [float(r["p_F_analytic_min"]) for r in kept] == sorted(
        float(r["p_F_analytic_min"]) for r in kept)


@pytest.mark.unit
def test_selection_truncates_to_max_rows():
    rows = [dict(zip(HEADER, r)) for r in
            cell_rows("VAT", "M", 5, 1e-9) + cell_rows("T2D", "F", 5, 2e-9)]
    assert len(B.select_targets(rows, max_per_cell=4, max_rows=3)) == 3


@pytest.mark.unit
def test_written_table_is_tsv_with_scientific_p_values(tmp_path):
    rows = [dict(zip(HEADER, ("ENSG00000174327.6", "4.2716493520184413e-11",
                              "6.5148807231004e-09", 15, "Cholesterol",
                              "M")))]
    out = tmp_path / "targets.tsv"
    B.write_targets(rows, str(out))
    written = list(csv.DictReader(out.open(newline=""), delimiter="\t"))
    assert written[0]["p_F_analytic_min"] == "4.27E-11"
    assert written[0]["p_F_analytic_max"] == "6.51E-09"
    assert written[0]["gene_id"] == "ENSG00000174327.6"


@pytest.mark.unit
@pytest.mark.skipif(
    not all(os.path.exists(p) for p in B.SOURCE_LISTS + [B.OUTPUT_TSV]),
    reason="target list and HPP exports are data, not committed",
)
def test_committed_targets_match_a_rebuild(tmp_path):
    """The repo's shortlist is exactly what the sources regenerate."""
    rebuilt = tmp_path / "rebuilt.tsv"
    B.write_targets(B.select_targets(B.union_sources(B.SOURCE_LISTS)),
                    str(rebuilt))
    assert rebuilt.read_text() == open(B.OUTPUT_TSV).read()


UNRANKED = "gene_id\tsex\ttrait\nENSG2.1\tAll\tHeight\nENSG1.1\tAll\tHeight\n"


@pytest.mark.unit
def test_unranked_export_without_p_values_reads(tmp_path):
    path = tmp_path / "unranked.tsv"
    path.write_text(UNRANKED, encoding="utf-8")
    rows = B.read_source(str(path))
    assert [r["gene_id"] for r in rows] == ["ENSG2.1", "ENSG1.1"]


@pytest.mark.unit
def test_unranked_rows_keep_file_order_after_ranked_ones(tmp_path):
    path = tmp_path / "unranked.tsv"
    path.write_text(UNRANKED, encoding="utf-8")
    ranked = dict(zip(HEADER, ("ENSG3.1", "1e-9", "1e-8", 5, "VAT", "M")))
    kept = B.select_targets(B.read_source(str(path)) + [ranked])
    assert [r["gene_id"] for r in kept] == ["ENSG3.1", "ENSG2.1", "ENSG1.1"]


@pytest.mark.unit
def test_unranked_rows_write_blank_p_values(tmp_path):
    path = tmp_path / "unranked.tsv"
    path.write_text(UNRANKED, encoding="utf-8")
    out = tmp_path / "targets.tsv"
    B.write_targets(B.read_source(str(path)), str(out))
    written = list(csv.DictReader(out.open(newline=""), delimiter="\t"))
    assert written[0]["p_F_analytic_min"] == ""
    assert written[0]["n_rows"] == ""
    assert (written[0]["trait"], written[0]["sex"]) == ("Height", "All")
