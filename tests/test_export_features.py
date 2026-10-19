"""export_features writes the .pt's raw fields, and rewrites old view tables."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import export_features as X  # noqa: E402
import feature_tables  # noqa: E402

GENE = "ENSG7.1"


def write_pt(root, n=5):
    data = {X.SUBJECT_KEY: [str(1000 + i) for i in range(n)]}
    for key in X.RAW_FIELDS:
        value = torch.randn(n, 6)
        data[key] = value.to(torch.bfloat16) if "tss500" in key else value
    (root / GENE).mkdir(parents=True)
    torch.save(data, root / GENE / f"{GENE}.pt")
    return data


@pytest.mark.unit
def test_table_holds_the_raw_fields(tmp_path):
    data = write_pt(tmp_path / "pt")
    X.export_gene(GENE, str(tmp_path / "pt"), str(tmp_path / "tab"))
    with np.load(tmp_path / "tab" / GENE / f"{GENE}.npz") as table:
        assert sorted(table.files) == sorted((X.SUBJECT_KEY,) + X.RAW_FIELDS)
        for key in X.RAW_FIELDS:
            assert np.array_equal(table[key], data[key].float().numpy())
            assert table[key].dtype == np.float32


@pytest.mark.unit
def test_old_view_table_is_rewritten(tmp_path):
    write_pt(tmp_path / "pt")
    old = tmp_path / "tab" / GENE / f"{GENE}.npz"
    old.parent.mkdir(parents=True)
    np.savez(old, subject_id=np.array(["1000"]), concat_cage=np.zeros((1, 6)))
    assert not X.is_current(str(old))
    assert X.export_gene(GENE, str(tmp_path / "pt"), str(tmp_path / "tab"))["status"] == "written"
    assert X.is_current(str(old))
    assert X.export_gene(GENE, str(tmp_path / "pt"), str(tmp_path / "tab"))["status"] == "exists"


@pytest.mark.unit
def test_diagnostic_views_derive_from_the_raw_fields(tmp_path):
    data = write_pt(tmp_path / "pt")
    X.export_gene(GENE, str(tmp_path / "pt"), str(tmp_path / "tab"))
    _, views = feature_tables.load_gene(GENE, str(tmp_path / "tab"))
    assert len(views) == 16
    cage = data["concat_cage_mean"].numpy()
    assert np.array_equal(views["concat_cage"], cage)
    assert np.array_equal(views["add_cage"], cage[:, :3] + cage[:, 3:])
