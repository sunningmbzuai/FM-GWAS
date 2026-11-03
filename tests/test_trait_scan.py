from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")
import trait_rules as R  # noqa: E402
import trait_scan as T  # noqa: E402


@pytest.fixture()
def dump(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "MIN_N", 50)
    monkeypatch.setattr(R, "MIN_MINOR_CLASS", 10)
    rng = np.random.default_rng(0)
    n = 300
    frame = pd.DataFrame({"eid": np.arange(1, n + 1)})
    frame["50-0.0"] = rng.normal(170, 10, n)                       # height: quantitative
    frame["2443-0.0"] = rng.choice([0.0, 1.0, -1.0, -3.0], n)      # binary after codes
    for a in range(8):
        frame[f"20002-0.{a}"] = rng.choice([1065.0, 1473.0, np.nan], n)  # multi-select
    frame["22009-0.1"] = rng.normal(size=n)                        # genetic PC: excluded
    frame["53-0.0"] = "2008-01-01"                                 # date: excluded
    frame["1747-0.0"] = rng.choice([1.0, 2.0, 3.0, 4.0], n)        # categorical: skipped
    frame["note"] = "x"                                            # text: skipped
    frame["height_copy"] = frame["50-0.0"]                         # exact duplicate
    path = tmp_path / "ukb.parquet"
    frame.to_parquet(path)
    return path, n


@pytest.mark.unit
def test_scan_keeps_traits_skips_the_rest_and_drops_duplicates(dump, tmp_path):
    path, n = dump
    ids = np.arange(1, n + 1, dtype=np.int64)
    T._init(ids, {})
    skipped: list[dict] = []
    names = T.header(path)
    records = [r for task in T.plan_batches(path, names, "eid", skipped)
               for r in T.scan_batch(task)]
    people = pd.DataFrame({"participant_id": ids, "sex": 0, "age": 50.0})
    out = tmp_path / "out"
    out.mkdir()
    catalog = T.assemble(records + skipped, out, people)

    kinds = dict(zip(catalog.field, catalog.kind))
    assert kinds == {"50": "quantitative", "2443": "binary"}
    values = np.load(out / "trait_values.npy")
    assert values.shape == (n, 2)
    binary = values[:, catalog.set_index("field").loc["2443", "column"]]
    assert set(np.unique(binary[np.isfinite(binary)])) == {0.0, 1.0}

    reasons = pd.read_csv(out / "trait_skipped.tsv", sep="\t").set_index("field").reason
    assert "multi-select" in reasons["20002"]
    assert "administrative" in reasons["22009"]
    assert "administrative" in reasons["53"]
    assert "levels" in reasons["1747"]
    assert reasons["note"] == "not numeric"
    assert reasons["height_copy"].startswith("duplicate of ")


@pytest.mark.unit
def test_tabular_files_walks_subdirectories(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    for name in ("a/x.parquet", "a/b/y.csv", "a/b/z.json", ".hidden.tsv"):
        (tmp_path / name).write_text("")
    found = sorted(p.relative_to(tmp_path).as_posix() for p in T.tabular_files(tmp_path))
    assert found == ["a/b/y.csv", "a/x.parquet"]
