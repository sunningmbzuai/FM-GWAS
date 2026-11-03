"""Rules that turn one raw UKB phenotype column group into a testable trait.

Pure functions, no I/O, so the decisions trait_scan.py makes can be tested and
argued with on their own. A "field" is everything that measures one quantity:
the "<field>-<instance>.<array>" columns of the raw dump, the "<name>_i<k>"
columns of the phenotypes export, or a single column of any other table.

Decision per field (values already restricted to QC-passed participants):

  1. Non-numeric (strings, dates)                    -> skip
  2. More than MAX_ARRAYS array columns per instance -> skip (multi-select codes,
     e.g. 20002 self-reported illness, 20003 medication)
  3. Instance: the one with the most participants measured; repeated array
     readings within it (BP, grip, spirometry) are averaged.
  4. Integer-valued fields: UKB "do not know" / "prefer not to answer" style
     negative codes become missing.
  5. Exactly 2 distinct values                       -> binary (lower = 0)
     at least MIN_QUANT_LEVELS distinct values       -> quantitative
     anything between                                -> skip (ordinal / categorical)
  6. Too few participants, or too few in the smaller binary class -> skip

A data dictionary (UKB Showcase ValueType), when one is found, overrides the
value-based guess in step 5: Categorical fields with more than two levels and
Date/Text/Compound fields are skipped whatever their values look like.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

MAX_ARRAYS = 4
MIN_QUANT_LEVELS = 10
MIN_N = 2000
MIN_MINOR_CLASS = 100

# UKB codings for missing-type answers (codings 100291, 100349, 100373, ...).
UKB_SPECIAL_CODES = frozenset({-1, -3, -4, -7, -10, -11, -13, -17, -21, -23,
                               -27, -121, -131, -313, -818})

# Administrative, genetic-QC, registry and covariate fields: never a trait.
EXCLUDED_FIELD_RANGES = (
    (31, 31), (34, 34), (52, 55), (189, 190), (200, 200), (21003, 21003),
    (21022, 21022), (20074, 20075), (22000, 22299), (40000, 42999),
)

DUMP_COLUMN = re.compile(r"^(\d+)-(\d+)\.(\d+)$")
EXPORT_COLUMN = re.compile(r"^(.+)_i(\d+)$")
ID_COLUMNS = ("eid", "f.eid", "participant_id", "ID", "id", "Eid")


@dataclass(frozen=True)
class Column:
    name: str
    field: str
    instance: int
    array: int


def parse_column(name: str) -> Column:
    m = DUMP_COLUMN.match(name)
    if m:
        return Column(name, m.group(1), int(m.group(2)), int(m.group(3)))
    m = EXPORT_COLUMN.match(name)
    if m:
        return Column(name, m.group(1), int(m.group(2)), 0)
    return Column(name, name, 0, 0)


def group_fields(names: list[str], id_column: str) -> dict[str, list[Column]]:
    """field -> its columns, in file order, id column left out."""
    groups: dict[str, list[Column]] = {}
    for name in names:
        if name == id_column:
            continue
        col = parse_column(name)
        groups.setdefault(col.field, []).append(col)
    return groups


def is_excluded(field: str) -> bool:
    if not field.isdigit():
        return False
    number = int(field)
    return any(lo <= number <= hi for lo, hi in EXCLUDED_FIELD_RANGES)


def too_many_arrays(columns: list[Column]) -> bool:
    per_instance: dict[int, int] = {}
    for col in columns:
        per_instance[col.instance] = per_instance.get(col.instance, 0) + 1
    return max(per_instance.values()) > MAX_ARRAYS


def clean_special_codes(values: pd.Series) -> pd.Series:
    """Missing-type negative codes -> NaN, for integer-valued fields only."""
    finite = values.dropna()
    if finite.empty or not np.all(np.equal(np.mod(finite, 1), 0)):
        return values
    return values.mask(values.isin(UKB_SPECIAL_CODES))


def best_instance(frame: pd.DataFrame, columns: list[Column]) -> tuple[int, pd.Series]:
    """(instance, per-participant mean of its arrays) for the best-covered instance."""
    best: tuple[int, pd.Series] | None = None
    for instance in sorted({c.instance for c in columns}):
        names = [c.name for c in columns if c.instance == instance]
        values = frame[names].apply(pd.to_numeric, errors="coerce")
        values = values.apply(clean_special_codes)
        merged = values.mean(axis=1, skipna=True)
        if best is None or merged.notna().sum() > best[1].notna().sum():
            best = (instance, merged)
    assert best is not None
    return best


@dataclass(frozen=True)
class Verdict:
    kind: str            # "quantitative" | "binary" | "skip"
    reason: str
    n: int
    n_minor: int = 0
    low: float = np.nan  # binary: the value coded 0
    high: float = np.nan  # binary: the value coded 1


def classify(values: pd.Series, value_type: str | None = None) -> Verdict:
    """Steps 5-6 of the module docstring on one field's merged values."""
    present = values.dropna()
    n = int(len(present))
    if value_type and value_type.lower() in {"date", "time", "text", "compound"}:
        return Verdict("skip", f"value type {value_type}", n)
    levels = np.unique(present.to_numpy())
    if len(levels) < 2:
        return Verdict("skip", "constant", n)
    if n < MIN_N:
        return Verdict("skip", f"n={n} < {MIN_N}", n)
    if len(levels) == 2:
        counts = present.value_counts()
        minor = int(counts.min())
        if minor < MIN_MINOR_CLASS:
            return Verdict("skip", f"minor class {minor} < {MIN_MINOR_CLASS}", n, minor)
        return Verdict("binary", "two levels", n, minor, float(levels[0]), float(levels[1]))
    categorical = bool(value_type and value_type.lower().startswith("categorical"))
    if categorical:
        return Verdict("skip", f"categorical, {len(levels)} levels", n)
    if len(levels) < MIN_QUANT_LEVELS:
        return Verdict("skip", f"{len(levels)} levels (ordinal/categorical)", n)
    return Verdict("quantitative", f"{len(levels)} levels", n)


def encode(values: pd.Series, verdict: Verdict) -> pd.Series:
    """The values the association sees: 0/1 for binary, as measured otherwise."""
    if verdict.kind == "binary":
        return values.map({verdict.low: 0.0, verdict.high: 1.0})
    return values.astype(np.float64)


def find_id_column(names: list[str]) -> str | None:
    return next((c for c in ID_COLUMNS if c in names), None)
