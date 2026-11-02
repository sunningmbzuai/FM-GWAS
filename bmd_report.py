"""Bone-density sources for Osteoporosis, for Ning's 2026-11-02 questions.

Ning asked which skeletal site the UKB T-scores come from, and whether raw
total femur BMD (she named fields 23291 and 23212) is available, so the HPP
cutoff (min left/right total femur BMD < 0.88 g/cm2) can be applied in UKB.

For QC-passed participants this prints and writes, under
OUT_ROOT/bmd_report/:

    bmd_columns.tsv   every femur/BMD column of the DXA export used or
                      considered: n measured, median, % below 0.88 (BMD) or
                      % <= -1.0 (T-score)
    bmd_summary.txt   the chosen total femur BMD columns, how many
                      participants have them, how many fall below 0.88, how
                      that agrees with the T-score rule used so far, and
                      whether Ning's raw field ids exist in the main dump

Run with the microbiome env, like build_cohorts.py:
    python bmd_report.py
"""
import os

import pandas as pd

import build_cohorts as B
import config
import ukb_fields as F
import ukb_source as S


def column_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per column: n measured, median, and the low-bone fraction."""
    rows = []
    for name in frame.columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        is_tscore = "t_score" in name
        if not is_tscore:
            values = values.replace(0.0, float("nan"))   # 0 = missing for BMD
        measured = values.dropna()
        low = (measured <= F.TSCORE_LOW_BONE_MASS if is_tscore
               else measured < F.FEMUR_BMD_LOW)
        rows.append({
            "column": name,
            "kind": "T-score" if is_tscore else "BMD g/cm2",
            "n_measured": len(measured),
            "median": measured.median() if len(measured) else float("nan"),
            "pct_low": 100 * low.mean() if len(measured) else float("nan"),
            "low_rule": ("T <= -1.0" if is_tscore
                         else f"BMD < {F.FEMUR_BMD_LOW}"),
        })
    return pd.DataFrame(rows)


def raw_field_columns(fields: tuple[str, ...]) -> dict[str, list[str]]:
    """Instance-2 columns of each raw UKB field id in the main dump."""
    names = S.column_names(config.UKB_MAIN_PARQUET)
    return {f: [n for n in names if n.startswith(f"{f}-2.")] for f in fields}


def agreement(femur_min: pd.Series, tscore_min: pd.Series) -> pd.DataFrame:
    """2x2 of the new femur BMD rule against the old lowest-T-score rule."""
    both = pd.DataFrame({"femur": femur_min, "tscore": tscore_min}).dropna()
    return pd.crosstab(
        both["femur"] < F.FEMUR_BMD_LOW,
        both["tscore"] <= F.TSCORE_LOW_BONE_MASS,
        rownames=[f"femur BMD < {F.FEMUR_BMD_LOW}"],
        colnames=["lowest T-score <= -1.0"],
    )


def main() -> None:
    out_dir = os.path.join(config.OUT_ROOT, "bmd_report")
    os.makedirs(out_dir, exist_ok=True)
    qc_ids = B.load_qc_ids()

    names = S.column_names(config.UKB_DXA_PARQUET)
    left, right = S.femur_bmd_columns(names)
    femur_all = sorted(n for n in names
                       if "femur" in n.lower() and n.endswith(F.DXA_INSTANCE_SUFFIX))
    wanted = list(dict.fromkeys(
        [left, right, *F.EXPORT_BMD_TSCORE_SITES, *femur_all]))
    dxa = S.read_columns(config.UKB_DXA_PARQUET, wanted)
    dxa = dxa[dxa.index.isin(qc_ids)]

    stats = column_stats(dxa)
    stats.to_csv(os.path.join(out_dir, "bmd_columns.tsv"), sep="\t",
                 index=False, float_format="%.3f")

    bmd = S.mask_zeros_as_missing(dxa, [left, right])
    femur_min = S.lowest_across_sites(bmd, [left, right])
    tscore_min = S.lowest_across_sites(dxa, F.EXPORT_BMD_TSCORE_SITES)
    measured = femur_min.dropna()
    raw = raw_field_columns(F.RAW_FEMUR_BMD_FIELDS)

    lines = [
        f"QC-passed participants: {len(qc_ids):,}",
        "",
        "T-scores used so far (lowest of the measured sites):",
        *(f"  {c}" for c in F.EXPORT_BMD_TSCORE_SITES),
        f"  measured: {tscore_min.notna().sum():,}",
        "",
        f"Total femur BMD columns: {left}, {right}",
        f"  measured (either side): {len(measured):,}",
        f"  both sides: {int(bmd[[left, right]].notna().all(axis=1).sum()):,}",
        f"  min(L, R) < {F.FEMUR_BMD_LOW}: {int((measured < F.FEMUR_BMD_LOW).sum()):,}"
        f" ({100 * (measured < F.FEMUR_BMD_LOW).mean():.1f}%)",
        f"  median min(L, R): {measured.median():.3f} g/cm2",
        "",
        "Agreement with the T-score rule (participants with both):",
        agreement(femur_min, tscore_min).to_string(),
        "",
        "Ning's raw field ids in the main dump (instance 2):",
        *(f"  {f}: {cols or 'not present'}" for f, cols in raw.items()),
    ]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "bmd_summary.txt"), "w",
              encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(stats.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()
    print(text)
    print(f"\nWrote bmd_columns.tsv and bmd_summary.txt to {out_dir}")


if __name__ == "__main__":
    main()
