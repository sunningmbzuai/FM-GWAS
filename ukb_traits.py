"""Per-trait label construction for the 8 FM-GWAS traits.

Two kinds of trait:

* Quantitative -- Cholesterol, Height, VAT. Labelled either as the raw measured
  value ("continuous") or as extreme tails, top decile 1 / bottom decile 0 with
  the middle DROPPED ("extreme", the default, matching the "_extreme" framing of
  the HPP gene lists). config.QUANT_LABEL_MODE selects.
* Disease -- Hyperlipidemia, Hypertension, T2D, Obesity, Osteoporosis. Always
  binary, PRESENCE/ABSENCE per the paper's Methods ("1 denotes the presence of
  disease and 0 denotes absence"): a case is the union of hospital ICD-10,
  self-reported illness, touchscreen medication and a guideline measurement
  cutoff, and everyone else with the evidence to be assessed is a control.
  There is no dropped middle -- deliberately, because the paper's cohort sizes
  and label distributions (Fig 1.c) are whole-cohort, not extreme-contrast.

  Two case cutoffs follow the paper rather than the strict clinical definition:
  osteoporosis includes OSTEOPENIA, and T2D includes PREDIABETES.

Every builder returns a Series of labels indexed by eid, restricted to
participants with the evidence needed to be classified at all.
"""
import pandas as pd

import ukb_fields as F


def extreme_tails(values: pd.Series, tail_frac: float) -> pd.Series:
    """Top tail -> 1, bottom tail -> 0, middle dropped."""
    clean = values.dropna()
    if clean.empty:
        return clean
    low, high = clean.quantile(tail_frac), clean.quantile(1.0 - tail_frac)
    if not low < high:
        raise ValueError(
            f"degenerate distribution: {tail_frac:.0%} quantile {low} is not "
            f"below the {1 - tail_frac:.0%} quantile {high}"
        )
    labelled = pd.concat([
        pd.Series(0, index=clean.index[clean <= low]),
        pd.Series(1, index=clean.index[clean >= high]),
    ])
    return labelled.sort_index()


def quantitative(values: pd.Series, mode: str, tail_frac: float) -> pd.Series:
    if mode == "continuous":
        return values.dropna()
    if mode == "extreme":
        return extreme_tails(values, tail_frac)
    raise ValueError(f"unknown QUANT_LABEL_MODE {mode!r}; use 'continuous' or 'extreme'")


def _binary(case: pd.Series, assessable: pd.Series) -> pd.Series:
    """Presence/absence label: 1 where `case`, 0 where assessable and not a case.

    `assessable` is the evidence requirement -- a participant with neither a
    diagnosis record nor the relevant measurement is dropped, because absence of
    evidence is not evidence of absence. Everyone else is classified, so a
    participant who is diagnosed but whose measurement is now normal (a treated
    hypertensive, say) is a case: diagnosis outranks a single measurement.
    """
    label = pd.Series(pd.NA, index=case.index, dtype="Int64")
    label = label.mask(assessable, 0)
    label = label.mask(case, 1)
    return label.dropna().astype(int)


def obesity(bmi: pd.Series) -> pd.Series:
    return _binary(case=bmi >= F.BMI_OBESE, assessable=bmi.notna())


def hyperlipidemia(chol: pd.Series, icd: pd.Series, selfrep: pd.Series,
                   chol_med: pd.Series) -> pd.Series:
    diagnosed = icd | selfrep | chol_med
    return _binary(
        case=diagnosed | (chol >= F.CHOL_HIGH),
        assessable=diagnosed | chol.notna(),
    )


def hypertension(sbp: pd.Series, dbp: pd.Series, icd: pd.Series,
                 selfrep: pd.Series, bp_med: pd.Series) -> pd.Series:
    diagnosed = icd | selfrep | bp_med
    measured = sbp.notna() & dbp.notna()
    return _binary(
        case=diagnosed | (measured & ((sbp >= F.SBP_HIGH) | (dbp >= F.DBP_HIGH))),
        assessable=diagnosed | measured,
    )


def type_2_diabetes(icd_t2d: pd.Series, icd_t1d: pd.Series, selfrep_t2d: pd.Series,
                    selfrep_t1d: pd.Series, selfrep_unspec: pd.Series,
                    doctor_said: pd.Series, insulin: pd.Series,
                    hba1c: pd.Series) -> pd.Series:
    """Glycaemic-status label: prediabetes or any diabetes = 1, non-diabetic = 0.

    Follows the paper ("both prediabetes and diabetes are defined as 1, and
    non-diabetic individuals as 0"), which is why this is broader than clinical
    type-2 diabetes: type-1 and unspecified diabetes, insulin use and a
    prediabetic HbA1c all count as cases. The trait keeps the name T2D to match
    the HPP gene-list filenames.

    A control needs positive evidence of being non-diabetic: an explicit "no" on
    field 2443 or a normal HbA1c. Missing both is not evidence.
    """
    diagnosed = (
        icd_t2d | icd_t1d | selfrep_t2d | selfrep_t1d | selfrep_unspec
        | insulin | (doctor_said == 1)
    )
    return _binary(
        case=diagnosed | (hba1c >= F.HBA1C_PREDIABETES),
        assessable=diagnosed | (doctor_said == 0) | hba1c.notna(),
    )


def osteoporosis(tscore: pd.Series, icd: pd.Series, selfrep: pd.Series) -> pd.Series:
    """Osteopenia OR osteoporosis = 1, normal bone density = 0, per the paper."""
    diagnosed = icd | selfrep
    return _binary(
        case=diagnosed | (tscore <= F.TSCORE_LOW_BONE_MASS),
        assessable=diagnosed | tscore.notna(),
    )
