"""Disease labels aligned to the HPP curated phenotypes (Ning, 2026-11-02).

Ning compared the HPP case/control definitions with ours and mapped each one
onto UK Biobank sources (UKBB_case_control.csv):

    trait           HPP case / control                  UKB case / control
    Hypertension    hypertension / non-hypertensive     ICD-10 I10-I15, self-report / everyone else
    Hyperlipidemia  hyperlipidemia / non-               ICD-10 E78, self-report / everyone else
    Obesity         obesity + severe / normal           BMI > 30 / 18.5 < BMI < 24.9
    Osteoporosis    osteoporosis + osteopenia / normal  ICD-10 M80-M81, self-report,
                                                        min total femur BMD < 0.88 / normal BMD
    T2D             diabetes + prediabetes / none       ICD-10 E11, self-report, field 2443 / everyone else

Differences from the paper reading in ukb_traits.py: medication and the
blood-pressure, cholesterol and HbA1c cutoffs no longer make a case;
controls no longer need a measurement; overweight and underweight
participants are dropped from Obesity; Osteoporosis uses total femur BMD in
place of the lowest T-score.

Osteoporosis has two options, since the BMD threshold is still being
confirmed: "bmd" (with the femur BMD cutoff, controls with a measured normal
BMD) and "diagnosis" (ICD-10 and self-report only, everyone else a control).

Every labeller returns 0/1 labels indexed by eid, over the participants it can
classify.
"""
import pandas as pd

import ukb_fields as F


def _case_or_rest(case: pd.Series) -> pd.Series:
    """1 for cases, 0 for every other participant in the index."""
    return case.fillna(False).astype(bool).astype(int)


def hypertension(icd: pd.Series, selfrep: pd.Series) -> pd.Series:
    return _case_or_rest(icd | selfrep)


def hyperlipidemia(icd: pd.Series, selfrep: pd.Series) -> pd.Series:
    return _case_or_rest(icd | selfrep)


def type_2_diabetes(icd_t2d: pd.Series, selfrep_any: pd.Series,
                    doctor_said: pd.Series) -> pd.Series:
    return _case_or_rest(icd_t2d | selfrep_any | (doctor_said == 1))


def obesity(bmi: pd.Series) -> pd.Series:
    """BMI > 30 -> 1, 18.5 < BMI < 24.9 -> 0, everyone else dropped."""
    low, high = F.BMI_HPP_NORMAL
    case = bmi > F.BMI_HPP_OBESE
    control = (bmi > low) & (bmi < high)
    return pd.concat([
        pd.Series(1, index=bmi.index[case]),
        pd.Series(0, index=bmi.index[control]),
    ]).sort_index().astype(int)


def osteoporosis(femur_bmd: pd.Series, icd: pd.Series, selfrep: pd.Series,
                 option: str) -> pd.Series:
    """Ning's option 1 ("bmd") or option 2 ("diagnosis").

    bmd: a case is diagnosed or has min total femur BMD < 0.88 g/cm2; a
    control is undiagnosed with a measured BMD >= 0.88, matching HPP's
    "normal BMD" class. Undiagnosed participants without a DXA scan are
    dropped.
    diagnosis: a case is diagnosed; everyone else is a control.
    """
    diagnosed = (icd | selfrep).fillna(False).astype(bool)
    if option == "diagnosis":
        return _case_or_rest(diagnosed)
    if option != "bmd":
        raise ValueError(f"unknown osteoporosis option {option!r}; "
                         f"use 'bmd' or 'diagnosis'")
    low_bmd = femur_bmd < F.FEMUR_BMD_LOW
    case = diagnosed | low_bmd
    control = ~case & femur_bmd.notna()
    return pd.concat([
        pd.Series(1, index=femur_bmd.index[case]),
        pd.Series(0, index=femur_bmd.index[control]),
    ]).sort_index().astype(int)
