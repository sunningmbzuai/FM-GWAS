"""UKB field ids, codings and case/control code sets for the 8 FM-GWAS traits.

Separated from build_cohorts.py so the phenotype DEFINITIONS are reviewable on
their own -- they are the part of this pipeline a reader will want to argue
with, and they are what has to match the HPP side of the replication.

Column naming in the raw dump (ukb676772.parquet) is "<field>-<instance>.<array>":
instance 0 = baseline, 1 = repeat, 2 = first imaging visit, 3 = repeat imaging.
The phenotypes export uses "<biomarker>_i<instance>" instead.
"""

# ---- raw-dump single columns ------------------------------------------------
EID = "eid"
SEX = "31-0.0"            # coding 9: 0 Female, 1 Male -- already FM-GWAS's encoding
AGE = "21022-0.0"         # age at recruitment
HEIGHT = "50-0.0"         # standing height, cm, baseline
BMI = "21001-0.0"         # BMI, baseline (far larger N than the DXA visit-2 BMI)
DIABETES_SELFREP = "2443-0.0"   # coding 100349: 1 Yes, 0 No, -1 Do not know, -3 Prefer not
HBA1C = "30750-0.0"       # glycated haemoglobin, mmol/mol, baseline

# Blood pressure: two automated readings per visit, with the manual sphygmo-
# manometer (93/94) as fallback for participants the automated device failed on.
SBP_AUTO = ("4080-0.0", "4080-0.1")
DBP_AUTO = ("4079-0.0", "4079-0.1")
SBP_MANUAL = ("93-0.0", "93-0.1")
DBP_MANUAL = ("94-0.0", "94-0.1")

# ---- raw-dump multi-array field prefixes (matched as "<prefix>-0." ) -------
ICD10_ALL = "41270"        # all hospital-inpatient ICD-10 diagnoses, array-only
SELFREP_ILLNESS = "20002"  # self-reported non-cancer illness, coding 6
MEDS_MEN = "6177"          # coding 100626
MEDS_WOMEN = "6153"        # coding 100625 (superset: adds HRT / oral contraceptive)

# 6153/6177 shared codes. Women's coding adds 4/5 (HRT, contraceptive) which we
# never look at, so the two can be tested with the same code set.
MED_CHOLESTEROL = 1
MED_BLOOD_PRESSURE = 2
MED_INSULIN = 3

# ---- disease definitions ----------------------------------------------------
# ICD-10 values are strings like "E119"; matched by prefix.
ICD_HYPERLIPIDEMIA = ("E78",)                      # disorders of lipoprotein metabolism
ICD_HYPERTENSION = ("I10", "I11", "I12", "I13", "I15")  # essential + hypertensive organ disease
ICD_T2D = ("E11",)                                 # non-insulin-dependent DM
ICD_T1D = ("E10",)                                 # excluded from T2D controls AND cases
ICD_OSTEOPOROSIS = ("M80", "M81")                  # with / without pathological fracture

# Self-reported illness codes (coding 6).
SELFREP_HIGH_CHOLESTEROL = 1473
SELFREP_HYPERTENSION = (1065, 1072)   # essential hypertension, high blood pressure
SELFREP_T2D = 1223
SELFREP_T1D = 1222
SELFREP_DIABETES_UNSPEC = 1220
SELFREP_OSTEOPOROSIS = 1309

# ---- quantitative thresholds ------------------------------------------------
# Binary traits are PRESENCE/ABSENCE, per the paper's Methods: "1 denotes the
# presence of disease and 0 denotes absence". There is no dropped middle -- a
# participant with the measurement is always classifiable -- so each trait has
# one cutoff, not a case cutoff plus a stricter control cutoff.
BMI_OBESE = 30.0             # WHO obesity
SBP_HIGH, DBP_HIGH = 140.0, 90.0   # stage-1 hypertension
CHOL_HIGH = 6.2              # mmol/L, ~240 mg/dL, "high" total cholesterol

# "For osteoporosis, both osteoporosis and osteopenia are defined as 1, and
# normal as 0" -- so the case cutoff is the OSTEOPENIA one (WHO: normal is
# T > -1.0), not the osteoporosis one. TSCORE_OSTEOPOROSIS is unused by the
# labeller and kept only to document what the stricter cutoff would be.
TSCORE_LOW_BONE_MASS = -1.0  # osteopenia or worse => case
TSCORE_OSTEOPOROSIS = -2.5   # WHO osteoporosis proper (not the case cutoff)

# "for type II diabetes (T2D), both prediabetes and diabetes are defined as 1,
# and non-diabetic individuals as 0" -- so the case cutoff is the PREDIABETES
# one. ADA: HbA1c 5.7% = 39 mmol/mol prediabetes, 6.5% = 48 diabetes.
HBA1C_PREDIABETES = 39.0     # mmol/mol, prediabetes or worse => case
HBA1C_DIABETES = 48.0        # documents the diabetes threshold; not the cutoff

# Fraction per tail for the non-default "extreme" quantitative label mode.
EXTREME_TAIL_FRAC = 0.10

# ---- phenotypes-export columns ---------------------------------------------
EXPORT_CHOLESTEROL = "cholesterol_i0"
EXPORT_VAT_VOLUME = "vat_volume_i2"

# WHO osteopenia/osteoporosis T-score cutoffs are defined at the femoral neck
# and lumbar spine, never whole-body: total_bmd_t_score_i2 runs systematically
# higher and understates case prevalence. Clinical practice classifies on the
# LOWEST of the measured sites, which is what the labeller uses.
EXPORT_BMD_TSCORE_SITES = (
    "femur_neck_bmd_t_score_left_i2",
    "femur_neck_bmd_t_score_right_i2",
    "l1_l4_bmd_t_score_i2",
)

# DXA columns where an exact 0 is a missingness placeholder, not a measurement:
# a visceral fat volume of 0 is physically impossible. Flagged by the export's
# own README ("mask zeros to NaN in BMD/BMC/mass/area columns before
# modelling") and by its dictionaries' n_exact_zero column. NOT applied to
# T-scores, where 0 means "at the young-adult reference mean" and is real.
DXA_ZERO_IS_MISSING = (EXPORT_VAT_VOLUME,)
