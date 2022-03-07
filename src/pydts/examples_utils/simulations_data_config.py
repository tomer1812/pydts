import pandas as pd

DEFAULT_N_PATIENTS = 10000

min_year = 2000
max_year = 2015

bmi_coef = 0.003
age_coef = 0.002
gender_coef = -0.15
smk_coef = 0.1
max_p = 0.65

PATIENT_NO_COL = 'ptno'
AGE_COL = 'age'
GENDER_COL = 'sex'
ADMISSION_YEAR_COL = 'admyear'
FIRST_ADMISSION_COL = 'firstadm'
ADMISSION_SERIAL_COL = 'admserial'
WEIGHT_COL = 'Weight'
HEIGHT_COL = 'Height'
BMI_COL = 'bmi'
SMOKING_COL = 'smk'
HYPERTENSION_COL = 'htn'
DIABETES_COL = 'dm'
ART_FIB_COL = 'af'  # Arterial Fibrillation
COPD_COL = 'copd'  # Chronic Obstructive Pulmonary Disease
CRF_COL = 'crf'  # Chronic Renal Failure
IN_HOSPITAL_DEATH_COL = 'inhospital_death'
DISCHARGE_RELATIVE_COL = 'dischargerelative_date'
DEATH_RELATIVE_COL = 'death_relative_date_in_hosp'
DEATH_MISSING_COL = 'death_date_in_hosp_missing'
RETURNING_PATIENT_COL = 'returning_patient'

COEFS = pd.Series({
        AGE_COL: 0.1,
        GENDER_COL: -0.1,
        BMI_COL: 0.2,
        SMOKING_COL: 0.2,
        HYPERTENSION_COL: 0.2,
        DIABETES_COL: 0.2,
        ART_FIB_COL: 0.2,
        COPD_COL: 0.2,
        CRF_COL: 0.2,
})

ADMISSION_SERIAL_BINS = [0, 1.5, 4.5, 8.5, 21]
ADMISSION_SERIAL_LABELS = [0, 1, 2, 3]
SIMULATED_DATA_FILENAME = 'simulated_data.csv'

preconditions = [SMOKING_COL, HYPERTENSION_COL, DIABETES_COL, ART_FIB_COL, COPD_COL, CRF_COL]
font_sz = 14
title_sz = 18
AGE_BINS = list(range(0, 125, 5))
AGE_LABELS = [f'{AGE_BINS[a]}' for a in range(len(AGE_BINS)-1)]
