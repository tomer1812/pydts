import pandas as pd
import numpy as np

DEFAULT_N_PATIENTS = 10000

min_year = 2000
max_year = 2015

bmi_coef = 0.003
age_coef = 0.002
gender_coef = -0.15
smk_coef = 0.1
max_p = 0.65

PATIENT_NO_COL = 'ID'
AGE_COL = 'Age'
GENDER_COL = 'Sex'
ADMISSION_YEAR_COL = 'Admyear'
FIRST_ADMISSION_COL = 'Firstadm'
ADMISSION_SERIAL_COL = 'Admserial'
WEIGHT_COL = 'Weight'
HEIGHT_COL = 'Height'
BMI_COL = 'BMI'
SMOKING_COL = 'Smoking'
HYPERTENSION_COL = 'Hypertension'
DIABETES_COL = 'Diabetes'
ART_FIB_COL = 'AF'  # Arterial Fibrillation
COPD_COL = 'COPD'  # Chronic Obstructive Pulmonary Disease
CRF_COL = 'CRF'  # Chronic Renal Failure
IN_HOSPITAL_DEATH_COL = 'In_hospital_death'
DISCHARGE_RELATIVE_COL = 'Discharge_relative_date'
DEATH_RELATIVE_COL = 'Death_relative_date_in_hosp'
DEATH_MISSING_COL = 'Death_date_in_hosp_missing'
RETURNING_PATIENT_COL = 'Returning_patient'

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

DEFAULT_REAL_COEF_DICT = {
    "alpha": {
        1: lambda t: -1 - 0.3 * np.log(t),
        2: lambda t: -1.75 - 0.15 * np.log(t)
    },
    "beta": {
        1: -np.log([0.8, 3, 3, 2.5, 2]),
        2: -np.log([1, 3, 4, 3, 2])
    }
}
