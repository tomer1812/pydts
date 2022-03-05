import numpy as np
from examples.simulations_data_config import *
from config import *
import pandas as pd


def sample_los(new_patient, age_mean, age_std, bmi_mean, bmi_std, coefs=COEFS, baseline_hazard_scale=8,
               los_bounds=[1, 150]):
    # Columns normalization:
    new_patient[AGE_COL] = (new_patient[AGE_COL] - age_mean) / age_std
    new_patient[GENDER_COL] = 2 * (new_patient[GENDER_COL] - 0.5)
    new_patient[BMI_COL] = (new_patient[BMI_COL] - bmi_mean) / bmi_std
    new_patient[SMOKING_COL] = new_patient[SMOKING_COL] - 1
    new_patient[HYPERTENSION_COL] = 2 * (new_patient[HYPERTENSION_COL] - 0.5)
    new_patient[DIABETES_COL] = 2 * (new_patient[DIABETES_COL] - 0.5)
    new_patient[ART_FIB_COL] = 2 * (new_patient[ART_FIB_COL] - 0.5)
    new_patient[COPD_COL] = 2 * (new_patient[COPD_COL] - 0.5)
    new_patient[CRF_COL] = 2 * (new_patient[CRF_COL] - 0.5)
    new_patient = pd.Series(new_patient)

    # Baseline hazard
    baseline_hazard = np.random.exponential(scale=baseline_hazard_scale)

    # Patient's correction
    beta_x = coefs.dot(new_patient[coefs.index])

    # Sample, round (for ties), and clip to bounds the patient's length of stay at the hospital
    los = np.clip(np.round(baseline_hazard * np.exp(beta_x)), a_min=los_bounds[0], a_max=los_bounds[1])
    los_death = np.nan if new_patient[IN_HOSPITAL_DEATH_COL] == 0 else los
    return los, los_death


def hide_weight_info(row):
    admyear = row[ADMISSION_YEAR_COL]
    p_weight = 0.1 + int(admyear > (min_year + 3)) * 0.8 * ((admyear - min_year) / (max_year - min_year))
    sample_weight = np.random.binomial(1, p=p_weight)
    if sample_weight == 0:
        row[WEIGHT_COL] = np.nan
        row[BMI_COL] = np.nan
    return row


def main(seed=0, N_patients=DEFAULT_N_PATIENTS, output_dir=OUTPUT_DIR, filename=SIMULATED_DATA_FILENAME):
    # Set random seed for consistent sampling
    np.random.seed(seed)

    # Female - 1, Male - 0
    gender = np.random.binomial(n=1, p=0.5, size=N_patients)

    simulated_patients_df = pd.DataFrame()

    for p in range(N_patients):
        # Sample gender dependent age for each patient
        age = np.round(np.random.normal(loc=72 + 5 * gender[p], scale=12), decimals=1)

        # Random sample admission year
        admyear = np.random.randint(low=min_year, high=max_year)

        # Sample gender dependent height
        height = np.random.normal(loc=175 - 5 * gender[p], scale=7)

        # Sample height, gender and age dependent weight
        weight = np.random.normal(loc=(height / 175) * 80 - 5 * gender[p] + (age / 20), scale=8)

        # Calculate body mass index (BMI) from weight and height
        bmi = weight / ((height / 100) ** 2)

        # Random sample of previous admissions
        admserial = np.clip(np.round(np.random.lognormal(mean=0, sigma=0.75)), 1, 20)

        # Random sample of categorical smoking status: No - 0, Previously - 1, Currently - 2
        smoking = np.random.choice([0, 1, 2], p=[0.5, 0.3, 0.2])

        # Sample patient's preconditions based on gender, age, BMI, and smoking status with limits on the value of p
        pre_p = np.clip((bmi_coef * bmi + gender_coef * gender[p] + age_coef * age + smk_coef * smoking),
                        a_min=0.05, a_max=max_p)
        hypertension = np.random.binomial(n=1, p=pre_p)
        diabetes = np.random.binomial(n=1, p=pre_p + bmi_coef * bmi)
        artfib = np.random.binomial(n=1, p=pre_p)  # Arterial Fibrillation
        copd = np.random.binomial(n=1, p=pre_p + smk_coef * smoking)  # Chronic Obstructive Pulmonary Disease
        crf = np.random.binomial(n=1, p=pre_p)  # Chronic Renal Failure

        # Sample outcome - in-hospital death based on gender, age, BMI, smoking status, and preconditions with limits
        # on the value of p
        dp = np.clip(0.25 * pre_p + 0.1 * (hypertension + diabetes + artfib + copd + crf),
                     a_min=0.05, a_max=0.35)
        inhospital_death = np.random.binomial(n=1, p=dp)

        new_patient = {
            PATIENT_NO_COL: p,
            AGE_COL: age,
            GENDER_COL: gender[p],
            ADMISSION_YEAR_COL: int(admyear),
            FIRST_ADMISSION_COL: int(admserial == 1),
            ADMISSION_SERIAL_COL: int(admserial),
            WEIGHT_COL: weight,
            HEIGHT_COL: height,
            BMI_COL: bmi,
            SMOKING_COL: smoking,
            HYPERTENSION_COL: hypertension,
            DIABETES_COL: diabetes,
            ART_FIB_COL: artfib,
            COPD_COL: copd,
            CRF_COL: crf,
            IN_HOSPITAL_DEATH_COL: inhospital_death
        }

        simulated_patients_df = simulated_patients_df.append(new_patient, ignore_index=True)

    age_mean = simulated_patients_df[AGE_COL].mean()
    age_std = simulated_patients_df[AGE_COL].std()
    bmi_mean = simulated_patients_df[BMI_COL].mean()
    bmi_std = simulated_patients_df[BMI_COL].std()

    # Sample length of stay
    tmp_df = simulated_patients_df.copy()
    simulated_patients_df[[DISCHARGE_RELATIVE_COL, DEATH_RELATIVE_COL]] = tmp_df.apply(sample_los,
        age_mean=age_mean, age_std=age_std,  bmi_mean=bmi_mean, bmi_std=bmi_std, axis=1, result_type='expand')
    del tmp_df

    # Remove weight and bmi based on admission year
    simulated_patients_df = simulated_patients_df.apply(hide_weight_info, axis=1)

    simulated_patients_df[DEATH_MISSING_COL] = simulated_patients_df[DEATH_RELATIVE_COL].isnull().astype(int)
    simulated_patients_df[RETURNING_PATIENT_COL] = pd.cut(simulated_patients_df[ADMISSION_SERIAL_COL],
                                                        bins=ADMISSION_SERIAL_BINS, labels=ADMISSION_SERIAL_LABELS)

    simulated_patients_df.set_index(PATIENT_NO_COL).to_csv(os.path.join(output_dir, filename))


def default_sample_T(patients_df, d_times):
    patients_df['T'] = np.clip(np.round(d_times * (patients_df['Z1'])), a_min=1, a_max=d_times+1)
    return patients_df

def default_sample_C(patients_df, d_times, n_patients):
    patients_df['C'] = np.random.randint(low=1, high=d_times + 1, size=n_patients)
    return patients_df

def default_sample_J(patients_df, j_events, n_patients):
    patients_df['J'] = np.random.randint(low=1, high=1 + j_events, size=n_patients)
    return patients_df

def generate_quick_start_df(n_patients=10000, d_times=150, j_events=2, n_cov=5, seed=0, pid_col='pid',
                            sample_T=default_sample_T, sample_C=default_sample_C, sample_J=default_sample_J):
    np.random.seed(seed)
    covariates = [f'Z{i + 1}' for i in range(n_cov)]
    patients_df = pd.DataFrame(data=np.random.uniform(low=0.0, high=1.0, size=[n_patients, n_cov]),
                               columns=covariates)
    patients_df.index.name = pid_col
    patients_df = sample_T(patients_df, d_times)
    patients_df = sample_C(patients_df, d_times, n_patients)
    patients_df['X'] = patients_df[['T', 'C']].min(axis=1)
    patients_df = sample_J(patients_df, j_events, n_patients)
    patients_df.loc[patients_df['C'] < patients_df['T'], 'J'] = 0
    return patients_df.reset_index()


if __name__ == "__main__":
    main()
