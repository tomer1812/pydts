import numpy as np
from pydts.examples_utils.simulations_data_config import *
from pydts.config import *
import pandas as pd
from scipy.special import expit
from pandarallel import pandarallel


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
    #admyear = row[ADMISSION_YEAR_COL]
    p_weight = 0.3 #+ int(admyear > (min_year + 3)) * 0.8 * ((admyear - min_year) / (max_year - min_year))
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
        }

        simulated_patients_df = simulated_patients_df.append(new_patient, ignore_index=True)

    # [age, gender, ADMISSION_YEAR, FIRST_ADMISSION, ADMISSION_SERIAL, WEIGHT_COL, HEIGHT_COL, BMI, SMOKING_COL, 5*preconditions]
    #b1 = [-0.003, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    b2 = [0.003, -0.05, 0, 0, 0, 0.003, 0.0001, 0.005, 0.05, 0.02, 0.02, 0.02, 0.02, 0.02]

    real_coef_dict = {
        "alpha": {
            1: lambda t: -2.2 - 0.2 * np.log(t),
            2: lambda t: -2.3 - 0.2 * np.log(t)
        },
        "beta": {
            1: [-1*b for b in b2],
            2: [b for b in b2]
        }
    }

    # Sample event type and relative time
    events_df = new_sample_logic(simulated_patients_df.set_index(PATIENT_NO_COL), j_events=2, d_times=30,
                                             real_coef_dict=real_coef_dict)
    simulated_patients_df = pd.concat([simulated_patients_df, events_df], axis=1)

    simulated_patients_df[DEATH_RELATIVE_COL] = np.nan
    simulated_patients_df.loc[simulated_patients_df.J == 1, DEATH_RELATIVE_COL] = simulated_patients_df.loc[
        simulated_patients_df.J == 1, 'T'].values
    simulated_patients_df = simulated_patients_df.rename(columns={'T': DISCHARGE_RELATIVE_COL})
    simulated_patients_df.loc[simulated_patients_df.J == 0, DISCHARGE_RELATIVE_COL] = 31
    simulated_patients_df = simulated_patients_df.drop('J', axis=1)

    # Remove weight and bmi based on admission year
    simulated_patients_df = simulated_patients_df.apply(hide_weight_info, axis=1)

    simulated_patients_df[DEATH_MISSING_COL] = simulated_patients_df[DEATH_RELATIVE_COL].isnull().astype(int)
    simulated_patients_df[IN_HOSPITAL_DEATH_COL] = simulated_patients_df[DEATH_RELATIVE_COL].notnull().astype(int)
    simulated_patients_df[RETURNING_PATIENT_COL] = pd.cut(simulated_patients_df[ADMISSION_SERIAL_COL],
                                                        bins=ADMISSION_SERIAL_BINS, labels=ADMISSION_SERIAL_LABELS)

    simulated_patients_df.set_index(PATIENT_NO_COL).to_csv(os.path.join(output_dir, filename))


def default_sampling_logic(Z, d_times):
    alpha1t = -1 -0.3*np.log(np.arange(start=1, stop=d_times+1))
    beta1 = -np.log([0.8, 3, 3, 2.5, 2])
    alpha2t = -1.75 -0.15*np.log(np.arange(start=1, stop=d_times+1))
    beta2 = -np.log([1, 3, 4, 3, 2])
    hazard1 = expit(alpha1t+(Z*beta1).sum())
    hazard2 = expit(alpha2t+(Z*beta2).sum())
    surv_func = np.array([1, *np.cumprod(1-hazard1-hazard2)[:-1]])
    proba1 = hazard1*surv_func
    proba2 = hazard2*surv_func
    sum1 = np.sum(proba1)
    sum2 = np.sum(proba2)
    probj1t = proba1 / sum1
    probj2t = proba2 / sum2
    j_i = np.random.choice(a=[0, 1, 2], p=[1-sum1-sum2, sum1, sum2])
    if j_i == 0:
        T_i = d_times
    elif j_i == 1:
        T_i = np.random.choice(a=np.arange(1, d_times+1), p=probj1t)
    else:
        T_i = np.random.choice(a=np.arange(1, d_times+1), p=probj2t)
    return j_i, T_i


def calculate_jt(sums, probs_jt, d_times, j_events):
    events = range(1, j_events + 1)
    temp_sums = pd.concat([1 - sum(sums), *sums], axis=1, keys=[0, *events])

    j_df = (temp_sums.cumsum(1) > np.random.rand(temp_sums.shape[0])[:, None]).idxmax(axis=1).to_frame('J')

    temp_ts = []
    for j in events:
        rel_j = j_df.query("J==@j").index
        prob_df = probs_jt[j - 1]  # the prob j to sample from
        # sample T
        temp_ts.append((prob_df.loc[rel_j].cumsum(1) >= np.random.rand(rel_j.shape[0])[:, None]).idxmax(axis=1))

    temp_ts.append(pd.Series(d_times+1, index=j_df.query('J==0').index))

    j_df["T"] = pd.concat(temp_ts).sort_index()
    return j_df


def new_sample_logic(patients_df: pd.DataFrame, j_events: int, d_times: int, real_coef_dict: dict) -> pd.DataFrame:
    """
    A quicker sample logic, that uses coefficients supplied by the user
    """
    events = range(1, j_events + 1)
    a_t = {event: {t: real_coef_dict['alpha'][event](t) for t in range(1, d_times+1)} for event in events}
    b = pd.concat([patients_df.dot(real_coef_dict['beta'][j]) for j in events], axis=1, keys=events)

    hazards = [pd.concat([expit(a_t[j][t] + b[j]) for t in range(1, d_times + 1)],
                         axis=1, keys=(range(1, d_times + 1))) for j in events]
    surv_func = pd.concat([pd.Series(1, index=hazards[0].index),
                           (1 - sum(hazards)).cumprod(axis=1).iloc[:, :-1]], axis=1)
    surv_func.columns += 1

    probs = [hazard * surv_func for hazard in hazards]
    sums = [prob.sum(axis=1) for prob in probs]
    probjt = [prob.div(sumj, axis=0) for prob, sumj in zip(probs, sums)]

    ret = calculate_jt(sums, probjt, d_times, j_events)
    return ret


def generate_quick_start_df(n_patients=10000, d_times=30, j_events=2, n_cov=5, seed=0, pid_col='pid',
                            real_coef_dict: dict = DEFAULT_REAL_COEF_DICT, sampling_logic=new_sample_logic,
                            censoring_prob=1.):
    np.random.seed(seed)
    assert real_coef_dict is not None, "The user should supply the coefficients of the experiment"
    covariates = [f'Z{i + 1}' for i in range(n_cov)]
    patients_df = pd.DataFrame(data=np.random.uniform(low=0.0, high=1.0, size=[n_patients, n_cov]),
                               columns=covariates)
    sampled = sampling_logic(patients_df, j_events, d_times, real_coef_dict)
    patients_df = pd.concat([patients_df, sampled], axis=1)
    patients_df.index.name = pid_col
    patients_df['C'] = np.where(np.random.rand(n_patients) < censoring_prob,
                                np.random.randint(low=1, high=d_times+1,
                                                  size=n_patients), d_times+1)
    patients_df['X'] = patients_df[['T', 'C']].min(axis=1)
    patients_df.loc[patients_df['C'] < patients_df['T'], 'J'] = 0
    return patients_df.reset_index()


if __name__ == "__main__":
    main(N_patients=50000)
    #generate_quick_start_df(n_patients=2)
