from pydts.config import *
from pydts.examples_utils.mimic_consts import *
from pydts.evaluation import *
from .mimic_consts import *
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import importlib
import subprocess
import sys

slicer = pd.IndexSlice


DATASETS_DIR = os.path.join(os.path.dirname((os.path.dirname(__file__))), 'datasets')
slicer = pd.IndexSlice


def ensure_package(pkg_name):
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        print(f"{pkg_name} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])


def load_LOS_simulated_data():
    os.path.join(os.path.dirname(__file__))
    return pd.read_csv(os.path.join(DATASETS_DIR, 'LOS_simulated_data.csv'))


def get_mimic_df(mimic_data_dir, return_table1=False, chunksize=10 ** 6, minimum_year=2014):
    """
    Generate an example DataFrame from MIMIC-IV v2.0 files.

    Args:
        mimic_data_dir (str): Path to the directory containing the MIMIC-IV v2.0 data files.
        print_table1_latex (bool, optional): If True, print a LaTeX-formatted version of Table 1.
            Defaults to False.
        chunksize (int, optional): Number of rows per chunk when iterating through large files.
            Reduce this value if system memory is limited. Defaults to 1,000,000.
        minimum_year (int, optional): Minimum year of admission to include in the dataset. Defaults to 2014.

    Returns:
        pd.DataFrame: A sample dataset extracted from the MIMIC-IV v2.0 files.
    """

    patients_file = os.path.join(mimic_data_dir, 'hosp', 'patients.csv.gz')
    admissions_file = os.path.join(mimic_data_dir, 'hosp', 'admissions.csv.gz')
    lab_file = os.path.join(mimic_data_dir, 'hosp', 'labevents.csv.gz')
    lab_meta_file = os.path.join(mimic_data_dir, 'hosp', 'd_labitems.csv.gz')

    patients_df = pd.read_csv(patients_file, compression='gzip')
    COLUMNS_TO_DROP = ['dod']
    patients_df.drop(COLUMNS_TO_DROP, axis=1, inplace=True)

    admissions_df = pd.read_csv(admissions_file, compression='gzip', parse_dates=[ADMISSION_TIME_COL,
                                                                                  DISCHARGE_TIME_COL,
                                                                                  DEATH_TIME_COL,
                                                                                  ED_REG_TIME, ED_OUT_TIME])

    COLUMNS_TO_DROP = ['hospital_expire_flag', 'edouttime', 'edregtime', 'deathtime', 'language']
    admissions_df.drop(COLUMNS_TO_DROP, axis=1, inplace=True)

    admissions_df = admissions_df.merge(patients_df, on=[SUBJECT_ID_COL])

    # # Calculate Age at Admission and Group of Admission Year
    # Based on mimic IV example https://mimic.mit.edu/docs/iv/modules/hosp/patients/

    # Diff column first
    admissions_df[ADMISSION_YEAR_COL] = (admissions_df[ADMISSION_TIME_COL].dt.year - admissions_df['anchor_year'])

    # Age at admission calculation
    admissions_df[ADMISSION_AGE_COL] = (admissions_df[AGE_COL] + admissions_df[ADMISSION_YEAR_COL])

    # Admission year group lower bound calculation
    admissions_df[ADMISSION_YEAR_COL] = admissions_df[ADMISSION_YEAR_COL] + admissions_df[YEAR_GROUP_COL].apply(
        lambda x: int(x.split(' ')[0]))

    # # Calculating LOS (exact, days resolution) and night admission indicator
    admissions_df[LOS_EXACT_COL] = (admissions_df[DISCHARGE_TIME_COL] - admissions_df[ADMISSION_TIME_COL])
    admissions_df[NIGHT_ADMISSION_FLAG] = ((admissions_df[ADMISSION_TIME_COL].dt.hour >= 20) | \
                                           (admissions_df[ADMISSION_TIME_COL].dt.hour < 8)).values
    admissions_df[LOS_DAYS_COL] = admissions_df[LOS_EXACT_COL].dt.ceil('1d')

    max_clip_days = 28

    # # Taking only SPECIFIC_ADMISSION_TYPE admissions from now on
    SPECIFIC_ADMISSION_TYPE = ['DIRECT EMER.', 'EW EMER.']
    admissions_df = admissions_df[admissions_df[ADMISSION_TYPE_COL].isin(SPECIFIC_ADMISSION_TYPE)]

    # # Add direct emergency flag
    admissions_df[DIRECT_IND_COL] = (admissions_df[ADMISSION_TYPE_COL] == 'DIRECT EMER.').astype(int)

    # # Counting SPECIFIC_ADMISSION_TYPE admissions to each patient
    number_of_admissions = admissions_df.groupby(SUBJECT_ID_COL)[ADMISSION_ID_COL].nunique()
    number_of_admissions.name = ADMISSION_COUNT_COL

    admissions_df = admissions_df.merge(number_of_admissions, on=SUBJECT_ID_COL)

    # # Add recurrent admissions group per patient according to last admission
    ADMISSION_COUNT_BINS = [1, 1.5, 2.5, 5000]
    ADMISSION_COUNT_LABELS = ['1', '2', '3up']

    admissions_df[ADMISSION_COUNT_GROUP_COL] = pd.cut(admissions_df[ADMISSION_COUNT_COL],
                                                      bins=ADMISSION_COUNT_BINS,
                                                      labels=ADMISSION_COUNT_LABELS,
                                                      include_lowest=True)

    # # Adds last admission with previous admission in past month indicator
    indicator_diff = pd.to_timedelta('30d')

    tmp_admissions = admissions_df[admissions_df[ADMISSION_COUNT_COL] > 1]
    ind_ser = tmp_admissions.sort_values(by=[SUBJECT_ID_COL, ADMISSION_TIME_COL]).groupby(
        SUBJECT_ID_COL).apply(
        lambda tmp_df: (tmp_df[ADMISSION_TIME_COL] - tmp_df[DISCHARGE_TIME_COL].shift(1)) <= indicator_diff)

    ind_ser.index = ind_ser.index.droplevel(1)
    ind_ser.name = PREV_ADMISSION_IND_COL
    ind_ser = ind_ser.iloc[ind_ser.reset_index().drop_duplicates(subset=[SUBJECT_ID_COL], keep='last').index]

    admissions_df = admissions_df.merge(ind_ser.astype(int), left_on=SUBJECT_ID_COL, right_index=True, how='outer')
    admissions_df[PREV_ADMISSION_IND_COL].fillna(0, inplace=True)

    # # Keep only last admission per patient
    only_last_admission = admissions_df.sort_values(by=[ADMISSION_TIME_COL]).drop_duplicates(
        subset=[SUBJECT_ID_COL],
        keep='last')

    # # Take only patients with last admission after MINIMUM YEAR
    only_last_admission = only_last_admission[only_last_admission[ADMISSION_YEAR_COL] >= minimum_year]

    pids = only_last_admission[SUBJECT_ID_COL].drop_duplicates()
    adm_ids = only_last_admission[ADMISSION_ID_COL].drop_duplicates()

    # # Load relevant lab tests
    LOAD_SPECIFIC_COLUMNS = [SUBJECT_ID_COL, ADMISSION_ID_COL, ITEM_ID_COL, 'storetime', 'flag']

    full_df = pd.DataFrame()
    with pd.read_csv(lab_file, chunksize=chunksize, compression='gzip', parse_dates=[STORE_TIME_COL],
                     usecols=LOAD_SPECIFIC_COLUMNS) as reader:
        for chunk in reader:
            tmp_chunk = chunk[chunk[SUBJECT_ID_COL].isin(pids) & chunk[ADMISSION_ID_COL].isin(adm_ids)]
            tmp_adms = only_last_admission[
                only_last_admission[SUBJECT_ID_COL].isin(pids) & only_last_admission[ADMISSION_ID_COL].isin(
                    adm_ids)]
            # tmp_patinets = patients_df[patients_df[SUBJECT_ID_COL].isin(pids)]
            tmp_chunk = tmp_chunk.merge(tmp_adms, on=[SUBJECT_ID_COL, ADMISSION_ID_COL])
            # tmp = tmp_chunk.merge(tmp_patinets, on=[SUBJECT_ID_COL])
            full_df = pd.concat([full_df, tmp_chunk])

    # # Continue only with included patients_df and admissions_df and full_df
    pids = full_df[SUBJECT_ID_COL].drop_duplicates().values
    adms_ids = full_df[ADMISSION_ID_COL].drop_duplicates().values
    patients_df = patients_df[patients_df[SUBJECT_ID_COL].isin(pids)]
    admissions_df = admissions_df[admissions_df[ADMISSION_ID_COL].isin(adms_ids)]

    # # Regrouping discharge location
    discharge_regrouping_df = pd.Series(DISCHARGE_REGROUPING_DICT).to_frame()
    discharge_regrouping_df.index.name = 'Original Group'
    discharge_regrouping_df.columns = ['Regrouped']
    admissions_df[DISCHARGE_LOCATION_COL].replace(DISCHARGE_REGROUPING_DICT, inplace=True)
    full_df[DISCHARGE_LOCATION_COL].replace(DISCHARGE_REGROUPING_DICT, inplace=True)

    # # Regroup Race
    race_regrouping_df = pd.Series(RACE_REGROUPING_DICT).to_frame()
    race_regrouping_df.index.name = 'Original Group'
    race_regrouping_df.columns = ['Regrouped']
    admissions_df[RACE_COL].replace(RACE_REGROUPING_DICT, inplace=True)
    full_df[RACE_COL].replace(RACE_REGROUPING_DICT, inplace=True)

    # # Taking only results 24 hours from admission
    full_df[ADMISSION_TO_RESULT_COL] = (full_df[STORE_TIME_COL] - full_df[ADMISSION_TIME_COL])
    full_df = full_df[full_df[ADMISSION_TO_RESULT_COL] <= pd.to_timedelta('1d')]

    full_df.sort_values(by=[ADMISSION_TIME_COL, STORE_TIME_COL]).drop_duplicates(
        subset=[SUBJECT_ID_COL, ADMISSION_ID_COL, ITEM_ID_COL],
        inplace=True, keep='last')

    # # Most common lab tests upon arrival
    lab_meta_df = pd.read_csv(lab_meta_file, compression='gzip')
    threshold = 25000

    common_tests = full_df.groupby(ITEM_ID_COL)[ADMISSION_ID_COL].nunique().sort_values(ascending=False)
    included_in_threshold = common_tests[common_tests > threshold].to_frame().merge(lab_meta_df, on=ITEM_ID_COL)
    full_df = full_df[full_df[ITEM_ID_COL].isin(included_in_threshold[ITEM_ID_COL].values)]
    minimal_item_id = included_in_threshold.iloc[-1][ITEM_ID_COL]

    pids = full_df[full_df[ITEM_ID_COL] == minimal_item_id][SUBJECT_ID_COL].drop_duplicates().values
    adms_ids = full_df[full_df[ITEM_ID_COL] == minimal_item_id][ADMISSION_ID_COL].drop_duplicates().values
    patients_df = patients_df[patients_df[SUBJECT_ID_COL].isin(pids)]
    admissions_df = admissions_df[admissions_df[ADMISSION_ID_COL].isin(adms_ids)]
    full_df = full_df[full_df[SUBJECT_ID_COL].isin(pids)]
    full_df = full_df[full_df[ADMISSION_ID_COL].isin(adms_ids)]

    full_df['flag'].fillna('normal', inplace=True)
    full_df['flag'].replace({'normal': 0, 'abnormal': 1}, inplace=True)
    full_df['flag'].value_counts()

    full_df = full_df.sort_values(by=[ADMISSION_TIME_COL, STORE_TIME_COL]).drop_duplicates(
        subset=[SUBJECT_ID_COL, ADMISSION_ID_COL, ITEM_ID_COL],
        keep='last')

    tmp = full_df[[SUBJECT_ID_COL, ADMISSION_ID_COL, ITEM_ID_COL, 'flag']]
    fitters_table = pd.pivot_table(tmp, values=['flag'], index=[SUBJECT_ID_COL, ADMISSION_ID_COL],
                                   columns=[ITEM_ID_COL], aggfunc=np.sum)

    fitters_table = fitters_table.droplevel(1, axis=0).droplevel(0, axis=1)
    dummies_df = full_df.drop_duplicates(subset=[SUBJECT_ID_COL]).set_index(SUBJECT_ID_COL)

    del full_df
    del admissions_df
    del patients_df

    # # Standardize age
    scaler = StandardScaler()
    dummies_df[STANDARDIZED_AGE_COL] = scaler.fit_transform(dummies_df[[AGE_COL]])

    J_DICT = {'HOME': 1, 'FURTHER TREATMENT': 2, 'DIED': 3, 'CENSORED': 0}
    GENDER_DICT = {'F': 1, 'M': 0}

    dummies_df[GENDER_COL] = dummies_df[GENDER_COL].replace(GENDER_DICT)

    included_in_threshold['label'] = included_in_threshold['label'].apply(lambda x: x.replace(' ', '')).apply(
        lambda x: x.replace(',', ''))
    RENAME_ITEMS_DICT = included_in_threshold[[ITEM_ID_COL, 'label']].set_index(ITEM_ID_COL).to_dict()['label']

    table1 = pd.concat([
        fitters_table.copy(),
        dummies_df[[NIGHT_ADMISSION_FLAG,
                    GENDER_COL,
                    DIRECT_IND_COL,
                    PREV_ADMISSION_IND_COL,
                    ADMISSION_AGE_COL]].astype(int),
        dummies_df[[INSURANCE_COL,
                    MARITAL_STATUS_COL,
                    RACE_COL,
                    ADMISSION_COUNT_GROUP_COL]],
        dummies_df[LOS_DAYS_COL].dt.days,
        dummies_df[DISCHARGE_LOCATION_COL].dropna().replace(J_DICT).astype(int)
    ], axis=1)

    table1.rename(RENAME_ITEMS_DICT, inplace=True, axis=1)
    table1.dropna(inplace=True)

    # %%
    ADMINISTRATIVE_CENSORING = 28

    censoring_index = table1[table1[LOS_DAYS_COL] > ADMINISTRATIVE_CENSORING].index
    table1.loc[censoring_index, DISCHARGE_LOCATION_COL] = 0
    table1.loc[censoring_index, LOS_DAYS_COL] = ADMINISTRATIVE_CENSORING + 1
    table1[GENDER_COL].replace(table1_rename_sex, inplace=True)
    table1[RACE_COL].replace(table1_rename_race, inplace=True)
    table1[MARITAL_STATUS_COL].replace(table1_rename_marital, inplace=True)
    table1[DIRECT_IND_COL].replace(table1_rename_yes_no, inplace=True)
    table1[NIGHT_ADMISSION_FLAG].replace(table1_rename_yes_no, inplace=True)
    table1[PREV_ADMISSION_IND_COL].replace(table1_rename_yes_no, inplace=True)
    table1[DISCHARGE_LOCATION_COL].replace(table1_rename_discharge, inplace=True)
    table1[ADMISSION_COUNT_GROUP_COL].replace({'3up': '3+'}, inplace=True)
    table1.rename(table1_rename_columns, inplace=True, axis=1)

    # %%
    columns = ['gender', 'admission_age', 'race', 'insurance', 'marital_status',
               'direct_emrgency_flag', 'night_admission', 'last_less_than_diff',
               'admissions_count_group', 'LOS days', 'discharge_location']
    columns = [table1_rename_columns[c] for c in columns]
    categorical = ['gender', 'race', 'insurance', 'marital_status',
                   'direct_emrgency_flag', 'night_admission', 'last_less_than_diff',
                   'admissions_count_group', 'discharge_location']
    categorical = [table1_rename_columns[c] for c in categorical]
    table1.dropna(inplace=True)
    groupby = [table1_rename_columns[DISCHARGE_LOCATION_COL]]

    if return_table1:
        ensure_package("tableone")
        from tableone import TableOne
        mytable = TableOne(table1, columns, categorical, groupby, missing=False)
        # Patients' characteristics
        # print(mytable.tableone.round(3).to_latex())
        characteristics_table1 = mytable.tableone.round(3).copy()

    columns = [DISCHARGE_LOCATION_COL, 'AnionGap', 'Bicarbonate', 'CalciumTotal', 'Chloride', 'Creatinine',
               'Glucose', 'Magnesium', 'Phosphate', 'Potassium', 'Sodium',
               'UreaNitrogen', 'Hematocrit', 'Hemoglobin', 'MCH', 'MCHC', 'MCV',
               'PlateletCount', 'RDW', 'RedBloodCells', 'WhiteBloodCells']
    categorical = [DISCHARGE_LOCATION_COL, 'AnionGap', 'Bicarbonate', 'CalciumTotal', 'Chloride', 'Creatinine',
                   'Glucose', 'Magnesium', 'Phosphate', 'Potassium', 'Sodium',
                   'UreaNitrogen', 'Hematocrit', 'Hemoglobin', 'MCH', 'MCHC', 'MCV',
                   'PlateletCount', 'RDW', 'RedBloodCells', 'WhiteBloodCells']
    columns = [table1_rename_columns[c] for c in columns]
    categorical = [table1_rename_columns[c] for c in categorical]
    groupby = [table1_rename_columns[DISCHARGE_LOCATION_COL]]

    if return_table1:
        mytable = TableOne(table1.dropna().replace(table1_rename_normal_abnormal), columns, categorical, groupby,
                           missing=False)

        # Patients' lab tests
        # print(mytable.tableone.round(3).to_latex())
        labs_table1 = mytable.tableone.round(3).copy()

    # %%
    fitters_table = pd.concat([
        fitters_table.copy(),
        pd.get_dummies(dummies_df[INSURANCE_COL], prefix='Insurance', drop_first=True),
        pd.get_dummies(dummies_df[MARITAL_STATUS_COL], prefix='Marital', drop_first=True),
        pd.get_dummies(dummies_df[RACE_COL], prefix='Ethnicity', drop_first=True),
        pd.get_dummies(dummies_df[ADMISSION_COUNT_GROUP_COL], prefix='AdmsCount', drop_first=True),
        dummies_df[[NIGHT_ADMISSION_FLAG,
                    GENDER_COL,
                    DIRECT_IND_COL,
                    PREV_ADMISSION_IND_COL]].astype(int),
        dummies_df[STANDARDIZED_AGE_COL],
        dummies_df[LOS_DAYS_COL].dt.days,
        dummies_df[DISCHARGE_LOCATION_COL].dropna().replace(J_DICT).astype(int)
    ], axis=1)

    fitters_table.dropna(inplace=True)
    fitters_table = fitters_table[fitters_table.index.isin(table1.index)]

    fitters_table.reset_index(inplace=True)
    fitters_table.rename({DISCHARGE_LOCATION_COL: 'J', LOS_DAYS_COL: 'X', SUBJECT_ID_COL: 'pid'}, inplace=True,
                         axis=1)
    fitters_table.rename(RENAME_ITEMS_DICT, inplace=True, axis=1)

    # %%
    fitters_table = fitters_table[fitters_table['X'] > 0]
    fitters_table.loc[fitters_table.X > ADMINISTRATIVE_CENSORING, 'J'] = 0
    fitters_table.loc[fitters_table.X > ADMINISTRATIVE_CENSORING, 'X'] = ADMINISTRATIVE_CENSORING + 1
    fitters_table['J'] = fitters_table['J'].astype(int)

    if return_table1:
        return fitters_table, characteristics_table1, labs_table1

    return fitters_table