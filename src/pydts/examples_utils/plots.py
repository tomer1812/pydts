import numpy as np
from pydts.examples_utils.simulations_data_config import *
from matplotlib import pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import warnings
warnings.filterwarnings('ignore')


def plot_first_model_coefs(models, times, expanded_train_df, n_cov=5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.set_title(r'$\alpha_{jt}$', fontsize=26)
    ax.scatter(times, models[1].params[:len(times)].values, label='J=1 (Pred)', color='tab:blue')
    ax.plot(times, -1 -0.3*np.log(times), label='J=1 (True)', ls='--', color='tab:blue')
    ax.scatter(times, models[2].params[:len(times)].values, label='J=2 (Pred)', color='tab:green')
    ax.plot(times, -1.75 -0.15*np.log(times), label='J=2 (True)', ls='--', color='tab:green')
    ax.set_xlabel(r'Time', fontsize=16)
    ax.set_ylabel(r'$\alpha_{t}$', fontsize=20)
    ax.legend(loc='upper center', fontsize=14)
    ax.set_ylim([-3, 0.5])
    ax2 = ax.twinx()
    ax2.hist(expanded_train_df['X'], color='r', alpha=0.3, bins=times)
    ax2.set_ylabel('N patients', fontsize=16, color='red')
    ax2.tick_params(axis='y', colors='red')

    ax = axes[1]
    ax.set_title(r'$\beta_{j}$', fontsize=26)
    ax.bar(np.arange(1, n_cov+1), models[1].params[-n_cov:], label='J=1 (Pred)', width=0.3, alpha=0.4, color='tab:blue')
    ax.scatter(-0.2+np.arange(1, n_cov+1), -np.log([0.8, 3, 3, 2.5, 2]), color='tab:blue', label='J=1 (True)',
               marker="4", s=130)
    ax.bar(np.arange(1, n_cov+1), models[2].params[-n_cov:], color='tab:green', label='J=2 (Pred)', align='edge',
           width=0.3, alpha=0.4)
    ax.scatter(0.35+np.arange(1, n_cov+1), -np.log([1, 3, 4, 3, 2]), color='tab:green', label='J=2 (True)', marker="3",
               s=130)
    ax.legend(loc='upper center', fontsize=14)
    ax.set_ylim([-1.5, 1])
    fig.tight_layout()


def plot_second_model_coefs(alpha_df, beta_models, times, n_cov=5):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.set_title(r'$\alpha_{jt}$', fontsize=26)
    tmp_ajt = alpha_df[alpha_df['J'] == 1]
    ax.scatter(tmp_ajt['X'].values, tmp_ajt['alpha_jt'].values, label='J=1 (Pred)', color='tab:blue')
    ax.plot(times, -1 -0.3*np.log(times), label='J=1 (True)', ls='--', color='tab:blue' )
    tmp_ajt = alpha_df[alpha_df['J'] == 2]
    ax.scatter(tmp_ajt['X'].values, tmp_ajt['alpha_jt'].values, label='J=2 (Pred)', color='tab:green')
    ax.plot(times, -1.75 -0.15*np.log(times), label='J=2 (True)', ls='--', color='tab:green')
    ax.set_xlabel(r'Time', fontsize=16)
    ax.set_ylabel(r'$\alpha_{t}$', fontsize=20)
    ax.legend(loc='upper center', fontsize=14)
    ax.set_ylim([-3, 0.5])
    ax2 = ax.twinx()
    ax2.bar(alpha_df.groupby('X')['n_jt'].sum().index, alpha_df.groupby('X')['n_jt'].sum().values, color='r', alpha=0.3)
    ax2.set_ylabel('N patients', fontsize=16, color='red')
    ax2.tick_params(axis='y', colors='red')

    ax = axes[1]
    ax.set_title(r'$\beta_{j}$', fontsize=26)
    ax.bar(np.arange(1, n_cov+1), beta_models[1].params_.values, label='J=1 (Pred)', width=0.3, alpha=0.4,
           color='tab:blue')
    ax.scatter(-0.2+np.arange(1, n_cov+1), -np.log([0.8, 3, 3, 2.5, 2]), color='tab:blue', label='J=1 (True)',
               marker="4", s=130)
    ax.bar(np.arange(1, n_cov+1), beta_models[2].params_.values, color='tab:green', label='J=2 (Pred)', align='edge',
           width=0.3, alpha=0.4)
    ax.scatter(0.35+np.arange(1, n_cov+1), -np.log([1, 3, 4, 3, 2]), color='tab:green', label='J=2 (True)', marker="3",
               s=130)
    ax.legend(loc='upper center', fontsize=14)
    ax.set_ylim([-1.5, 1])
    fig.tight_layout()
    ax.legend(loc='upper center', fontsize=14)
    fig.tight_layout()


def plot_LOS_simulation_figure1(data_df):
    text_sz = 16

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    data_df.groupby([ADMISSION_YEAR_COL, DEATH_MISSING_COL]).size().unstack().plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel('Number of patients', fontsize=font_sz)
    ax.set_xlabel('Year of admission', fontsize=font_sz)
    ax.legend(labels=['Yes', 'No'], title="In hospital death", fontsize='small', fancybox=True)
    ax.set_ylim([0, 1000])

    ax = axes[0, 1]
    tmp = data_df[[AGE_COL, GENDER_COL]]
    tmp[AGE_COL] = pd.cut(tmp[AGE_COL], bins=AGE_BINS, labels=AGE_LABELS)
    tmp.groupby([AGE_COL, GENDER_COL]).size().unstack().plot(kind='bar', ax=ax)
    ax.set_xlabel('Age [years]', fontsize=font_sz)
    ax.set_ylabel('Number of patients', fontsize=font_sz)
    ax.set_xticklabels(AGE_LABELS, rotation=90)

    ax = axes[1, 0]
    ser = data_df.sort_values(by=[PATIENT_NO_COL, ADMISSION_SERIAL_COL]).drop_duplicates(
        subset=[PATIENT_NO_COL], keep='last')[ADMISSION_SERIAL_COL]
    sns.distplot(ser, kde=False, ax=ax)
    ax.set_yscale('log')
    ax.set_ylabel('Number of patients', fontsize=font_sz)
    ax.set_xlabel('Number of admissions', fontsize=font_sz)
    ax.axvline(1.5, ls='--', color='r')
    ax.axvline(4.5, ls='--', color='r')
    ax.axvline(8.5, ls='--', color='r')
    ax.text(x=0.5, y=1100, s='0', fontsize=text_sz, color='r')
    ax.text(x=2.5, y=1100, s='1', fontsize=text_sz, color='r')
    ax.text(x=6.5, y=1100, s='2', fontsize=text_sz, color='r')
    ax.text(x=10, y=1100, s='3', fontsize=text_sz, color='r')
    ax.text(x=9, y=200, s='Returning patient\ngroup', fontsize=text_sz, color='r')

    ax.grid('both', 'both')

    ax = axes[1, 1]
    kmf = KaplanMeierFitter(label='In-hospital death censoring')
    T = data_df[DISCHARGE_RELATIVE_COL].values
    E = data_df[DEATH_RELATIVE_COL].isnull().astype(int).values
    kmf.fit(durations=T, event_observed=E)
    kmf.plot_survival_function(ax=ax)

    kmf = KaplanMeierFitter(label='No censoring')
    T = data_df[DISCHARGE_RELATIVE_COL].values
    E = np.ones_like(T)
    kmf.fit(durations=T, event_observed=E)
    kmf.plot_survival_function(ax=ax)

    ax.set_ylabel('Population', fontsize=font_sz)
    ax.set_xlabel('Days from hospitalization', fontsize=font_sz)
    ax.grid()

    fig.tight_layout()

def plot_LOS_simulation_figure2(data_df):
    tmp = data_df.copy()
    tmp['binned_age'] = pd.cut(tmp[AGE_COL], bins=AGE_BINS, labels=AGE_LABELS)
    tmp['death_at_hosp_ind'] = tmp[DEATH_RELATIVE_COL].notnull().astype(int)

    max_time = 130
    total_ad = len(data_df)
    ihd = (tmp.groupby(DEATH_RELATIVE_COL).size() / total_ad).reindex(range(0, max_time + 1)).fillna(0)
    ihd = ihd.loc[:max_time].cumsum()
    released = ((tmp.groupby(DISCHARGE_RELATIVE_COL).size() -
                 tmp.groupby(DEATH_RELATIVE_COL).size()) / total_ad).reindex(range(0, max_time + 1)).fillna(0)
    released = released.loc[:max_time].cumsum()
    o = np.ones_like(released)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    text_sz = 12
    every_nth = 5

    ax = axes[0, 0]
    ax.fill_between(x=released.index, y1=(o - released.values - ihd.values), color='b', alpha=0.4)
    ax.fill_between(x=released.index, y1=o, y2=(o - released.values), color='g', alpha=0.4)
    ax.fill_between(x=released.index, y1=(o - released.values), y2=(o - released.values - ihd.values), color='r',
                    alpha=0.4)
    ax.text(x=1, y=0.025, s='Hospitalized', fontsize=text_sz)
    ax.text(x=max_time - 30, y=0.8, s='Released', fontsize=text_sz)
    ax.text(x=max_time - 30, y=0.05, s='Dead', fontsize=text_sz)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, max_time])
    ax.set_xlabel('Days from hospitalization day', fontsize=font_sz)
    ax.set_ylabel('Patient status ratio', fontsize=font_sz)

    ax = axes[0, 1]
    tmp[tmp['death_at_hosp_ind'] == 1].groupby(['binned_age', GENDER_COL]).size().unstack().plot(kind='bar', ax=ax)
    ax.set_xlabel('Age [years]', fontsize=font_sz)
    ax.set_ylabel('Number of patients', fontsize=font_sz)
    ax.set_xticklabels(AGE_LABELS, rotation=90)
    ax.set_title('In-hospital Death', fontsize=title_sz)

    ax = axes[1, 0]
    ser = tmp[tmp['death_at_hosp_ind'] != 1].groupby(DISCHARGE_RELATIVE_COL).size()
    ser = ser.reindex(range(0, max_time + 1)).fillna(0)
    ser.plot(kind='bar', ax=ax)
    ax.set_xlim([0, max_time])
    ax.set_ylabel('Released', fontsize=font_sz)
    ax.set_xlabel('Days from Hospitalization', fontsize=font_sz)
    ax.grid(axis='y')

    for idl, label in enumerate(ax.xaxis.get_ticklabels()):
        if (idl % every_nth) > 0:
            label.set_visible(False)

    ax = axes[1, 1]
    ser = tmp[tmp['death_at_hosp_ind'] == 1].groupby(DEATH_RELATIVE_COL).size()
    ser = ser.reindex(range(0, max_time + 1)).fillna(0)
    ser.plot(kind='bar', ax=ax)
    ax.set_xlim([0, max_time])
    ax.set_ylabel('Died', fontsize=font_sz)
    ax.set_xlabel('Days from Hospitalization', fontsize=font_sz)
    ax.grid(axis='y')

    for idl, label in enumerate(ax.xaxis.get_ticklabels()):
        if (idl % every_nth) > 0:
            label.set_visible(False)

    ax = axes[2, 0]
    ser = tmp[tmp['death_at_hosp_ind'] != 1].groupby(DISCHARGE_RELATIVE_COL).size()
    ser = ser.reindex(range(0, max_time + 1)).fillna(0)
    ser.plot(kind='bar', logy=True, ax=ax)
    ax.set_xlim([0, max_time])
    ax.set_ylabel('Released', fontsize=font_sz)
    ax.set_xlabel('Days from Hospitalization', fontsize=font_sz)
    ax.grid(axis='y')
    ax.set_ylim([0, 1000])

    for idl, label in enumerate(ax.xaxis.get_ticklabels()):
        if (idl % every_nth) > 0:
            label.set_visible(False)

    ax = axes[2, 1]
    ser = tmp[tmp['death_at_hosp_ind'] == 1].groupby(DEATH_RELATIVE_COL).size()
    ser = ser.reindex(range(0, max_time + 1)).fillna(0)
    ser.plot(kind='bar', logy=True, ax=ax)
    ax.set_xlim([0, max_time])
    ax.set_ylabel('Died', fontsize=font_sz)
    ax.set_xlabel('Days from Hospitalization', fontsize=font_sz)
    ax.grid(axis='y')
    ax.set_ylim([0, 1000])

    for idl, label in enumerate(ax.xaxis.get_ticklabels()):
        if (idl % every_nth) > 0:
            label.set_visible(False)

    fig.tight_layout()

def plot_LOS_simulation_figure3(data_df):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    missingness_cols = [WEIGHT_COL]
    missingness_titles = [WEIGHT_COL]
    cols = missingness_cols
    tmp = data_df[cols + [ADMISSION_YEAR_COL]]
    tmp['missing'] = tmp.apply(lambda row: row.isnull().any(), axis=1)
    tmp.groupby([ADMISSION_YEAR_COL, 'missing']).size().unstack().plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel('Number of patients', fontsize=font_sz)
    ax.set_xlabel('Year of admission', fontsize=font_sz)
    ax.set_title(missingness_titles[0], fontsize=title_sz)
    ax.set_ylim([0, 1100])
    fig.tight_layout()
