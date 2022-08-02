from typing import Iterable, Tuple

import numpy as np
from .simulations_data_config import *
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from lifelines import KaplanMeierFitter
from ..config import *
import os
import string
import warnings
warnings.filterwarnings('ignore')


def add_panel_text(ax, text, xplace=-0.15, fsz=17):
    ax.text(xplace, 1.1, text, transform=ax.transAxes, fontsize=fsz,
            fontweight='bold', va='top', ha='right')


def plot_first_model_coefs(models, times, train_df, n_cov=5, filename=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    add_panel_text(ax=ax, text='a')
    ax.set_title(r'$\alpha_{jt}$', fontsize=26)
    ax.scatter(times, models[1].params[:len(times)].values, label='J=1 (Pred)', color='tab:blue')
    ax.plot(times, -1 -0.3*np.log(times), label='J=1 (True)', ls='--', color='tab:blue')
    ax.scatter(times, models[2].params[:len(times)].values, label='J=2 (Pred)', color='tab:green')
    ax.plot(times, -1.75 -0.15*np.log(times), label='J=2 (True)', ls='--', color='tab:green')
    ax.set_xlabel(r'Time', fontsize=18)
    ax.set_ylabel(r'$\alpha_{t}$', fontsize=18)
    ax.set_ylim([-3, 0.5])
    ax.legend(loc='upper center', fontsize=12)
    ax2 = ax.twinx()
    tmp_ajt = train_df[train_df['J'] == 1].groupby('X')['pid'].count()
    ax2.bar(tmp_ajt.index, tmp_ajt.values, label='J=1', color='tab:red', alpha=0.4, width=0.5)
    tmp_ajt = train_df[train_df['J'] == 2].groupby('X')['pid'].count()
    ax2.bar(tmp_ajt.index, tmp_ajt.values, label='J=2', color='tab:brown', alpha=0.6, align='edge', width=0.5)
    ax2.set_ylabel('Number of observed events', fontsize=16, color='red')
    ax2.tick_params(axis='y', colors='red')
    ax2.legend(loc='upper right', fontsize=12)

    ax = axes[1]
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    add_panel_text(ax=ax, text='b')
    ax.set_title(r'$\beta_{j}$', fontsize=26)
    ax.set_xlabel(r'Covariate', fontsize=18)
    ax.set_ylabel(r'$\beta}$', fontsize=18)
    ax.bar(np.arange(1, n_cov+1), models[1].params[-n_cov:], label='J=1 (Pred)', width=0.3, alpha=0.4, color='tab:blue')
    ax.scatter(-0.2+np.arange(1, n_cov+1), -np.log([0.8, 3, 3, 2.5, 2]), color='tab:blue', label='J=1 (True)',
               marker="4", s=130)
    ax.bar(np.arange(1, n_cov+1), models[2].params[-n_cov:], color='tab:green', label='J=2 (Pred)', align='edge',
           width=0.3, alpha=0.4)
    ax.scatter(0.35+np.arange(1, n_cov+1), -np.log([1, 3, 4, 3, 2]), color='tab:green', label='J=2 (True)', marker="3",
               s=130)
    ax.legend(loc='upper center', fontsize=12)
    ax.set_ylim([-1.5, 1])
    fig.tight_layout()
    if filename is not None:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)


def plot_second_model_coefs(alpha_df, beta_models, times, n_cov=5, filename=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    add_panel_text(ax=ax, text='a')
    ax.set_title(r'$\alpha_{jt}$', fontsize=26)
    tmp_ajt = alpha_df[alpha_df['J'] == 1]
    ax.scatter(tmp_ajt['X'].values, tmp_ajt['alpha_jt'].values, label='J=1 (two-step)', color='tab:blue')
    ax.plot(times, -1 -0.3*np.log(times), label='J=1 (True)', ls='--', color='tab:blue' )
    tmp_ajt = alpha_df[alpha_df['J'] == 2]
    ax.scatter(tmp_ajt['X'].values, tmp_ajt['alpha_jt'].values, label='J=2 (two-step)', color='tab:green')
    ax.plot(times, -1.75 -0.15*np.log(times), label='J=2 (True)', ls='--', color='tab:green')
    ax.set_xlabel(r'Time', fontsize=18)
    ax.set_ylabel(r'$\alpha_{t}$', fontsize=18)
    ax.legend(loc='upper center', fontsize=12)
    ax.set_ylim([-3, 0.5])
    ax2 = ax.twinx()
    tmp_ajt = alpha_df[alpha_df['J'] == 1].groupby('X')['n_jt'].sum()
    ax2.bar(tmp_ajt.index, tmp_ajt.values, label='J=1', color='tab:red', alpha=0.4, width=0.5)
    tmp_ajt = alpha_df[alpha_df['J'] == 2].groupby('X')['n_jt'].sum()
    ax2.bar(tmp_ajt.index, tmp_ajt.values, label='J=2', color='tab:brown', alpha=0.6, align='edge', width=0.5)
    ax2.legend(loc='upper right', fontsize=12)
    ax2.set_ylabel('Number of observed events', fontsize=16, color='red')
    ax2.tick_params(axis='y', colors='red')

    ax = axes[1]
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    add_panel_text(ax=ax, text='b')
    ax.set_title(r'$\beta_{j}$', fontsize=26)
    ax.bar(np.arange(1, n_cov+1), beta_models[1].params_.values, label='J=1 (two-step)', width=0.3, alpha=0.4,
           color='tab:blue')
    ax.scatter(-0.2+np.arange(1, n_cov+1), -np.log([0.8, 3, 3, 2.5, 2]), color='tab:blue', label='J=1 (True)',
               marker="4", s=130)
    ax.bar(np.arange(1, n_cov+1), beta_models[2].params_.values, color='tab:green', label='J=2 (two-step)', align='edge',
           width=0.3, alpha=0.4)
    ax.scatter(0.35+np.arange(1, n_cov+1), -np.log([1, 3, 4, 3, 2]), color='tab:green', label='J=2 (True)', marker="3",
               s=130)
    ax.legend(loc='upper center', fontsize=12)
    ax.set_xlabel('Covariate', fontsize=18)
    ax.set_ylabel(r'$\beta}$', fontsize=18)
    ax.set_ylim([-1.5, 1])
    fig.tight_layout()
    ax.legend(loc='upper center', fontsize=14)
    fig.tight_layout()
    if filename is not None:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)


def plot_models_coefficients(alpha_dict: dict, beta_dict: dict, times: Iterable,
                             counts_df: pd.DataFrame,
                             n_cov: int = 5,
                             first_model_name: str = 'Lee et al.',
                             second_model_name: str = 'two-step',
                             filename: str = None) -> None:
    """
    This method takes the repetitive runs results and plotting the comparison between the methods coefs

    Args:
        alpha_dict (dict): a dict that contains for each event type (key) a dataframe of all the $\alpha_t$ (value)
        beta_dict (dict): a dict that contains for each event type(key) a dataframe of all the $\beta_t$ (value)
        times (Iterable): array like that contains all the unique times that were used
        counts_df (pandas.DataFrame): pandas dataframe which contains how many events per each time t
        n_cov (int): number of covariates (used to plot beta plot)
        first_model_name (Optional[str]): the name of the first model
        second_model_name (Optional[str]): the name of the second model

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    add_panel_text(ax=ax, text='a')
    ax.set_title(r'$\alpha_{jt}$', fontsize=26)
    tmp_ajt = alpha_dict[1]
    ax.scatter(times, tmp_ajt[f'Lee_mean'].values,
               label=f'J=1 ({first_model_name})', color='tab:blue', marker='o', alpha=0.4, s=40)
    ax.scatter(times, tmp_ajt[f'Ours_mean'].values,
               label=f'J=1 ({second_model_name})', color='navy', marker='*', alpha=0.7, s=40)
    ax.plot(times, tmp_ajt['real_mean'].values, label='J=1 (True)', ls='--', color='tab:blue')
    tmp_ajt = alpha_dict[2]
    ax.scatter(times, tmp_ajt[f'Lee_mean'].values,
               label=f'J=2 ({first_model_name})', color='tab:green', alpha=0.4, s=30)
    ax.scatter(times, tmp_ajt[f'Ours_mean'].values,
               label=f'J=2 ({second_model_name})', color='darkgreen', marker='*', alpha=0.7, s=30)
    ax.plot(times, tmp_ajt['real_mean'].values, label='J=2 (True)', ls='--', color='tab:green')
    ax.set_xlabel(r'Time', fontsize=18)
    ax.set_ylabel(r'$\alpha_{t}$', fontsize=18)
    ax.legend(loc='upper center', fontsize=12)
    ax.set_ylim([-3, 0.5])
    ax2 = ax.twinx()
    ax2.bar(times, counts_df.loc[1].values.squeeze(), label='J=1', color='tab:red', alpha=0.4, width=0.5)
    ax2.bar(times, counts_df.loc[2].values.squeeze(), label='J=2', color='tab:brown', alpha=0.6, align='edge',
            width=0.5)
    ax2.legend(loc='upper right', fontsize=12)
    ax2.set_ylabel('Number of observed events', fontsize=16, color='red')
    ax2.tick_params(axis='y', colors='red')

    ax = axes[1]
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.set_title(r'$\beta_{j}$', fontsize=26)
    add_panel_text(ax=ax, text='b')
    beta_j = beta_dict[1]
    ax.bar(np.arange(1, n_cov + 1), beta_j['real_mean'].values, label='J=1 (True)', width=0.3, alpha=0.4,
           color='tab:blue')
    ax.scatter(-0.2 + np.arange(1, n_cov + 1), beta_j[f'Lee_mean'].values,
               color='tab:blue', label=f'J=1 ({first_model_name})', marker="4", s=130)
    ax.scatter(-0.2 + np.arange(1, n_cov + 1), beta_j[f'Ours_mean'].values,
               color='navy', label=f'J=1 ({second_model_name})', marker=">", s=130, alpha=0.4)

    beta_j = beta_dict[2]
    ax.bar(np.arange(1, n_cov + 1), beta_j['real_mean'].values, color='tab:green', label='J=2 (True)', align='edge',
           width=0.3, alpha=0.4)
    ax.scatter(0.35 + np.arange(1, n_cov + 1), beta_j[f'Lee_mean'].values,
               color='tab:green', label=f'J=2 ({first_model_name})', marker="3",
               s=130)
    ax.scatter(0.35 + np.arange(1, n_cov + 1), beta_j[f'Ours_mean'].values,
               color='darkgreen', label=f'J=2 ({second_model_name})', marker="<",
               s=130, alpha=0.4)
    ax.set_xlabel('Covariate', fontsize=18)
    ax.set_ylabel(r'$\beta}$', fontsize=18)
    ax.legend(loc='upper center', fontsize=12)
    ax.set_ylim([-1.5, 1])
    fig.tight_layout()
    if filename is not None:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)


def plot_LOS_simulation_figure1(data_df):
    text_sz = 16

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    add_panel_text(ax=ax, text='a')
    data_df.groupby([ADMISSION_YEAR_COL, DEATH_MISSING_COL]).size().unstack().plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel('Number of patients', fontsize=font_sz)
    ax.set_xlabel('Year of admission', fontsize=font_sz)
    ax.legend(labels=['Yes', 'No'], title="In hospital death", fontsize='small', fancybox=True)
    ax.set_ylim([0, 1000])

    ax = axes[0, 1]
    add_panel_text(ax=ax, text='b')
    tmp = data_df[[AGE_COL, GENDER_COL]]
    tmp[AGE_COL] = pd.cut(tmp[AGE_COL], bins=AGE_BINS, labels=AGE_LABELS)
    tmp.groupby([AGE_COL, GENDER_COL]).size().unstack().plot(kind='bar', ax=ax)
    ax.set_xlabel('Age [years]', fontsize=font_sz)
    ax.set_ylabel('Number of patients', fontsize=font_sz)
    ax.legend(labels=['Male', 'Female'], title="Sex")
    ax.set_xticklabels(AGE_LABELS, rotation=90)

    ax = axes[1, 0]
    add_panel_text(ax=ax, text='c')
    ser = data_df.sort_values(by=[PATIENT_NO_COL, ADMISSION_SERIAL_COL]).drop_duplicates(
        subset=[PATIENT_NO_COL], keep='last')[ADMISSION_SERIAL_COL]
    sns.distplot(ser, kde=False, ax=ax)
    ax.set_yscale('log')
    ax.set_ylabel('Number of patients', fontsize=font_sz)
    ax.set_xlabel('Number of admissions', fontsize=font_sz)
    ax.set_xticks(list(range(1, int(data_df[ADMISSION_SERIAL_COL].max()+1))))
    ax.set_xticklabels(list(range(1, int(data_df[ADMISSION_SERIAL_COL].max()+1))))

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
    add_panel_text(ax=ax, text='d')
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
    ax.set_xlabel('LOS (Days)', fontsize=font_sz)
    ax.set_xlim([0, 30])
    ax.grid()

    fig.tight_layout()

def plot_LOS_simulation_figure2(data_df):
    tmp = data_df.copy()
    tmp['binned_age'] = pd.cut(tmp[AGE_COL], bins=AGE_BINS, labels=AGE_LABELS)
    tmp['death_at_hosp_ind'] = tmp[DEATH_RELATIVE_COL].notnull().astype(int)

    max_time = 30
    total_ad = len(data_df)
    ihd = (tmp.groupby(DEATH_RELATIVE_COL).size() / total_ad).reindex(range(0, max_time + 1)).fillna(0)
    ihd = ihd.loc[:max_time].cumsum()
    released = ((tmp.groupby(DISCHARGE_RELATIVE_COL).size() -
                 tmp.groupby(DEATH_RELATIVE_COL).size()) / total_ad).reindex(range(0, max_time + 1)).fillna(0)
    released = released.loc[:max_time].cumsum()
    o = np.ones_like(released)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    text_sz = 12
    every_nth = 5

    ax = axes[0, 0]
    add_panel_text(ax=ax, text='a')
    ax.fill_between(x=released.index, y1=(o - released.values - ihd.values), color='b', alpha=0.4)
    ax.fill_between(x=released.index, y1=o, y2=(o - released.values), color='g', alpha=0.4)
    ax.fill_between(x=released.index, y1=(o - released.values), y2=(o - released.values - ihd.values), color='r',
                    alpha=0.4)
    ax.text(x=1, y=0.2, s='Hospitalized', fontsize=text_sz)
    ax.text(x=20, y=0.8, s='Released', fontsize=text_sz)
    ax.text(x=20, y=0.3, s='Dead', fontsize=text_sz)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, max_time])
    ax.set_xlabel('LOS (Days)', fontsize=font_sz)
    ax.set_ylabel('Patient status ratio', fontsize=font_sz)

    ax = axes[0, 1]
    add_panel_text(ax=ax, text='b')
    tmp[tmp['death_at_hosp_ind'] == 1].groupby(['binned_age', GENDER_COL]).size().unstack().plot(kind='bar', ax=ax)
    ax.set_xlabel('Age [years]', fontsize=font_sz)
    ax.set_ylabel('Number of patients', fontsize=font_sz)
    ax.set_xticklabels(AGE_LABELS, rotation=90)
    ax.set_title('In-hospital Death', fontsize=title_sz)

    ax = axes[1, 0]
    add_panel_text(ax=ax, text='c')
    ser = tmp[tmp['death_at_hosp_ind'] != 1].groupby(DISCHARGE_RELATIVE_COL).size()
    ser = ser.reindex(range(0, max_time + 1)).fillna(0)
    ser.plot(kind='bar', ax=ax)
    ax.set_xlim([0, max_time])
    ax.set_ylabel('Released', fontsize=font_sz)
    ax.set_xlabel('LOS (Days)', fontsize=font_sz)
    ax.grid(axis='y')

    for idl, label in enumerate(ax.xaxis.get_ticklabels()):
        if (idl % every_nth) > 0:
            label.set_visible(False)

    ax = axes[1, 1]
    add_panel_text(ax=ax, text='d')
    ser = tmp[tmp['death_at_hosp_ind'] == 1].groupby(DEATH_RELATIVE_COL).size()
    ser = ser.reindex(range(0, max_time + 1)).fillna(0)
    ser.plot(kind='bar', ax=ax)
    ax.set_xlim([0, max_time])
    ax.set_ylabel('Died', fontsize=font_sz)
    ax.set_xlabel('LOS (Days)', fontsize=font_sz)
    ax.grid(axis='y')

    for idl, label in enumerate(ax.xaxis.get_ticklabels()):
        if (idl % every_nth) > 0:
            label.set_visible(False)

    # ax = axes[2, 0]
    # add_panel_text(ax=ax, text='e')
    # ser = tmp[tmp['death_at_hosp_ind'] != 1].groupby(DISCHARGE_RELATIVE_COL).size()
    # ser = ser.reindex(range(0, max_time + 1)).fillna(0)
    # ser.plot(kind='bar', logy=True, ax=ax)
    # ax.set_xlim([0, max_time])
    # ax.set_ylabel('Released', fontsize=font_sz)
    # ax.set_xlabel('Days from Hospitalization', fontsize=font_sz)
    # ax.grid(axis='y')
    # ax.set_ylim([0, 1000])
    #
    # for idl, label in enumerate(ax.xaxis.get_ticklabels()):
    #     if (idl % every_nth) > 0:
    #         label.set_visible(False)
    #
    # ax = axes[2, 1]
    # add_panel_text(ax=ax, text='f')
    # ser = tmp[tmp['death_at_hosp_ind'] == 1].groupby(DEATH_RELATIVE_COL).size()
    # ser = ser.reindex(range(0, max_time + 1)).fillna(0)
    # ser.plot(kind='bar', logy=True, ax=ax)
    # ax.set_xlim([0, max_time])
    # ax.set_ylabel('Died', fontsize=font_sz)
    # ax.set_xlabel('Days from Hospitalization', fontsize=font_sz)
    # ax.grid(axis='y')
    # ax.set_ylim([0, 1000])
    #
    # for idl, label in enumerate(ax.xaxis.get_ticklabels()):
    #     if (idl % every_nth) > 0:
    #         label.set_visible(False)

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


def plot_LOS_simulation_desc_figure(data_df):
    text_sz = 16

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    ax.tick_params(axis='both', which='major', labelsize=text_sz-2)
    ax.tick_params(axis='both', which='minor', labelsize=text_sz-2)
    add_panel_text(ax=ax, text='a')
    tmp = data_df[[AGE_COL, GENDER_COL]]
    tmp[AGE_COL] = pd.cut(tmp[AGE_COL], bins=AGE_BINS, labels=AGE_LABELS)
    tmp.groupby([AGE_COL, GENDER_COL]).size().unstack().plot(kind='bar', ax=ax)
    ax.set_xlabel('Age (years)', fontsize=font_sz)
    ax.set_ylabel('Number of patients', fontsize=font_sz)
    ax.legend(labels=['Male', 'Female'], title="Sex")
    ax.set_xticklabels(AGE_LABELS, rotation=90)

    ax = axes[0, 1]
    ax.tick_params(axis='both', which='major', labelsize=text_sz-2)
    ax.tick_params(axis='both', which='minor', labelsize=text_sz-2)
    add_panel_text(ax=ax, text='b')
    tmp = data_df[[SMOKING_COL, HYPERTENSION_COL, DIABETES_COL, ART_FIB_COL, COPD_COL, CRF_COL]].sum(axis=1).to_frame()
    tmp.columns = ['pre']
    h = tmp.groupby('pre').size()
    ax.bar(h.index, h.values, width=0.4)
    ax.set_xlabel('Number of preconditions', fontsize=font_sz)
    ax.set_ylabel('Number of patients', fontsize=font_sz)

    tmp = data_df.copy()
    tmp['binned_age'] = pd.cut(tmp[AGE_COL], bins=AGE_BINS, labels=AGE_LABELS)
    tmp['death_at_hosp_ind'] = tmp[DEATH_RELATIVE_COL].notnull().astype(int)

    ax = axes[1, 0]
    ax.tick_params(axis='both', which='major', labelsize=text_sz-2)
    ax.tick_params(axis='both', which='minor', labelsize=text_sz-2)
    add_panel_text(ax=ax, text='c')
    ser = tmp[tmp['death_at_hosp_ind'] != 1].groupby(DISCHARGE_RELATIVE_COL).size()
    ser = ser.reindex(range(0, 31)).fillna(0)
    ser.plot(kind='bar', ax=ax)
    ax.set_xlim([0, 30])
    ax.set_ylabel('Released', fontsize=font_sz)
    ax.set_xlabel('LOS (Days)', fontsize=font_sz)
    ax.grid(axis='y')

    ax = axes[1, 1]
    ax.tick_params(axis='both', which='major', labelsize=text_sz-2)
    ax.tick_params(axis='both', which='minor', labelsize=text_sz-2)
    add_panel_text(ax=ax, text='d')
    ser = tmp[tmp['death_at_hosp_ind'] == 1].groupby(DEATH_RELATIVE_COL).size()
    ser = ser.reindex(range(0, 31)).fillna(0)
    ser.plot(kind='bar', ax=ax)
    ax.set_ylabel('Died', fontsize=font_sz)
    ax.set_xlabel('LOS (Days)', fontsize=font_sz)
    ax.grid(axis='y')
    ax.set_xlim([0, 30])

    fig.tight_layout()


def compare_beta_models_for_example(first_models: dict, second_models: dict,
                                    n_cov: int = 5, real_coef_dict: dict = None) -> dict:

    from pydts.utils import compare_models_coef_per_event
    models_dict = {
        "alpha": {},
        "beta": {}
    }
    assert real_coef_dict is not None, "The user should supply the coefficients of the experiment"
    for event in first_models.keys():
        for model_type in models_dict.keys():
            if model_type == "alpha":
                first_slicing = slice(-n_cov)    # for alpha, similar to [:-n_cov]
                first_model = first_models[event].params[first_slicing].copy()
                first_model.index = first_model.index.str.replace(r"\D+", "", regex=True)
                first_model = first_model.add_prefix("a")
                second_model = second_models[event][1][["X", "alpha_jt"]].copy()
                second_model = second_model.set_index("X")["alpha_jt"].add_prefix("a")
                real_coef = real_coef_dict[model_type][event](np.arange(1, first_model.index.shape[0] + 1))
            else:
                first_slicing = slice(-n_cov, None)  # for beta, similar to [-n_cov:]
                first_model = first_models[event].params[first_slicing].copy()
                second_model = second_models[event][0].params_ # beta
                real_coef = real_coef_dict[model_type][event]
            models_dict[model_type][event] = compare_models_coef_per_event(first_model=first_model,
                                                                           second_model=second_model,
                                                                           real_values=real_coef,
                                                                           event=event,
                                                                           first_model_label="Lee",
                                                                           second_model_label="Ours"
                                                                           )
    return models_dict


def plot_reps_coef_std(rep_dict: dict, return_summary: bool = True, filename: str = None, paper_plots: bool = False):
    alphabet_list = list(string.ascii_lowercase)
    first_key = next(iter(rep_dict))    # deal with cases where there isn't 0 in samples
    coef_types = list(rep_dict[first_key].keys())  # alpha, beta
    event_types = rep_dict[first_key][coef_types[0]].keys()
    mapping = {t: i for i, t in enumerate(coef_types)}
    fig, axes = plt.subplots(len(coef_types), len(event_types), figsize=(12, 10))
    res_dict = {coef: {event_type: None for event_type in event_types} for coef in coef_types}
    for idct, coef_type in enumerate(coef_types):
        for idet, event_type in enumerate(event_types):
            ax = axes[mapping[coef_type]][event_type - 1]
            add_panel_text(ax=ax, text=alphabet_list[idct*len(event_types)+idet])
            ax.tick_params(axis='both', which='major', labelsize=15)
            ax.tick_params(axis='both', which='minor', labelsize=15)
            df = pd.concat([dfs[coef_type][event_type] for dfs in rep_dict.values()])
            temp_df = df.groupby(df.index).agg(["mean", "std"])
            prefix = "a" if coef_type == "alpha" else "Z"
            temp_df = temp_df.loc[[f"{prefix}{idx}_{event_type}" for idx in range(1, temp_df.shape[0]+1)]]
            temp_df.columns = temp_df.columns.get_level_values(0) + "_" + temp_df.columns.get_level_values(1)
            res_dict[coef_type][event_type] = temp_df.copy()
            temp_df.plot(x="Lee_std", y="Ours_std", kind="scatter", ax=ax)
            ax.set_xlabel("Lee et al. std", fontsize=18)
            ax.set_ylabel("two-step std", fontsize=18)
            ax.plot([0, 1], [0, 1], "--", transform=ax.transAxes, alpha=0.3, color="tab:green");
            ax.grid()
            if paper_plots:
                if ((idct == 0) and (idet == 0)):
                    ax.set_xlim([0, 0.2])
                    ax.set_ylim([0, 0.2])
                elif ((idct == 0) and (idet == 1)):
                    ax.set_xlim([0, 0.25])
                    ax.set_ylim([0, 0.25])
                elif ((idct == 1) and (idet == 0)):
                    ax.set_xlim([0.025, 0.035])
                    ax.set_ylim([0.025, 0.035])
                elif ((idct == 1) and (idet == 1)):
                    ax.set_xlim([0.035, 0.05])
                    ax.set_ylim([0.035, 0.05])
            latter = "\\alpha" if coef_type == "alpha" else "\\beta"
            ax.set_title(f"${latter}{event_type}$", fontsize=18)
    fig.tight_layout()
    fig.show()
    if filename is not None:
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)

    if return_summary:
        return res_dict


def plot_times(times_dict: dict,
               filename: str = None,
               ax = None, color='tab:blue') -> None:
    if ax is None:

        ax = pd.DataFrame.from_dict(times_dict).boxplot(figsize=(8, 6), boxprops={"lw": 1.5, "color": "tab:blue"},
                                                        medianprops={"lw": 2, "color": "tab:green"},
                                                        flierprops={'markeredgecolor': "tab:blue"})
    else:
        pd.DataFrame.from_dict(times_dict).boxplot(boxprops={"lw": 1.5, "color": color},
                                                   flierprops={'markeredgecolor': color},
                                                   medianprops={"lw": 2, "color": color}, ax=ax)

    # ax.set_ylim(0, 15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.set_ylabel("Fitting Time [seconds]", fontdict={"size": 18}) # "weight": 'bold',
    ax.set_xlabel("Model type", fontdict={"size": 18}) # "weight": 'bold',

    #ax.tick_params(labelsize=14, grid_lw=0.5, grid_alpha=0.6)
    plt.tight_layout()
    plt.show()
    if filename is not None:
        ax.figure.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    return ax


def plot_cif_plots(pred_df: pd.DataFrame, event: str, return_ax: bool = False, ax: plt.Axes = None,
                   pad: float = 0.15, scale: int = 5) -> None:
    """
    this method plot cif given pred df with cif and event
    """
    cif_cols = pred_df.columns[pred_df.columns.str.startswith("cif")]

    event_cif_cols = cif_cols[cif_cols.str.contains(f"j{event}")]

    event_x = event_cif_cols.str.extract(r"(t\d+)")[0].str.extract((r"(\d+)")).apply(pd.to_numeric).values.flatten()
    if ax is None:
        ax = pred_df[event_cif_cols].T.plot(figsize=(10, 10))
    else:
        pred_df[event_cif_cols].T.plot(ax=ax)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.set_xticks(event_x)
    ax.set_xticklabels(event_x)
    y_min, y_max = get_y_perc_limits(pred_df, event_cif_cols, pad=pad, scale=scale)
    ax.set_ylim(bottom=y_min, top=y_max)
    ax.set_xlabel("t", fontdict={'size': 18})  # , "weight": "bold"
    ax.set_ylabel(f"CIF of $J_{event}$", fontdict={'size': 18})  # , "weight": "bold"
    ax.grid()
    if return_ax:
        return ax
    else:
        plt.tight_layout()
        plt.show()


def scale_perc_limits(num: float, scale: int, up: bool = False):
    func = np.ceil if up else np.floor
    return (func((num * 100) / scale) * scale) / 100


def get_y_perc_limits(df: pd.DataFrame, cols: Iterable, pad: float = 0.15, scale: int = 5) -> Tuple[float, float]:
    y_min = max(df[cols].min().min() - pad, 0)
    y_min = scale_perc_limits(y_min, scale=scale, up=False)

    y_max = min(df[cols].max().max() + pad, 1)
    y_max = scale_perc_limits(y_max, scale=scale, up=True)
    return y_min, y_max


def plot_events_occurrence(patients_df: pd.DataFrame, ax: plt.Axes = None, event_type_col: str = 'J',
                           pid_col: str = 'pid', event_time_col: str = 'X', fname: str = None):
    if ax is None:
        tight = True
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    else:
        tight = False
    patients_df.groupby([event_type_col, event_time_col])[pid_col].count().unstack('J').fillna(0).plot(ax=ax,
                                                                                                       kind='bar',
                                                                                                       width=0.8)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=15)
    ax.set_xlabel(f"Time", fontdict={'size': 18})
    ax.set_ylabel(f"Number of Observations", fontdict={'size': 18})
    if tight:
        fig.tight_layout()
    if fname is not None:
        fig.savefig(fname, dpi=300)
    return ax


def plot_example_pred_output(pred_df, fname: str = None):
    gs = gridspec.GridSpec(4, 4)
    ax1 = plt.subplot(gs[0, 0:2])
    ax2 = plt.subplot(gs[0, 2:])
    ax3 = plt.subplot(gs[1, 0:2])
    ax4 = plt.subplot(gs[1, 2:])
    ax5 = plt.subplot(gs[2, 0:2])
    ax6 = plt.subplot(gs[2, 2:])
    ax7 = plt.subplot(gs[3, 1:3])
    fig = plt.gcf()
    fig.set_size_inches(14, 14)
    ax_lst = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

    titles = ['Hazard', 'Probability', 'CIF', 'Overall Survival']
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    for idp, pref in enumerate(['hazard', 'prob', 'cif', 'overall_survival']):
        for ide, event in enumerate(['j1', 'j2']):
            if idp*2+ide > 6:
                break
            ax = ax_lst[idp*2+ide]
            add_panel_text(ax, letters[idp*2+ide])
            for patient in pred_df.columns:
                times = [m for m in range(1, 31)]
                if pref == 'overall_survival':
                    index_val = [f'{pref}_t{m}' for m in times]
                elif pref == 'hazard':
                    index_val = [f'{pref}_{event}_t{m}' for m in times]
                else:
                    index_val = [f'{pref}_{event}_at_t{m}' for m in times]
                tmp = pred_df.loc[index_val, patient]
                if pref in ['cif', 'overall_survival']:
                    ax.step(tmp.index, tmp.values, label=patient)
                else:
                    ax.plot(tmp.index, tmp.values, label=patient)
            ax.set_xticks(range(len(times)))
            ax.set_xticklabels(times, rotation=90)
            ax.set_xlabel('Time', fontsize=15)
            if idp*2+ide < 6:
                ax.set_ylabel(f'{titles[idp]}   {event}', fontsize=15)
            else:
                ax.set_ylabel(f'{titles[idp]}', fontsize=15)
                
            ax.tick_params(axis='both', which='major', labelsize=14)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            ax.legend()
        
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname, dpi=300)


def plot_example_estimated_params(fitter, fname: str = None):
    fig, axes = plt.subplots(1,2, figsize=(14,7))
    ax = axes[0]
    fitter.plot_all_events_alpha(ax=ax, show=False)
    ax.grid()
    ax.legend(fontsize=16, loc='center right')
    add_panel_text(ax, 'a')
    ax = axes[1]
    fitter.plot_all_events_beta(ax=ax, show=False, xlabel='Value')
    ax.legend(fontsize=16, loc='center right')
    add_panel_text(ax, 'b')
    fig.tight_layout()
    if fname is not None:
        fig.savefig(fname, dpi=300)