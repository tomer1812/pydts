from typing import Iterable, Optional

import pandas as pd
import numpy as np
from scipy.special import expit


def get_expanded_df(df, event_type_col='J', duration_col='X', pid_col='pid'):
    """
    This function gets a dataframe describing each sample the time of the observed events,
    and returns an expanded dataframe as explained in TODO add reference
    Right censoring is allowed and must be marked as event type 0.

    :param df: original dataframe (pd.DataFrame)
    :param event_type_col: event type column name (str)
    :param duration_col: time column name (str)
    :param pid_col: patient id column name (str)

    :return: result_df: expanded dataframe
    """
    unique_times = df[duration_col].sort_values().unique()
    result_df = df.reindex(df.index.repeat(df[duration_col]))
    result_df[duration_col] = result_df.groupby(pid_col).cumcount() + 1
    # drop times that didn't happen
    result_df.drop(index=result_df.loc[~result_df[duration_col].isin(unique_times)].index, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    last_idx = result_df.drop_duplicates(subset=[pid_col], keep='last').index
    events = sorted(df[event_type_col].unique())
    result_df.loc[last_idx, [f'j_{e}' for e in events]] = pd.get_dummies(
        result_df.loc[last_idx, event_type_col]).values
    result_df[[f'j_{e}' for e in events]] = result_df[[f'j_{e}' for e in events]].fillna(0)
    result_df[f'j_0'] = 1 - result_df[[f'j_{e}' for e in events if e > 0]].sum(axis=1)
    return result_df


def compare_models_coef_per_event(first_model: pd.Series,
                                  second_model: pd.Series,
                                  real_values: np.array,
                                  event: int,
                                  first_model_label:str = "first",
                                  second_model_label:str = "second"
                                  ) -> pd.DataFrame:
    event_suffix = f"_{event}"
    assert (first_model.index == second_model.index).all(), "All index should be the same"
    models = pd.concat([first_model.to_frame(first_model_label),
                        second_model.to_frame(second_model_label)], axis=1)
    models.index += event_suffix
    real_values_s = pd.Series(real_values, index=models.index)

    return pd.concat([models, real_values_s.to_frame("real")], axis=1)


def present_coefs(res_dict):
    from IPython.display import display
    for coef_type, events_dict in res_dict.items():
        print(f"for coef: {coef_type.capitalize()}")
        df = pd.concat([temp_df for temp_df in events_dict.values()])
        display(df)


def get_real_hazard(df, real_coef_dict, times, events):
    a_t = {event: {t: real_coef_dict['alpha'][event](t) for t in times} for event in events}
    b = pd.concat([df.dot(real_coef_dict['beta'][j]) for j in events], axis=1, keys=events)

    for j in events:
        df[[f'hazard_j{j}_t{t}' for t in times]] = pd.concat([expit(a_t[j][t] + b[j]) for t in times],
                                                             axis=1).values
    return df


def assert_fit(event_df, times, event_type_col='J', duration_col='X'):
    if not event_df['success'].all():
        problematic_times = event_df.loc[~event_df['success'], duration_col].tolist()
        event = event_df[event_type_col].max()  # all the events in the dataframe are the same
        raise RuntimeError(f"Number of observed events at some time points are too small. Consider collapsing neighbor time points."
                           f"\n See https://tomer1812.github.io/pydts/UsageExample-RegroupingData/ for more details.")
    if event_df.shape[0] != len(times):
        event = event_df[event_type_col].max()  # all the events in the dataframe are the same
        problematic_times = pd.Index(event_df[duration_col]).symmetric_difference(times).tolist()
        raise RuntimeError(f"Number of observed events at some time points are too small. Consider collapsing neighbor time points."
                           f"\n See https://tomer1812.github.io/pydts/UsageExample-RegroupingData/ for more details.")


def create_df_for_cif_plots(df: pd.DataFrame, field: str,
                            covariates: Iterable,
                            vals: Optional[Iterable] = None,
                            quantiles: Optional[Iterable] = None,
                            zero_others: Optional[bool] = False
                            ) -> pd.DataFrame:
    """
    This method creates df for cif plot, where it zeros

    Args:
        df (pd.DataFrame): Dataframe which we yield the statiscal propetrics (means, quantiles, etc) and stacture
        field (str): The field which will represent the change
        covariates (Iterable): The covariates of the given model
        vals (Optional[Iterable]): The values to use for the field
        quantiles (Optional[Iterable]): The quantiles to use as values for the field
        zero_others (bool): Whether to zero the other covarites or to zero them

    Returns:
        df (pd.DataFrame): A dataframe that contains records per value for cif ploting
    """

    cov_not_fitted = [cov for cov in covariates if cov not in df.columns]
    assert len(cov_not_fitted) == 0, \
        f"Required covariates are missing from df: {cov_not_fitted}"

    df_for_ploting = df.copy()
    if vals is not None:
        pass
    elif quantiles is not None:
        vals = df_for_ploting[field].quantile(quantiles).values
    else:
        raise NotImplemented("Only Quantiles or specific values is supported")
    temp_series = []
    template_s = df_for_ploting.iloc[0][covariates].copy()
    if zero_others:
        impute_val = 0
    else:
        impute_val = df_for_ploting[covariates].mean().values
    for val in vals:
        temp_s = template_s.copy()
        temp_s[covariates] = impute_val
        temp_s[field] = val
        temp_series.append(temp_s)

    return pd.concat(temp_series, axis=1).T