import pandas as pd
import numpy as np

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
    # todo: consider dealing with highly not continues cases
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
    """

    Args:
        first_model:
        second_model:
        real_values:
        event:
        first_model_label:
        second_model_label:

    Returns:

    """
    event_suffix = f"_{event}"
    assert (first_model.index == second_model.index).all(), "All index should be the same"
    models = pd.concat([first_model.to_frame(first_model_label),
                        second_model.to_frame(second_model_label)], axis=1)
    models.index += event_suffix
    real_values_s = pd.Series(real_values, index=models.index)

    return pd.concat([models, real_values_s.to_frame("real")], axis=1)


#todo: move from here
def present_coefs(res_dict):
    from IPython.display import display
    for coef_type, events_dict in res_dict.items():
        print(f"for coef: {coef_type.capitalize()}")
        df = pd.concat([temp_df for temp_df in events_dict.values()])
        display(df)
