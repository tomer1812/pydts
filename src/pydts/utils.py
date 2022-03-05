import pandas as pd


def get_expanded_df(df, event_type_col='J', time_col='X', pid_col='pid'):
    """
    This function gets a dataframe describing each sample the time of the observed events,
    and returns an expanded dataframe as explained in TODO add reference
    Right censoring is allowed and must be marked as event type 0.

    :param df: original dataframe (pd.DataFrame)
    :param event_type_col: event type column name (str)
    :param time_col: time column name (str)
    :param pid_col: patient id column name (str)

    :return: result_df: expanded dataframe
    """
    result_df = df.reindex(df.index.repeat(df[time_col]))
    result_df[time_col] = result_df.groupby(pid_col).cumcount()+1
    result_df.reset_index(drop=True, inplace=True)
    last_idx = result_df.drop_duplicates(subset=[pid_col], keep='last').index
    events = sorted(df[event_type_col].unique())
    result_df.loc[last_idx, [f'j_{e}' for e in events]] = pd.get_dummies(
        result_df.loc[last_idx, event_type_col]).values
    result_df[[f'j_{e}' for e in events]] = result_df[[f'j_{e}' for e in events]].fillna(0)
    result_df[f'j_0'] = 1 - result_df[[f'j_{e}' for e in events if e > 0]].sum(axis=1)
    return result_df