import pandas as pd


def get_expanded_df(df, event_type_col='J', time_col='X', pid_col='pid'):
    '''
    Expand df
    '''
    result_df = df.reindex(df.index.repeat(df[time_col]))
    result_df[time_col] = result_df.groupby(level=0).cumcount()+1
    last_idx = result_df.drop_duplicates(subset=[pid_col], keep='last').index
    events = sorted(df[event_type_col].unique())
    result_df.loc[last_idx, [f'd_{e}' for e in events]] = pd.get_dummies(
        result_df.loc[last_idx, event_type_col]).values
    result_df[[f'd_{e}' for e in events]] = result_df[[f'd_{e}' for e in events]].fillna(0)
    result_df[f'd_0'] = 1 - result_df[[f'd_{e}' for e in events if e > 0]].sum(axis=1)
    return result_df