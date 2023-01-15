import pandas as pd
import numpy as np
import warnings
pd.set_option("display.max_rows", 500)
warnings.filterwarnings('ignore')
slicer = pd.IndexSlice
from typing import Optional, List, Union


def event_specific_prediction_error(pred_df: pd.DataFrame,
                                    event: int,
                                    event_type_col: str = 'J',
                                    duration_col: str = 'X') -> pd.DataFrame:
    """
    This function implements the calculation of the cause-specific prediction error (PE).

    Args:
        pred_df (pd.DataFrame): Data to calculate prediction error.
                                Must contain the observed duration and event-type, and the cumulative incident function prediction results.
                                See TwoStagesFitter.predict_cumulative_incident_function()
        event (int): Event-type to calculate the PE for.
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        res (pd.DataFrame): cause-specific PE results.
    """

    cif_event_columns = [c for c in pred_df.columns if f'cif_j{event}_' in c]
    cif_df = pred_df[sorted(cif_event_columns, key=lambda c: int(c.split('_at_t')[1]))]
    ind_df = pd.DataFrame(data=np.zeros(cif_df.shape),
                          columns=list(range(1, cif_df.shape[1]+1)),
                          index=pred_df.index)
    for idx_row, row in pred_df.iterrows():
        if row[event_type_col] == event:
            ind_df.loc[idx_row, row[duration_col]:] = 1
        elif row[event_type_col] == 0:
            ind_df.loc[idx_row, row[duration_col] + 1:] = np.nan
    prediction_error = (ind_df - cif_df.values)**2
    risk_set_size = prediction_error.notnull().sum(axis=0)
    res = pd.Series(prediction_error.sum(axis=0) / risk_set_size,
                    index=list(range(1, cif_df.shape[1]+1)),
                    name=event)
    res.index.name = 'event'
    return res


def prediction_error(pred_df: pd.DataFrame,
                     event_type_col: str = 'J',
                     duration_col: str ='X') -> pd.DataFrame:
    """
    This function implements the calculation of the total prediction error (PE).

    Args:
        pred_df (pd.DataFrame): Data to calculate prediction error.
                                Must contain the observed duration and event-type, and the cumulative incident function prediction results.
                                See TwoStagesFitter.predict_cumulative_incident_function()
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        prediction_error (pd.DataFrame): PE results.
    """

    events = [e for e in pred_df[event_type_col].unique() if e != 0]
    prediction_error = pd.DataFrame()
    for event in sorted(events):
        prediction_error = pd.concat([prediction_error,
                                      event_specific_prediction_error(pred_df=pred_df,
                                                                      event=event,
                                                                      event_type_col=event_type_col,
                                                                      duration_col=duration_col)],
                                     axis=1)
    return prediction_error.T
