import pandas as pd
import numpy as np
import warnings
pd.set_option("display.max_rows", 500)
warnings.filterwarnings('ignore')
slicer = pd.IndexSlice
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter
from typing import Optional, List, Union


# Weights
def event_specific_weights(pred_df: pd.DataFrame,
                           event: int,
                           event_type_col: str = 'J',
                           duration_col: str = 'X') -> pd.Series:
    """
    This function implements the calculation of the event specific time-weights.

    Args:
        pred_df (pd.DataFrame): Data to calculate prediction error.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for the event.
                                See TwoStagesFitter.predict_prob_events()
        event (int): Event-type to calculate the weights for.
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        result (pd.Series): event specific weights.
    """

    event_df = pred_df[pred_df[event_type_col] == event]
    if len(event_df) == 0:
        print(f'Could not calculate weights for event {event} - no test events')
        return np.nan
    weights = event_df.groupby(duration_col).size() / event_df.groupby(duration_col).size().sum()
    times = sorted(pred_df[duration_col].unique())[:-1]
    return weights.reindex(times).fillna(0)


# BRIER SCORE
def event_specific_brier_score_at_t(pred_df: pd.DataFrame,
                                    event: int,
                                    t: int,
                                    event_type_col: str = 'J',
                                    duration_col: str = 'X') -> float:
    """
    This function implements the calculation of the event specific Brier Score at time t.

    Args:
        pred_df (pd.DataFrame): Data to calculate Brier Score for.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for all events.
                                See TwoStagesFitter.predict_prob_events()
        event (int): Event-type to calculate the Brier Score for.
        t (int): time to calculate the Brier Score for.
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        result (pd.Series): event specific Brier Score at time t.
    """

    pi_ij = pred_df.loc[:, f'prob_j{event}_at_t{t}']
    D_ij = ((pred_df.loc[:, event_type_col] == event) & (pred_df.loc[:, duration_col] == t)).astype(int)
    censoring_kmf = KaplanMeierFitter()
    censoring_kmf.fit(durations=pred_df[duration_col], event_observed=(pred_df[event_type_col] == 0))
    in_risk_set_at_t = (pred_df.loc[:, duration_col] >= t).astype(int)
    W_ij = (in_risk_set_at_t / censoring_kmf.predict(times=t))
    BS_jt = ((W_ij*((D_ij - pi_ij)**2)).sum() / in_risk_set_at_t.sum())
    return BS_jt


def event_specific_brier_score_at_t_all(pred_df: pd.DataFrame,
                                        event: int,
                                        event_type_col: str = 'J',
                                        duration_col: str = 'X') -> pd.Series:
    """
    This function implements the calculation of the event specific Brier Score at time t for all times included in duration_col of pred_df.

    Args:
        pred_df (pd.DataFrame): Data to calculate Brier Score.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for the event.
                                See c
        event (int): Event-type to calculate the Brier Score for.
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        result (pd.Series): event specific Brier Score for all times included in duration_col of pred_df.
    """

    res = {}
    for t in sorted(pred_df[duration_col].unique())[:-1]:
        res[t] = event_specific_brier_score_at_t(pred_df=pred_df, event=event, t=t,
                                                 event_type_col=event_type_col, duration_col=duration_col)
    return pd.Series(res, name=event)


def event_specific_integrated_brier_score(pred_df: pd.DataFrame,
                                          event: int,
                                          event_type_col: str = 'J',
                                          duration_col: str = 'X',
                                          weights: Union[pd.Series, None] = None) -> float:
    """
    This function implements the calculation of the event specific integrated Brier Score.

    Args:
        pred_df (pd.DataFrame): Data to calculate Brier Score.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for the event.
                                See TwoStagesFitter.predict_prob_events()
        event (int): Event-type to calculate the integrated Brier Score for.
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
        weights (pd.Series): Optional. Weights vector with time as index and weight as value. Length must be the number of possible event times.
    Returns:
        result (float): integrated Brier Score results.
    """

    brier_score_at_t = event_specific_brier_score_at_t_all(pred_df=pred_df, event=event,
                                                           event_type_col=event_type_col,
                                                           duration_col=duration_col)

    if brier_score_at_t.isnull().any():
        print(f'There are NaN values in BS(t) during Integrated Brier Score calculation for event {event}. Times: {brier_score_at_t[brier_score_at_t.isnull()].index}\n')

    if weights is None:
        weights = event_specific_weights(pred_df=pred_df, event=event,
                                         event_type_col=event_type_col,
                                         duration_col=duration_col)

    result = brier_score_at_t.dot(weights.sort_index())
    return result


def global_brier_score(pred_df: pd.DataFrame,
                       event_type_col: str = 'J',
                       duration_col: str = 'X') -> float:
    """
    This function implements the calculation of the global Brier Score.

    Args:
        pred_df (pd.DataFrame): Data to calculate Brier score.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for all events.
                                See TwoStagesFitter.predict_prob_events()
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        global_auc (float): global Brier Score results.
    """

    e_j_ser = pred_df[pred_df[event_type_col] != 0].groupby('J').size().sort_index()
    total_e = e_j_ser.sum()
    global_bs = 0
    for event, e_j in e_j_ser.iteritems():
        global_bs += (e_j / total_e) * event_specific_integrated_brier_score(
            pred_df=pred_df, event=event, event_type_col=event_type_col,
            duration_col=duration_col)
    return global_bs


def events_integrated_brier_score(pred_df: pd.DataFrame, event_type_col: str = 'J',
                                  duration_col: str ='X') -> dict:
    """
    This function implements the calculation of the integrated Brier Score to all events.

    Args:
        pred_df (pd.DataFrame): Data to calculate integrated Brier Score.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for all events.
                                See TwoStagesFitter.predict_prob_events()
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        integrated_brier_score (dict): integrated Brier Score results.
    """

    events = [e for e in pred_df[event_type_col].unique() if e != 0]
    integrated_brier_score = {}
    for event in sorted(events):
        integrated_brier_score[event] = event_specific_integrated_brier_score(pred_df=pred_df,
                                                                              event=event,
                                                                              event_type_col=event_type_col,
                                                                              duration_col=duration_col)
    return integrated_brier_score


def events_brier_score_at_t(pred_df: pd.DataFrame,
                            event_type_col: str = 'J',
                            duration_col: str ='X') -> pd.DataFrame:
    """
    This function implements the calculation of the events Brier score at t.

    Args:
        pred_df (pd.DataFrame): Data to calculate prediction error.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for all events.
                                See TwoStagesFitter.predict_prob_events()
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        event_brier_score_at_t_df (pd.DataFrame): events Brier score at t results.
    """

    events = [e for e in pred_df[event_type_col].unique() if e != 0]
    event_brier_score_at_t_df = pd.DataFrame()
    for event in sorted(events):
        event_brier_score_at_t_df = pd.concat([event_brier_score_at_t_df,
                                               event_specific_brier_score_at_t_all(pred_df=pred_df,
                                                                                   event=event,
                                                                                   event_type_col=event_type_col,
                                                                                   duration_col=duration_col)],
                                              axis=1)
    return event_brier_score_at_t_df.T


# AUC
def events_auc_at_t(pred_df: pd.DataFrame,
                    event_type_col: str = 'J',
                    duration_col: str ='X') -> pd.DataFrame:
    """
    This function implements the calculation of the events AUC at t.

    Args:
        pred_df (pd.DataFrame): Data to calculate AUC.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for all events.
                                See TwoStagesFitter.predict_prob_events()
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        event_auc_at_t_df (pd.DataFrame): events AUC at t results.
    """

    events = [e for e in pred_df[event_type_col].unique() if e != 0]
    event_auc_at_t_df = pd.DataFrame()
    for event in sorted(events):
        event_auc_at_t_df = pd.concat([event_auc_at_t_df,
                                       event_specific_auc_at_t_all(pred_df=pred_df,
                                                                   event=event,
                                                                   event_type_col=event_type_col,
                                                                   duration_col=duration_col)],
                                       axis=1)
    return event_auc_at_t_df.T


def event_specific_auc_at_t(pred_df: pd.DataFrame,
                            event: int,
                            t: int,
                            event_type_col: str = 'J',
                            duration_col: str = 'X') -> float:
    """
    This function implements the calculation of the event specific AUC at time t.

    Args:
        pred_df (pd.DataFrame): Data to calculate prediction error.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for the event.
                                See TwoStagesFitter.predict_prob_events()
        event (int): Event-type to calculate the integrated AUC for.
        t (int): time to calculate the AUC for.
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        result (pd.Series): event specific AUC for all times included in duration_col of pred_df.
    """

    event_observed_at_t_df = pred_df[(pred_df[event_type_col] == event) & (pred_df[duration_col] == t)]
    no_event_at_t_df = pred_df[pred_df[duration_col] >= t]
    no_event_at_t_df = no_event_at_t_df[~((no_event_at_t_df[event_type_col] == event) &
                                          (no_event_at_t_df[duration_col] == t))]
    total_t = (len(event_observed_at_t_df)*len(no_event_at_t_df))
    if total_t == 0:
        print(f'AUC could not be calculated for event {event} at time {t} - no test pairs of with and without observed event {event} at time {t}')
        return np.nan
    correct_order = 0
    for i_idx, i_row in event_observed_at_t_df.iterrows():
        pi_ij = i_row.loc[f'prob_j{event}_at_t{t}']
        pi_mj = no_event_at_t_df.loc[:, f'prob_j{event}_at_t{t}']
        correct_order += ((pi_ij > pi_mj).sum()+0.5*(pi_ij == pi_mj).sum())
    return correct_order / total_t


def event_specific_auc_at_t_all(pred_df: pd.DataFrame,
                                event: int,
                                event_type_col: str = 'J',
                                duration_col: str = 'X') -> pd.Series:
    """
    This function implements the calculation of the event specific AUC at time t for all times included in duration_col of pred_df.

    Args:
        pred_df (pd.DataFrame): Data to calculate prediction error.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for the event.
                                See TwoStagesFitter.predict_prob_events()
        event (int): Event-type to calculate the AUC for.
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        result (pd.Series): event specific AUC for all times included in duration_col of pred_df.
    """

    res = {}
    for t in sorted(pred_df[duration_col].unique())[:-1]:
        res[t] = event_specific_auc_at_t(pred_df=pred_df, event=event, t=t,
                                         event_type_col=event_type_col, duration_col=duration_col)
    return pd.Series(res, name=event)


def event_specific_integrated_auc(pred_df: pd.DataFrame,
                                  event: int,
                                  event_type_col: str = 'J',
                                  duration_col: str = 'X',
                                  weights: Union[pd.Series, None] = None) -> float:
    """
    This function implements the calculation of the event specific integrated auc.

    Args:
        pred_df (pd.DataFrame): Data to calculate prediction error.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for the event.
                                See TwoStagesFitter.predict_prob_events()
        event (int): Event-type to calculate the integrated AUC for.
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
        weights (pd.Series): Optional. Weights vector with time as index and weight as value. Length must be the number of possible event times.
    Returns:
        result (float): integrated AUC results.
    """

    auc_at_t = event_specific_auc_at_t_all(pred_df=pred_df, event=event,
                                           event_type_col=event_type_col,
                                           duration_col=duration_col)

    if auc_at_t.isnull().any():
        print(f'There are NaN values in AUC(t) during Integrated AUC calculation for event {event}. Times: {auc_at_t[auc_at_t.isnull()].index}\n')

    if weights is None:
        if auc_at_t.isnull().any():
            print(f'Please check there are events of type {event} at each time in pred_df or provide a weights vector with weight 0 for the problematic times.')
            return np.nan
        weights = event_specific_weights(pred_df=pred_df, event=event,
                                         event_type_col=event_type_col,
                                         duration_col=duration_col)

    result = auc_at_t.dot(weights.sort_index())
    return result


def global_auc(pred_df: pd.DataFrame,
               event_type_col: str = 'J',
               duration_col: str = 'X') -> float:
    """
    This function implements the calculation of the global AUC.

    Args:
        pred_df (pd.DataFrame): Data to calculate prediction error.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for all events.
                                See TwoStagesFitter.predict_prob_events()
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        global_auc (float): global AUC results.
    """

    e_j_ser = pred_df[pred_df[event_type_col] != 0].groupby('J').size().sort_index()
    total_e = e_j_ser.sum()
    global_auc = 0
    for event, e_j in e_j_ser.iteritems():
        global_auc += (e_j / total_e) * event_specific_integrated_auc(
            pred_df=pred_df, event=event, event_type_col=event_type_col,
            duration_col=duration_col)
    return global_auc


def events_integrated_auc(pred_df: pd.DataFrame, event_type_col: str = 'J',
                          duration_col: str ='X') -> dict:
    """
    This function implements the calculation of the integrated AUC to all events.

    Args:
        pred_df (pd.DataFrame): Data to calculate prediction error.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for all events.
                                See TwoStagesFitter.predict_prob_events()
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        integrated_auc (dict): integrated AUC results.
    """

    events = [e for e in pred_df[event_type_col].unique() if e != 0]
    integrated_auc = {}
    for event in sorted(events):
        integrated_auc[event] = event_specific_integrated_auc(pred_df=pred_df,
                                                              event=event,
                                                              event_type_col=event_type_col,
                                                              duration_col=duration_col)
    return integrated_auc
