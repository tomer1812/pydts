import pandas as pd
import numpy as np
import warnings
pd.set_option("display.max_rows", 500)
warnings.filterwarnings('ignore')
slicer = pd.IndexSlice
from lifelines.fitters.kaplan_meier_fitter import KaplanMeierFitter


def event_specific_prediction_error(pred_df: pd.DataFrame,
                                    event: int,
                                    event_type_col: str = 'J',
                                    duration_col: str = 'X') -> pd.Series:
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
    return pd.Series(res)


def event_specific_auc_weights(pred_df: pd.DataFrame,
                               event: int,
                               event_type_col: str = 'J',
                               duration_col: str = 'X') -> pd.Series:
    """
    This function implements the calculation of the event specific AUC weights.

    Args:
        pred_df (pd.DataFrame): Data to calculate prediction error.
                                Must contain the observed duration and event-type, and the probability of event at time t prediction results for the event.
                                See TwoStagesFitter.predict_prob_events()
        event (int): Event-type to calculate the AUC weights for.
        duration_col (str): Last follow up time column name (must be a column in pred_df).
        event_type_col (str): The event type column name (must be a column in df),
                              Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
    Returns:
        result (pd.Series): event specific AUC weights.
    """

    kmf_overall = KaplanMeierFitter()
    kmf_overall.fit(durations=pred_df[duration_col], event_observed=(pred_df[event_type_col] > 0))
    kmf_event = KaplanMeierFitter()
    kmf_event.fit(durations=pred_df[duration_col], event_observed=(pred_df[event_type_col] == event))
    prob_jt = np.abs(-1*kmf_event.survival_function_.diff()[1:])  # abs to fix negative zero
    overall_t_minus_1 = kmf_overall.survival_function_.iloc[:-1]
    numer = (prob_jt.values.squeeze()) * (overall_t_minus_1.values.squeeze())
    denom = (prob_jt.values.squeeze()).dot(overall_t_minus_1.values.squeeze())
    result = pd.Series(numer / denom, index=prob_jt.index).iloc[:-1]
    return result


def event_specific_integrated_auc(pred_df: pd.DataFrame,
                                  event: int,
                                  event_type_col: str = 'J',
                                  duration_col: str = 'X') -> float:
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
    Returns:
        result (float): integrated AUC results.
    """

    auc_at_t = event_specific_auc_at_t_all(pred_df=pred_df, event=event,
                                           event_type_col=event_type_col,
                                           duration_col=duration_col)
    weights = event_specific_auc_weights(pred_df=pred_df, event=event,
                                         event_type_col=event_type_col,
                                         duration_col=duration_col)
    result = auc_at_t.dot(weights)
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

    e_j_ser = pred_df.groupby('J').size().sort_index().iloc[1:]
    total_e = e_j_ser.sum()
    global_auc = 0
    for event, e_j in e_j_ser.iteritems():
        global_auc += (e_j / total_e) * event_specific_integrated_auc(
            pred_df=pred_df, event=event, event_type_col=event_type_col,
            duration_col=duration_col)
    return global_auc
