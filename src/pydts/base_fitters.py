from typing import Iterable, Tuple, Union

import pandas as pd
import numpy as np
from .utils import get_expanded_df



class BaseFitter:
    """
    This class implements the basic fitter methods and attributes api
    """

    def __init__(self):
        self.event_models = {}
        self.expanded_df = pd.DataFrame()
        self.event_type_col = None
        self.duration_col = None
        self.pid_col = None
        self.events = None
        self.covariates = None
        self.formula = None
        self.times = None

    def fit(self, df: pd.DataFrame, event_type_col: str = 'J', duration_col: str = 'X', pid_col: str = 'pid',
            **kwargs) -> dict:
        raise NotImplementedError

    def predict(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def evaluate(self, test_df: pd.DataFrame, oracle_col: str = 'T', **kwargs) -> float:
        raise NotImplementedError

    def print_summary(self, **kwargs) -> None:
        raise NotImplementedError

    def _validate_t(self, t, return_iter=True):
        _t = np.array([t]) if not isinstance(t, Iterable) else t
        t_i_not_fitted = [t_i for t_i in _t if (t_i not in self.times)]
        assert len(t_i_not_fitted) == 0, \
            f"Cannot predict for times which were not included during .fit(): {t_i_not_fitted}"
        if return_iter:
            return _t
        return t

    def _validate_covariates_in_df(self, df):
        cov_not_fitted = [cov for cov in self.covariates if cov not in df.columns]
        assert len(cov_not_fitted) == 0, \
            f"Cannot predict - required covariates are missing from df: {cov_not_fitted}"

    def _validate_cols(self, df, event_type_col, duration_col, pid_col):
        assert event_type_col in df.columns, f'Event type column is missing from df: {event_type_col}'
        assert duration_col in df.columns, f'Duration column is missing from df: {duration_col}'
        assert pid_col in df.columns, f'Observation ID column is missing from df: {pid_col}'


class ExpansionBasedFitter(BaseFitter):
    """
    This class implements the data expansion method which is common for the existing fitters
    """

    def _expand_data(self,
                     df: pd.DataFrame,
                     event_type_col: str,
                     duration_col: str,
                     pid_col: str) -> pd.DataFrame:
        """
        This method expands the raw data as explained in Lee et al. 2018

        Args:
            df (pandas.DataFrame): Dataframe to expand.
            event_type_col (str): The event type column name (must be a column in df),
                                  Right censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).

        Returns:
            Expanded df (pandas.DataFrame): the expanded dataframe.
        """
        self._validate_cols(df, event_type_col, duration_col, pid_col)
        return get_expanded_df(df=df, event_type_col=event_type_col, duration_col=duration_col, pid_col=pid_col)

    def predict_hazard_jt(self, df: pd.DataFrame, event: Union[str, int], t: Union[Iterable, int]) -> pd.DataFrame:
        """
        This method calculates the hazard for the given event at the given time values if they were included in
        the training set of the event.

        Args:
            df (pd.DataFrame): samples to predict for
            event (Union[str, int]): event name
            t (Union[Iterable, int]): times to calculate the hazard for

        Returns:
            df (pd.DataFrame): samples with the prediction columns
        """
        raise NotImplementedError

    def predict_hazard_t(self, df: pd.DataFrame, t: Union[int, np.array]) -> pd.DataFrame:
        """
        This function calculates the hazard for all the events at the requested time values if they were included in
        the training set of each event.

        Args:
            df (pd.DataFrame): samples to predict for
            t (int, np.array): times to calculate the hazard for

        Returns:
            df (pd.DataFrame): samples with the prediction columns
        """
        t = self._validate_t(t)
        self._validate_covariates_in_df(df.head())

        for event, model in self.event_models.items():
            df = self.predict_hazard_jt(df=df, event=event, t=t)
        return df

    def predict_hazard_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function calculates the hazard for all the events at all time values included in the training set for each
        event.

        Args:
            df (pd.DataFrame): samples to predict for

        Returns:
            df (pd.DataFrame): samples with the prediction columns

        """
        self._validate_covariates_in_df(df.head())
        df = self.predict_hazard_t(df, t=self.times)
        return df

    def predict_overall_survival(self,
                                 df: pd.DataFrame,
                                 t: int = None,
                                 return_hazards: bool = False) -> pd.DataFrame:
        """
        This function adds columns of the overall survival until time t.
        Args:
            df (pandas.DataFrame): dataframe with covariates columns
            t (int): time
            return_hazards (bool): if to keep the hazard columns

        Returns:
            df (pandas.DataFrame): dataframe with the additional overall survival columns

        """
        if t is not None:
            self._validate_t(t, return_iter=False)
        self._validate_covariates_in_df(df.head())

        all_hazards = self.predict_hazard_all(df)
        _times = self.times if t is None else [_t for _t in self.times if _t <= t]
        overall = pd.DataFrame()
        for t_i in _times:
            cols = [f'hazard_j{e}_t{t_i}' for e in self.events]
            t_i_hazard = 1 - all_hazards[cols].sum(axis=1)
            t_i_hazard.name = f'overall_survival_t{t_i}'
            overall = pd.concat([overall, t_i_hazard], axis=1)
        overall = pd.concat([df, overall.cumprod(axis=1)], axis=1)

        if return_hazards:
            cols = all_hazards.columns[all_hazards.columns.str.startswith("hazard_")]
            cols = cols.difference(overall.columns)
            if len(cols) > 0:
                overall = pd.concat([overall, all_hazards[cols]], axis=1)
        return overall

    def predict_prob_event_j_at_t(self, df: pd.DataFrame, event: Union[str, int], t: int) -> pd.DataFrame:
        """
        This function adds a column with probability of occurance of a specific event at a specific a time.

        Args:
            df (pandas.DataFrame): dataframe with covariates columns
            event (Union[str, int]): event name
            t (int): time

        Returns:
            df (pandas.DataFrame): dataframe an additional probability column

        """
        assert event in self.events, \
            f"Cannot predict for event {event} - it was not included during .fit()"
        self._validate_t(t, return_iter=False)
        self._validate_covariates_in_df(df.head())

        if f'prob_j{event}_at_t{t}' not in df.columns:
            if t == 1:
                if f'hazard_j{event}_t{t}' not in df.columns:
                    df = self.predict_hazard_jt(df=df, event=event, t=t)
                df[f'prob_j{event}_at_t{t}'] = df[f'hazard_j{event}_t{t}']
                return df
            elif not f'overall_survival_t{t - 1}' in df.columns:
                df = self.predict_overall_survival(df, t=t, return_hazards=True)
            elif not f'hazard_j{event}_t{t}' in df.columns:
                df = self.predict_hazard_t(df, t=np.array([_t for _t in self.times if _t <= t]))
            df[f'prob_j{event}_at_t{t}'] = df[f'overall_survival_t{t - 1}'] * df[f'hazard_j{event}_t{t}']
        return df

    def predict_prob_event_j_all(self, df: pd.DataFrame, event: Union[str, int]) -> pd.DataFrame:
        """
        This function adds columns of a specific event occurrence probabilities.

        Args:
            df (pandas.DataFrame): dataframe with covariates columns
            event (Union[str, int]): event name

        Returns:
            df (pandas.DataFrame): dataframe with probabilities columns

        """
        assert event in self.events, \
            f"Cannot predict for event {event} - it was not included during .fit()"
        self._validate_covariates_in_df(df.head())

        if f'overall_survival_t{self.times[-1]}' not in df.columns:
            df = self.predict_overall_survival(df, return_hazards=True)
        for t in self.times:
            if f'prob_j{event}_at_t{t}' not in df.columns:
                df = self.predict_prob_event_j_at_t(df=df, event=event, t=t)
        return df

    def predict_prob_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function adds columns of all the events occurance probabilities.
        Args:
            df (pandas.DataFrame): dataframe with covariates columns

        Returns:
            df (pandas.DataFrame): dataframe with probabilities columns

        """
        self._validate_covariates_in_df(df.head())

        for event in self.events:
            df = self.predict_prob_event_j_all(df=df, event=event)
        return df

    def predict_event_cumulative_incident_function(self, df: pd.DataFrame, event: Union[str, int]) -> pd.DataFrame:
        """
        This function adds a specific event columns of the predicted hazard function, overall survival, probabilities
        of event occurance and cumulative incident function (CIF) to the given dataframe.

        Args:
            df (pandas.DataFrame): dataframe with covariates columns included
            event (Union[str, int]): event name

        Returns:
            df (pandas.DataFrame): dataframe with additional prediction columns

        """
        assert event in self.events, \
            f"Cannot predict for event {event} - it was not included during .fit()"
        self._validate_covariates_in_df(df.head())

        if f'prob_j{event}_at_t{self.times[-1]}' not in df.columns:
            df = self.predict_prob_events(df=df)
        cols = [f'prob_j{event}_at_t{t}' for t in self.times]
        cif_df = df[cols].cumsum(axis=1)
        cif_df.columns = [f'cif_j{event}_at_t{t}' for t in self.times]
        df = pd.concat([df, cif_df], axis=1)
        return df

    def predict_cumulative_incident_function(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function adds columns of the predicted hazard function, overall survival, probabilities of event occurance
        and cumulative incident function (CIF) to the given dataframe.

        Args:
            df (pandas.DataFrame): dataframe with covariates columns included

        Returns:
            df (pandas.DataFrame): dataframe with additional prediction columns

        """
        self._validate_covariates_in_df(df.head())

        for event in self.events:
            if f'cif_j{event}_at_t{self.times[-1]}' not in df.columns:
                df = self.predict_event_cumulative_incident_function(df=df, event=event)
        return df

    def predict_marginal_prob_event_j(self, df: pd.DataFrame, event: Union[str, int]) -> pd.DataFrame:
        """
        This function calculates the marginal probability of an event given the covariates.

        Args:
            df (pandas.DataFrame): dataframe with covariates columns included
            event (Union[str, int]): event name

        Returns:
            df (pandas.DataFrame): dataframe with additional prediction columns
        """

        assert event in self.events, \
            f"Cannot predict for event {event} - it was not included during .fit()"
        self._validate_covariates_in_df(df.head())

        if f'prob_j{event}_at_t{self.times[-1]}' not in df.columns:
            df = self.predict_prob_event_j_all(df=df, event=event)
        cols = [f'prob_j{event}_at_t{_t}' for _t in self.times]
        marginal_prob = df[cols].sum(axis=1)
        marginal_prob.name = f'marginal_prob_j{event}'
        return pd.concat([df, marginal_prob], axis=1)

    def predict_marginal_prob_all_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function calculates the marginal probability per event given the covariates for all the events.

        Args:
            df (pandas.DataFrame): dataframe with covariates columns included

        Returns:
            df (pandas.DataFrame): dataframe with additional prediction columns
        """
        self._validate_covariates_in_df(df.head())
        for event in self.events:
            df = self.predict_marginal_prob_event_j(df=df, event=event)
        return df
