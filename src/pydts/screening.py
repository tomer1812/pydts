from .fitters import TwoStagesFitter, TwoStagesFitterExact
import pandas as pd
import numpy as np
from typing import Optional, List, Union
import psutil
from .utils import get_expanded_df
from joblib import Parallel, delayed


WORKERS = psutil.cpu_count(logical=False)


class MarginalTwoStagesFitter(TwoStagesFitter):

    def fit(self,
            expanded_df: pd.DataFrame,
            covariates: List = None,
            event_type_col: str = 'J',
            duration_col: str = 'X',
            pid_col: str = 'pid',
            x0: Union[np.array, int] = 0,
            fit_beta_kwargs: dict = {},
            verbose: int = 2,
            nb_workers: int = WORKERS) -> dict:
        """
        This method estimates only the parameters of the covariates (beta_j) without the time parameters (alpha_jt).
        Note that the expanded discrete-time data is expected as an input (see the Methods section of PyDTS documentation and pydts.utils.get_expanded_df).

        Args:
            expanded_df (pd.DataFrame): expanded training data for fitting the model
            covariates (list): list of covariates to be used in estimating the regression coefficients
            event_type_col (str): The event type column name (must be a column in df),
                                  Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).
            x0 (Union[numpy.array, int], Optional): initial guess to pass to scipy.optimize.minimize function
            fit_beta_kwargs (dict, Optional): Keyword arguments to pass on to the estimation procedure.
                                              If different model for beta is desired, it can be defined here.
                                              For example:
                                              fit_beta_kwargs={
                                                    model=CoxPHFitter, # model object
                                                    model_kwargs={},  # keywords arguments to pass on to the model instance initiation
                                                    model_fit_kwargs={}  # keywords arguments to pass on to model.fit() method
                                              }
            verbose (int, Optional): The verbosity level of pandaallel
            nb_workers (int, Optional): The number of workers to pandaallel. If not sepcified, defaults to the number of workers available.
        Returns:
            event_models (dict): Fitted models dictionary. Keys - event names, Values - fitted models for the event.
        """

        self._validate_cols(expanded_df, event_type_col, duration_col, pid_col)
        self.events = [c for c in sorted(expanded_df[event_type_col].unique()) if c != 0]

        if covariates is None:
            covariates = [col for col in expanded_df if col not in [event_type_col, duration_col, pid_col]]
        self.covariates = covariates
        self.event_type_col = event_type_col
        self.duration_col = duration_col
        self.pid_col = pid_col
        self.times = sorted(expanded_df[duration_col].unique())

        self.expanded_df = expanded_df

        self.beta_models = self._fit_beta(expanded_df, self.events, **fit_beta_kwargs)

        return self.beta_models


class MarginalTwoStagesFitterExact(TwoStagesFitterExact):

    def fit(self,
            expanded_df: pd.DataFrame,
            covariates: List = None,
            event_type_col: str = 'J',
            duration_col: str = 'X',
            pid_col: str = 'pid',
            x0: Union[np.array, int] = 0,
            fit_beta_kwargs: dict = {},
            verbose: int = 2,
            nb_workers: int = WORKERS) -> dict:
        """
        This method estimates only the parameters of the covariates (beta_j) without the time parameters (alpha_jt).
        Note that the expanded discrete-time data is expected as an input (see the Methods section of PyDTS documentation and pydts.utils.get_expanded_df).

        Args:
            expanded_df (pd.DataFrame): expanded training data for fitting the model
            covariates (list): list of covariates to be used in estimating the regression coefficients
            event_type_col (str): The event type column name (must be a column in df),
                                  Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).
            x0 (Union[numpy.array, int], Optional): initial guess to pass to scipy.optimize.minimize function
            fit_beta_kwargs (dict, Optional): Keyword arguments to pass on to the estimation procedure.
                                              If different model for beta is desired, it can be defined here.
                                              For example:
                                              fit_beta_kwargs={
                                                    model=CoxPHFitter, # model object
                                                    model_kwargs={},  # keywords arguments to pass on to the model instance initiation
                                                    model_fit_kwargs={}  # keywords arguments to pass on to model.fit() method
                                              }
            verbose (int, Optional): The verbosity level of pandaallel
            nb_workers (int, Optional): The number of workers to pandaallel. If not sepcified, defaults to the number of workers available.
        Returns:
            event_models (dict): Fitted models dictionary. Keys - event names, Values - fitted models for the event.
        """

        self._validate_cols(expanded_df, event_type_col, duration_col, pid_col)
        self.events = [c for c in sorted(expanded_df[event_type_col].unique()) if c != 0]

        if covariates is None:
            covariates = [col for col in expanded_df if col not in [event_type_col, duration_col, pid_col]]
        self.covariates = covariates
        self.event_type_col = event_type_col
        self.duration_col = duration_col
        self.pid_col = pid_col
        self.times = sorted(expanded_df[duration_col].unique())

        self.expanded_df = expanded_df

        self.beta_models = self._fit_beta(expanded_df, self.events, **fit_beta_kwargs)

        return self.beta_models


class BaseSISTwoStages(object):

    """
    This class implements the principled sure independence screening (PSIS) process of Zhao et al. (2012) for discrete-time data using the TwoStagesFitter and data-driven threshold.
    """

    def __init__(self):
        self.threshold = None
        self.marginal_estimates_df = pd.DataFrame()
        self.df = pd.DataFrame()
        self.expanded_df = pd.DataFrame()
        self.permuted_df = pd.DataFrame()
        self.permuted_expanded_df = pd.DataFrame()
        self.events = None
        self.covariates = None
        self.event_type_col = None
        self.duration_col = None
        self.pid_col = None
        self.times = None
        self.null_model_df = None
        self.final_model = None
        self.chosen_covariates_j = None
        self.chosen_covariates = None
        self.TwoStagesFitter_type = ""

    def fit_marginal_model(self,
                           expanded_df,
                           covariate: str,
                           event_type_col: str = 'J',
                           duration_col: str = 'X',
                           pid_col: str = 'pid',
                           x0: Union[np.array, int] = 0,
                           fit_beta_kwargs: dict = {},
                           verbose: int = 2,
                           nb_workers: int = 1):
        """
        This method fits a marginal model to data using a single covariate.
        Note that the expanded discrete-time data is expected as an input (see the Methods section of PyDTS documentation and pydts.utils.get_expanded_df).

        Args:
            expanded_df (pd.DataFrame): expanded training data for fitting the model
            covariate (str): a single covariate to be used in estimating the regression coefficients
            event_type_col (str): The event type column name (must be a column in df),
                                  Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).
            x0 (Union[numpy.array, int], Optional): initial guess to pass to scipy.optimize.minimize function
            fit_beta_kwargs (dict, Optional): Keyword arguments to pass on to the estimation procedure.
                                              If different model for beta is desired, it can be defined here.
                                              For example:
                                              fit_beta_kwargs={
                                                    model=CoxPHFitter, # model object
                                                    model_kwargs={},  # keywords arguments to pass on to the model instance initiation
                                                    model_fit_kwargs={}  # keywords arguments to pass on to model.fit() method
                                              }
            verbose (int, Optional): The verbosity level of pandaallel
            nb_workers (int, Optional): The number of workers to pandaallel. If not sepcified, defaults to the number of workers available.
        Returns:
            result (pd.DataFrame): Estimated parameter and standard errors. TwoStagesFitter.get_beta_SE() output.
        """

        if self.events is None:
            self.events = [c for c in sorted(expanded_df[event_type_col].unique()) if c != 0]

        if self.TwoStagesFitter_type == 'Exact':
            marginal_model = MarginalTwoStagesFitterExact()
        else:
            marginal_model = MarginalTwoStagesFitter()

        marginal_model.fit(
            expanded_df=expanded_df[[pid_col, covariate, event_type_col, duration_col, 'j_0'] +
                                    [f'j_{e}' for e in self.events]],
            covariates=[covariate],
            event_type_col=event_type_col,
            duration_col=duration_col,
            pid_col=pid_col,
            x0=x0,
            fit_beta_kwargs=fit_beta_kwargs,
            verbose=verbose,
            nb_workers=nb_workers)

        result = marginal_model.get_beta_SE()
        del marginal_model
        return result

    def get_marginal_estimates(self,
                               expanded_df,
                               covariates: Union[List, dict] = None,
                               event_type_col: str = 'J',
                               duration_col: str = 'X',
                               pid_col: str = 'pid',
                               verbose: int = 2,
                               x0: Union[np.array, int] = 0,
                               fit_beta_kwargs: dict = {},
                               nb_workers: int = WORKERS):

        """
        This method fits a marginal model to data to each of the covariates.
        Note that the expanded discrete-time data is expected as an input (see the Methods section of PyDTS documentation and pydts.utils.get_expanded_df).

        Args:
            expanded_df (pd.DataFrame): expanded training data for fitting the model
            covariates (list): list of covariates to estimate the marginal regression coefficient for.
            event_type_col (str): The event type column name (must be a column in df),
                                  Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).
            verbose (int, Optional): The verbosity level of pandaallel
            x0 (Union[numpy.array, int], Optional): initial guess to pass to scipy.optimize.minimize function
            fit_beta_kwargs (dict, Optional): Keyword arguments to pass on to the estimation procedure.
                                              If different model for beta is desired, it can be defined here.
                                              For example:
                                              fit_beta_kwargs={
                                                    model=CoxPHFitter, # model object
                                                    model_kwargs={},  # keywords arguments to pass on to the model instance initiation
                                                    model_fit_kwargs={}  # keywords arguments to pass on to model.fit() method
                                              }
            nb_workers (int, Optional): The number of workers to pandaallel. If not sepcified, defaults to the number of workers available.
        Returns:
            results_df (pd.DataFrame): Estimated parameters and standard errors of the marginal models. A concatenation of all the TwoStagesFitter.get_beta_SE() outputs.
        """

        if self.events is None:
            self.events = [c for c in sorted(expanded_df[event_type_col].unique()) if c != 0]

        if covariates is None:
            covariates = [col for col in expanded_df if col not in ([event_type_col, duration_col, pid_col, 'j_0'] +
                          [f'j_{e}' for e in self.events])]

        parallel = Parallel(n_jobs=nb_workers, verbose=verbose)
        results_df = pd.DataFrame()
        if isinstance(covariates, list):
            _results = parallel(delayed(self.fit_marginal_model)(expanded_df, cov,
                                                                 event_type_col, duration_col, pid_col,
                                                                 x0, fit_beta_kwargs, verbose, nb_workers)
                                                                 for cov in covariates)
            results_df = pd.concat(_results)
        elif isinstance(covariates, dict):
            for event in self.events:
                _results = parallel(delayed(self.fit_marginal_model)(expanded_df, cov,
                                                                     event_type_col, duration_col, pid_col,
                                                                     x0, fit_beta_kwargs, verbose, nb_workers)
                                                                     for cov in covariates[event])
                event_results_df = pd.concat(_results)
                results_df = pd.concat([results_df, event_results_df], axis=1)

        return results_df.astype(float)

    def permute_df(self,
                   df,
                   event_type_col: str = 'J',
                   duration_col: str = 'X',
                   pid_col: str = 'pid',
                   seed: int = None):

        """
        This method applies random permutation on the event-time and event-type columns of the training data such that the covariates are decoupled from the outcome; the permuted data follow the null model.

        Args:
            df (pd.DataFrame): training data for fitting the model
            event_type_col (str): The event type column name (must be a column in df),
                                  Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).

            seed (int, Optional): pseudo random state.
        Returns:
            permuted_df (pd.DataFrame): null model data.
        """

        permuted_df = df.copy()
        np.random.seed(seed)
        permuted_index = np.random.permutation(permuted_df.index)
        permuted_df.loc[:, duration_col] = df.loc[permuted_index, duration_col].values
        permuted_df.loc[:, event_type_col] = df.loc[permuted_index, event_type_col].values
        self.permuted_df = permuted_df
        self.permuted_expanded_df = get_expanded_df(df=self.permuted_df,
                                                    event_type_col=event_type_col,
                                                    duration_col=duration_col,
                                                    pid_col=pid_col)
        return permuted_df

    def get_data_driven_threshold(self,
                                  df,
                                  covariates: List = None,
                                  quantile: float = 1,
                                  event_type_col: str = 'J',
                                  duration_col: str = 'X',
                                  pid_col: str = 'pid',
                                  x0: Union[np.array, int] = 0,
                                  fit_beta_kwargs: dict = {},
                                  verbose: int = 2,
                                  nb_workers: int = WORKERS,
                                  seed: int = None):

        """
        This method calculates a data-driven threshold for each risk. It fits marginal models to the permuted data and returns the required quantile of the absolute values of the coefficients estimated from the null model.

        Args:
            df (pd.DataFrame): training data for fitting the model
            covariates (list): list of covariates to estimate the marginal regression coefficient for.
            quantile (float): represents the quantile of the absolute values of the coefficients from the null model that determines the data-driven threshold.
                              Defaults to 1, which corresponds to the maximum absolute value of the null model's coefficients.
            event_type_col (str): The event type column name (must be a column in df),
                                  Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).
            x0 (Union[numpy.array, int], Optional): initial guess to pass to scipy.optimize.minimize function
            fit_beta_kwargs (dict, Optional): Keyword arguments to pass on to the estimation procedure.
                                              If different model for beta is desired, it can be defined here.
                                              For example:
                                              fit_beta_kwargs={
                                                    model=CoxPHFitter, # model object
                                                    model_kwargs={},  # keywords arguments to pass on to the model instance initiation
                                                    model_fit_kwargs={}  # keywords arguments to pass on to model.fit() method
                                              }
            verbose (int, Optional): The verbosity level of pandaallel
            nb_workers (int, Optional): The number of workers to pandaallel. If not sepcified, defaults to the number of workers available.
            seed (int): pseudo random state.
        Returns:
            threshold (pd.Series): Estimated thresholds.
        """

        if self.events is None:
            self.events = [c for c in sorted(df[event_type_col].unique()) if c != 0]
        if covariates is None:
            covariates = [col for col in df if col not in [event_type_col, duration_col, pid_col]]

        self.permute_df(df=df, event_type_col=event_type_col,
                        duration_col=duration_col, pid_col=pid_col, seed=seed)
        self.null_model_df = self.get_marginal_estimates(expanded_df=self.permuted_expanded_df,
                                                         covariates=covariates,
                                                         event_type_col=event_type_col,
                                                         duration_col=duration_col,
                                                         pid_col=pid_col,
                                                         verbose=verbose,
                                                         x0=x0,
                                                         fit_beta_kwargs=fit_beta_kwargs,
                                                         nb_workers=nb_workers)

        _params_cols = self._get_params_cols_from_res_df(self.null_model_df)
        self.threshold = np.quantile(self.null_model_df[_params_cols].abs().values, q=quantile)
        return self.threshold

    def fit(self,
            df: pd.DataFrame,
            threshold: float = None,
            quantile: float = 1,
            covariates: List = None,
            event_type_col: str = 'J',
            duration_col: str = 'X',
            pid_col: str = 'pid',
            x0: Union[np.array, int] = 0,
            fit_beta_kwargs: dict = {},
            verbose: int = 2,
            nb_workers: int = WORKERS,
            seed: int = None,
            fit_final_model: bool = True):

        """
        This method performs the principled sure independence screening (PSIS) process of Zhao et al. (2012) for discrete-time data with data-driven threshold.

        Args:
            df (pd.DataFrame): training data for fitting the model
            threshold (float): a user defined threshold.
                               Defaults to None, i.e. data-driven threshold
            quantile (float): the quantile of the absolute values of the coefficients from the null model that determines the data-driven threshold.
                              Only in use when threshold = None.
                              Defaults to 1, which corresponds to the maximum absolute value of the null model's coefficients.
            covariates (list): list of covariates to estimate the marginal regression coefficient for.
            event_type_col (str): The event type column name (must be a column in df),
                                  Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).
            x0 (Union[numpy.array, int], Optional): initial guess to pass to scipy.optimize.minimize function
            fit_beta_kwargs (dict, Optional): Keyword arguments to pass on to the estimation procedure.
                                              If different model for beta is desired, it can be defined here.
                                              For example:
                                              fit_beta_kwargs={
                                                    model=CoxPHFitter, # model object
                                                    model_kwargs={},  # keywords arguments to pass on to the model instance initiation
                                                    model_fit_kwargs={}  # keywords arguments to pass on to model.fit() method
                                              }
            verbose (int, Optional): The verbosity level of pandaallel
            nb_workers (int, Optional): The number of workers to pandaallel. If not sepcified, defaults to the number of workers available.
            seed (int): pseudo random state.
            fit_final_model (boolean): True if to fit and return the TwoStagesFitter with the selected covariates.
        Returns:
            final_model (TwoStagesFitter): estimated model with the chosen covariates after PSIS.
        """

        self.events = [c for c in sorted(df[event_type_col].unique()) if c != 0]
        if covariates is None:
            covariates = [col for col in df if col not in [event_type_col, duration_col, pid_col]]
        self.covariates = covariates
        self.event_type_col = event_type_col
        self.duration_col = duration_col
        self.pid_col = pid_col
        self.times = sorted(df[duration_col].unique())

        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = self.get_data_driven_threshold(df=df,
                                                            covariates=covariates,
                                                            quantile=quantile,
                                                            event_type_col=event_type_col,
                                                            duration_col=duration_col,
                                                            pid_col=pid_col,
                                                            x0=x0,
                                                            fit_beta_kwargs=fit_beta_kwargs,
                                                            verbose=verbose,
                                                            nb_workers=nb_workers,
                                                            seed=seed)
        self.df = df
        self.expanded_df = get_expanded_df(df=df,
                                           event_type_col=event_type_col,
                                           duration_col=duration_col,
                                           pid_col=pid_col)

        self.marginal_estimates_df = self.get_marginal_estimates(expanded_df=self.expanded_df,
                                                                 covariates=covariates,
                                                                 event_type_col=event_type_col,
                                                                 duration_col=duration_col,
                                                                 pid_col=pid_col,
                                                                 verbose=verbose,
                                                                 x0=x0,
                                                                 fit_beta_kwargs=fit_beta_kwargs,
                                                                 nb_workers=nb_workers)

        chosen_covariates = []
        chosen_covariates_j = {}

        _params_cols = self._get_params_cols_from_res_df(self.marginal_estimates_df)

        for c in _params_cols:
            if self.TwoStagesFitter_type == 'Exact':
                event = int(c[0])
            else:
                event = int(c[1:].split('_')[0])
            chosen_covariates_j[event] = self.marginal_estimates_df[self.marginal_estimates_df[c].abs() >= self.threshold].index.tolist()
            chosen_covariates.extend(chosen_covariates_j[event])

        self.chosen_covariates = sorted(np.unique(chosen_covariates))
        self.chosen_covariates_j = chosen_covariates_j

        if fit_final_model:
            if self.TwoStagesFitter_type == 'Exact':
                self.final_model = TwoStagesFitterExact()
            else:
                self.final_model = TwoStagesFitter()
            self.final_model.fit(df=df,
                                 covariates=self.chosen_covariates_j,
                                 event_type_col=event_type_col,
                                 duration_col=duration_col,
                                 pid_col=pid_col,
                                 x0=x0,
                                 fit_beta_kwargs=fit_beta_kwargs,
                                 verbose=verbose,
                                 nb_workers=nb_workers)

        return self.final_model

    def _get_params_cols_from_res_df(self, res_df):
        if self.TwoStagesFitter_type == 'Exact':
            _params_cols = [c for c in res_df.columns if '   coef   ' in c]
        else:
            _params_cols = [c for c in res_df.columns if 'params' in c]
        return _params_cols


class SISTwoStagesFitter(BaseSISTwoStages):

    def __init__(self):
        super().__init__()
        self.TwoStagesFitter_type = 'CoxPHFitter'


class SISTwoStagesFitterExact(BaseSISTwoStages):

    def __init__(self):
        super().__init__()
        self.TwoStagesFitter_type = 'Exact'