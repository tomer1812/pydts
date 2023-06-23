from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import statsmodels.api as sm
from .base_fitters import ExpansionBasedFitter
from scipy.optimize import minimize
from scipy.special import logit, expit
import numpy as np
import pandas as pd
import psutil
from lifelines.fitters.coxph_fitter import CoxPHFitter
from pandarallel import pandarallel
from typing import Optional, List, Union
from matplotlib import colors as mcolors
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings("ignore")

from .examples_utils.generate_simulations_data import generate_quick_start_df
from .utils import assert_fit

COLORS = list(mcolors.TABLEAU_COLORS.keys())
WORKERS = psutil.cpu_count(logical=False)


class DataExpansionFitter(ExpansionBasedFitter):
    """
    This class implements the estimation procedure of Lee et al. (2018) [1].
    See also the Example section.

    Example:
        ```py linenums="1"
            from pydts.fitters import DataExpansionFitter
            fitter = DataExpansionFitter()
            fitter.fit(df=train_df, event_type_col='J', duration_col='X')
            fitter.print_summary()
        ```

    References:
        [1] Lee, Minjung and Feuer, Eric J. and Fine, Jason P., "On the analysis of discrete time competing risks data", Biometrics (2018) doi: 10.1111/biom.12881
    """

    def __init__(self):
        super().__init__()
        self.models_kwargs = dict(family=sm.families.Binomial())

    def _fit_event(self, model_fit_kwargs={}):
        """
        This method fits a model for a GLM model for a specific event.

        Args:
            model_fit_kwargs (dict, Optional): Keyword arguments to pass to model.fit() method.

        Returns:
            fitted GLM model
        """
        model = sm.GLM.from_formula(formula=self.formula, data=self.expanded_df, **self.models_kwargs)
        return model.fit(**model_fit_kwargs)

    def fit(self,
            df: pd.DataFrame,
            event_type_col: str = 'J',
            duration_col: str = 'X',
            pid_col: str = 'pid',
            covariates: Optional[list] = None,
            formula: Optional[str] = None,
            models_kwargs: Optional[dict] = None,
            model_fit_kwargs: Optional[dict] = {}) -> dict:
        """
        This method fits a model to the discrete data.

        Args:
            df (pd.DataFrame): training data for fitting the model
            event_type_col (str): The event type column name (must be a column in df),
                                  Right censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).
            covariates (list, Optional): A list of covariates, all must be columns in df.
                                         Defaults to all the columns of df except event_type_col, duration_col, and pid_col.
            formula (str, Optional): Model formula to be fitted. Patsy format string.
            models_kwargs (dict, Optional): Keyword arguments to pass to model instance initiation.
            model_fit_kwargs (dict, Optional): Keyword arguments to pass to model.fit() method.

        Returns:
            event_models (dict): Fitted models dictionary. Keys - event names, Values - fitted models for the event.
        """

        if models_kwargs is not None:
            self.models_kwargs = models_kwargs

        if 'C' in df.columns:
            raise ValueError('C is an invalid column name, to avoid errors with categorical symbol C() in formula')
        self._validate_cols(df, event_type_col, duration_col, pid_col)
        if covariates is not None:
            cov_not_in_df = [cov for cov in covariates if cov not in df.columns]
            if len(cov_not_in_df) > 0:
                raise ValueError(f"Error during fit - missing covariates from df: {cov_not_in_df}")

        self.events = [c for c in sorted(df[event_type_col].unique()) if c != 0]
        self.covariates = [col for col in df if col not in [event_type_col, duration_col, pid_col]] \
                          if covariates is None else covariates
        self.times = sorted(df[duration_col].unique())

        self.expanded_df = self._expand_data(df=df, event_type_col=event_type_col, duration_col=duration_col,
                                             pid_col=pid_col)
        for event in self.events:
            cov = ' + '.join(self.covariates)
            _formula = f'j_{event} ~ {formula}' if formula is not None else \
                f'j_{event} ~ {cov} + C({duration_col}) -1 '
            self.formula = _formula
            self.event_models[event] = self._fit_event(model_fit_kwargs=model_fit_kwargs)
        return self.event_models

    def print_summary(self,
                      summary_func: str = "summary",
                      summary_kwargs: dict = {}) -> None:
        """
        This method prints the summary of the fitted models for all the events.

        Args:
            summary_func (str, Optional): print summary method of the fitted model type ("summary", "print_summary").
            summary_kwargs (dict, Optional): Keyword arguments to pass to the model summary function.

        Returns:
            None
        """
        for event, model in self.event_models.items():
            _summary_func = getattr(model, summary_func, None)
            if _summary_func is not None:
                print(f'\n\nModel summary for event: {event}')
                print(_summary_func(**summary_kwargs))
            else:
                print(f'Not {summary_func} function in event {event} model')

    def predict_hazard_jt(self,
                          df: pd.DataFrame,
                          event: Union[str, int],
                          t: Union[Iterable, int],
                          n_jobs: int = -1) -> pd.DataFrame:
        """
        This method calculates the hazard for the given event at the given time values if they were included in
        the training set of the event.

        Args:
            df (pd.DataFrame): samples to predict for
            event (Union[str, int]): event name
            t (np.array): times to calculate the hazard for
            n_jobs: number of CPUs to use, defualt to every available CPU
        Returns:
            df (pd.DataFrame): samples with the prediction columns
        """
        t = self._validate_t(t, return_iter=True)
        assert event in self.events, \
            f"Cannot predict for event {event} - it was not included during .fit()"
        self._validate_covariates_in_df(df.head())

        _t = np.array([t_i for t_i in t if (f'hazard_j{event}_t{t_i}' not in df.columns)])
        if len(_t) == 0:
            return df

        temp_df = df.copy()
        model = self.event_models[event]
        res = Parallel(n_jobs=n_jobs)(delayed(model.predict)(df[self.covariates].assign(X=c)) for c in t)
        temp_hazard_df = pd.concat(res, axis=1)
        temp_df[[f'hazard_j{event}_t{c_}' for c_ in t]] = temp_hazard_df.values
        return temp_df

    def get_beta_SE(self):
        """
        This function returns the Beta coefficients and their Standard Errors for all the events.

        Returns:
            se_df (pandas.DataFrame): Beta coefficients and Standard Errors Dataframe

        """

        full_table = pd.DataFrame()
        for event in self.events:
            summary = self.event_models[event].summary()
            summary_df = pd.DataFrame([x.split(',') for x in summary.tables[1].as_csv().split('\n')])
            summary_df.columns = summary_df.iloc[0]
            summary_df = summary_df.iloc[1:].set_index(summary_df.columns[0])
            summary_df.columns = pd.MultiIndex.from_product([[event], summary_df.columns])
            full_table = pd.concat([full_table, summary_df.iloc[-len(self.covariates):]], axis=1)
        return full_table

    def get_alpha_df(self):
        """
        This function returns the Alpha coefficients and their Standard Errors for all the events.

        Returns:
            se_df (pandas.DataFrame): Alpha coefficients and Standard Errors Dataframe

        """

        full_table = pd.DataFrame()
        for event in self.events:
            summary = self.event_models[event].summary()
            summary_df = pd.DataFrame([x.split(',') for x in summary.tables[1].as_csv().split('\n')])
            summary_df.columns = summary_df.iloc[0]
            summary_df = summary_df.iloc[1:].set_index(summary_df.columns[0])
            summary_df.columns = pd.MultiIndex.from_product([[event], summary_df.columns])
            full_table = pd.concat([full_table, summary_df.iloc[:-len(self.covariates)-1]], axis=1)
        return full_table


class TwoStagesFitter(ExpansionBasedFitter):

    """
    This class implements the approach of Meir et al. (2022):

    Example:
        ```py linenums="1"
            from pydts.fitters import TwoStagesFitter
            fitter = TwoStagesFitter()
            fitter.fit(df=train_df, event_type_col='J', duration_col='X')
            fitter.print_summary()
        ```

    References:
        [1] Meir, Tomer\*, Gutman, Rom\*, and Gorfine, Malka, "PyDTS: A Python Package for Discrete-Time Survival Analysis with Competing Risks" (2022)
    """

    def __init__(self):
        super().__init__()
        self.alpha_df = pd.DataFrame()
        self.beta_models = {}

    def _alpha_jt(self, x, df, y_t, beta_j, n_jt, t):
        # Alpha_jt optimization objective
        partial_df = df[df[self.duration_col] >= t]
        expit_add = np.dot(partial_df[self.covariates], beta_j)
        return ((1 / y_t) * np.sum(expit(x + expit_add)) - (n_jt / y_t)) ** 2

    def _fit_event_beta(self, expanded_df, event, model=CoxPHFitter, model_kwargs={}, model_fit_kwargs={}):
        # Model fitting for conditional estimation of Beta_j for specific event
        strata_df = expanded_df[self.covariates + [f'j_{event}', self.duration_col]]
        strata_df[f'{self.duration_col}_copy'] = np.ones_like(expanded_df[self.duration_col])

        beta_j_model = model(**model_kwargs)
        beta_j_model.fit(df=strata_df[self.covariates + [f'{self.duration_col}', f'{self.duration_col}_copy', f'j_{event}']],
                         duration_col=f'{self.duration_col}_copy', event_col=f'j_{event}', strata=self.duration_col,
                         **model_fit_kwargs, batch_mode=False)
        return beta_j_model

    def _fit_beta(self, expanded_df, events, model=CoxPHFitter, model_kwargs={}, model_fit_kwargs={}):
        # Model fitting for conditional estimation of Beta_j for all events
        _model_kwargs_per_event = np.any([event in model_kwargs.keys() for event in events])
        beta_models = {}
        for event in events:
            _model_kwargs = model_kwargs[event] if _model_kwargs_per_event else model_kwargs
            beta_models[event] = self._fit_event_beta(expanded_df=expanded_df, event=event,
                                                      model=model, model_kwargs=_model_kwargs,
                                                      model_fit_kwargs=model_fit_kwargs)
        return beta_models

    def fit(self,
            df: pd.DataFrame,
            covariates: List = None,
            event_type_col: str = 'J',
            duration_col: str = 'X',
            pid_col: str = 'pid',
            x0: Union[np.array, int] = 0,
            fit_beta_kwargs: dict = {},
            verbose: int = 2,
            nb_workers: int = WORKERS) -> dict:
        """
        This method fits a model to the discrete data.

        Args:
            df (pd.DataFrame): training data for fitting the model
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

        self._validate_cols(df, event_type_col, duration_col, pid_col)
        if covariates is not None:
            cov_not_in_df = [cov for cov in covariates if cov not in df.columns]
            if len(cov_not_in_df) > 0:
                raise ValueError(f"Error during fit - missing covariates from df: {cov_not_in_df}")

        pandarallel.initialize(verbose=verbose, nb_workers=nb_workers)
        self.events = [c for c in sorted(df[event_type_col].unique()) if c != 0]
        if covariates is None:
            covariates = [col for col in df if col not in [event_type_col, duration_col, pid_col]]
        self.covariates = covariates
        self.event_type_col = event_type_col
        self.duration_col = duration_col
        self.pid_col = pid_col
        self.times = sorted(df[duration_col].unique())

        expanded_df = self._expand_data(df=df, event_type_col=event_type_col, duration_col=duration_col,
                                        pid_col=pid_col)

        self.beta_models = self._fit_beta(expanded_df, self.events, **fit_beta_kwargs)

        y_t = (df[duration_col]
               .value_counts()
               .sort_index(ascending=False)  # each event count for its occurring time and the times before
               .cumsum()
               .sort_index()
               )
        n_jt = df.groupby([event_type_col, duration_col]).size().to_frame().reset_index()
        n_jt.columns = [event_type_col, duration_col, 'n_jt']

        for event in self.events:
            n_et = n_jt[n_jt[event_type_col] == event]
            n_et['opt_res'] = n_et.parallel_apply(lambda row: minimize(self._alpha_jt, x0=x0,
                                    args=(df, y_t.loc[row[duration_col]], self.beta_models[event].params_, row['n_jt'],
                                    row[duration_col]), method='BFGS',
                                    options={'gtol': 1e-7, 'eps': 1.5e-08, 'maxiter': 200}), axis=1)
            n_et['success'] = n_et['opt_res'].parallel_apply(lambda val: val.success)
            n_et['alpha_jt'] = n_et['opt_res'].parallel_apply(lambda val: val.x[0])
            assert_fit(n_et, self.times[:-1], event_type_col=event_type_col, duration_col=duration_col)  # todo move basic input validation before any optimization
            self.event_models[event] = [self.beta_models[event], n_et]
            self.alpha_df = pd.concat([self.alpha_df, n_et], ignore_index=True)
        return self.event_models

    def print_summary(self,
                      summary_func: str = "print_summary",
                      summary_kwargs: dict = {}) -> None:
        """
        This method prints the summary of the fitted models for all the events.

        Args:
            summary_func (str, Optional): print summary method of the fitted model type ("summary", "print_summary").
            summary_kwargs (dict, Optional): Keyword arguments to pass to the model summary function.

        Returns:
            None
        """
        from IPython.display import display
        display(self.get_beta_SE())

        for event, model in self.event_models.items():
            print(f'\n\nModel summary for event: {event}')
            display(model[1].drop('opt_res', axis=1).set_index([self.event_type_col, self.duration_col]))


    def plot_event_alpha(self, event: Union[str, int], ax: plt.Axes = None, scatter_kwargs: dict = {},
                         show=True, title=None, xlabel='t', ylabel=r'$\alpha_{jt}$', fontsize=18,
                         color: str = None, label: str = None, ticklabelsize: int = 15) -> plt.Axes:
        """
        This function plots a scatter plot of the $ alpha_{jt} $ coefficients of a specific event.
        Args:
            event (Union[str, int]): event name
            ax (matplotlib.pyplot.Axes, Optional): ax to use
            scatter_kwargs (dict, Optional): keywords to pass to the scatter function
            show (bool, Optional): if to use plt.show()
            title (str, Optional): axes title
            xlabel (str, Optional): axes xlabel
            ylabel (str, Optional): axes ylabel
            fontsize (int, Optional): axes title, xlabel, ylabel fontsize
            color (str, Optional): color name to use
            label (str, Optional): label name
        Returns:
            ax (matplotlib.pyplot.Axes): output figure
        """

        assert event in self.events, f"Cannot plot event {event} alpha - it was not included during .fit()"

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)
        ax.tick_params(axis='both', which='minor', labelsize=ticklabelsize)
        title = r'$\alpha_{jt}$' + f' for event {event}' if title is None else title
        label = f'{event}' if label is None else label
        color = 'tab:blue' if color is None else color
        alpha_df = self.event_models[event][1]
        ax.scatter(alpha_df[self.duration_col].values, alpha_df['alpha_jt'].values, label=label,
                   color=color, **scatter_kwargs)
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        if show:
            plt.show()
        return ax

    def plot_all_events_alpha(self, ax: plt.Axes = None, scatter_kwargs: dict = {}, colors: list = COLORS,
                              show: bool = True, title: Union[str, None] = None, xlabel: str = 't',
                              ylabel: str = r'$\alpha_{jt}$', fontsize: int = 18, ticklabelsize: int = 15) -> plt.Axes:
        """
        This function plots a scatter plot of the $ alpha_{jt} $ coefficients of all the events.
        Args:
            ax (matplotlib.pyplot.Axes, Optional): ax to use
            scatter_kwargs (dict, Optional): keywords to pass to the scatter function
            colors (list, Optional): colors names
            show (bool, Optional): if to use plt.show()
            title (str, Optional): axes title
            xlabel (str, Optional): axes xlabel
            ylabel (str, Optional): axes ylabel
            fontsize (int, Optional): axes title, xlabel, ylabel fontsize

        Returns:
            ax (matplotlib.pyplot.Axes): output figure
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)
        ax.tick_params(axis='both', which='minor', labelsize=ticklabelsize)
        title = r'$\alpha_{jt}$' + f' for all events' if title is None else title
        for idx, (event, model) in enumerate(self.event_models.items()):
            label = f'{event}'
            color = colors[idx % len(colors)]
            self.plot_event_alpha(event=event, ax=ax, scatter_kwargs=scatter_kwargs, show=False, title=title,
                                  ylabel=ylabel, xlabel=xlabel, fontsize=fontsize, label=label, color=color,
                                  ticklabelsize=ticklabelsize)
        ax.legend()
        if show:
            plt.show()
        return ax

    def predict_hazard_jt(self,
                          df: pd.DataFrame,
                          event: Union[str, int],
                          t: Union[Iterable, int]) -> pd.DataFrame:
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
        self._validate_covariates_in_df(df.head())
        t = self._validate_t(t, return_iter=True)
        assert event in self.events, \
            f"Cannot predict for event {event} - it was not included during .fit()"

        model = self.event_models[event]
        alpha_df = model[1].set_index(self.duration_col)['alpha_jt'].copy()

        _t = np.array([t_i for t_i in t if (f'hazard_j{event}_t{t_i}' not in df.columns)])
        if len(_t) == 0:
            return df
        temp_df = df.copy()
        beta_j_x = temp_df[self.covariates].dot(model[0].params_)
        temp_df[[f'hazard_j{event}_t{c}' for c in _t]] = pd.concat(
            [self._hazard_inverse_transformation(alpha_df[c] + beta_j_x) for c in _t], axis=1).values
        return temp_df

    def _hazard_transformation(self, a: Union[int, np.array, pd.Series, pd.DataFrame]) -> \
            Union[int, np.array, pd.Series, pd.DataFrame]:
        """
        This function defines the transformation of the hazard function such that
        $ h ( \lambda_j (t | Z) ) = \alpha_{jt} + Z^{T} \beta_{j} $

        Args:
            a (Union[int, np.array, pd.Series, pd.DataFrame]):

        Returns:
            i (Union[int, np.array, pd.Series, pd.DataFrame]): the inverse function applied on a. $ h^{-1} (a)$
        """

        i = logit(a)
        return i

    def _hazard_inverse_transformation(self, a: Union[int, np.array, pd.Series, pd.DataFrame]) -> \
            Union[int, np.array, pd.Series, pd.DataFrame]:
        """
        This function defines the inverse transformation of the hazard function such that
        $\lambda_j (t | Z) = h^{-1} ( \alpha_{jt} + Z^{T} \beta_{j} )$

        Args:
            a (Union[int, np.array, pd.Series, pd.DataFrame]):

        Returns:
            i (Union[int, np.array, pd.Series, pd.DataFrame]): the inverse function applied on a. $ h^{-1} (a) $
        """
        i = expit(a)
        return i

    def get_beta_SE(self):
        """
        This function returns the Beta coefficients and their Standard Errors for all the events.

        Returns:
            se_df (pandas.DataFrame): Beta coefficients and Standard Errors Dataframe

        """
        se_df = pd.DataFrame()
        for event, model in self.beta_models.items():
            mdf = pd.concat([model.params_, model.standard_errors_], axis=1)
            mdf.columns = [f'j{event}_params', f'j{event}_SE']
            se_df = pd.concat([se_df, mdf], axis=1)
        return se_df

    def get_alpha_df(self):
        """
        This function returns the Alpha coefficients for all the events.

        Returns:
            alpha_df (pandas.DataFrame): Alpha coefficients Dataframe

        """

        alpha_df = pd.DataFrame()
        for event, model in self.event_models.items():
            model_alpha_df = model[1].drop('opt_res', axis=1).set_index([self.event_type_col, self.duration_col])
            model_alpha_df.columns = pd.MultiIndex.from_product([[event], model_alpha_df.columns])
            alpha_df = pd.concat([alpha_df, model_alpha_df], axis=1)

        return alpha_df

    def plot_all_events_beta(self, ax: plt.Axes = None, colors: list = COLORS, show: bool = True,
                             title: Union[str, None] = None, xlabel: str = 'Value',  ylabel: str = r'$\beta_{j}$',
                             fontsize: int = 18, ticklabelsize: int = 15) -> plt.Axes:
        """
        This function plots the $ beta_{j} $ coefficients and standard errors of all the events.
        Args:
            ax (matplotlib.pyplot.Axes, Optional): ax to use
            colors (list, Optional): colors names
            show (bool, Optional): if to use plt.show()
            title (str, Optional): axes title
            xlabel (str, Optional): axes xlabel
            ylabel (str, Optional): axes ylabel
            fontsize (int, Optional): axes title, xlabel, ylabel fontsize
            ticklabelsize (int, Optional): axes xticklabels, yticklabels fontsize
        Returns:
            ax (matplotlib.pyplot.Axes): output figure
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        title = r'$\beta_{j}$' + f' for all events' if title is None else title
        ax.tick_params(axis='both', which='major', labelsize=ticklabelsize)
        ax.tick_params(axis='both', which='minor', labelsize=ticklabelsize)
        se_df = self.get_beta_SE()

        for idx, col in enumerate(se_df.columns):
            if idx % 2 == 1:
                continue
            y = np.arange((idx//2)*len(se_df), (1+(idx//2))*len(se_df))
            ax.errorbar(x=se_df.iloc[:, idx].values, y=y,
                       color=colors[idx % len(colors)], xerr=se_df.iloc[:, idx+1].values, label=f'{col}',
                       markersize=6, ls='', marker='o')

        yt = list(se_df.index) * (len(se_df.columns) // 2)
        ax.set_yticks(np.arange(0, len(yt)))
        ax.set_yticklabels(yt)
        ax.set_title(title, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.grid()
        plt.gca().invert_yaxis()
        ax.legend()
        if show:
            plt.show()
        return ax


def repetitive_fitters(rep: int, n_patients: int, n_cov: int, d_times: int, j_events: int, pid_col: str,
                       drop_cols: Iterable = ("C", "T"),
                       model1: ExpansionBasedFitter = DataExpansionFitter,
                       model1_name="Lee",
                       model2: ExpansionBasedFitter = TwoStagesFitter,
                       model2_name: str = "Ours",
                       allow_fails: int = 20,
                       verbose: int = 2,
                       real_coef_dict: dict = None,
                       censoring_prob: float = 1.
                       ) -> Tuple[dict, dict, pd.DataFrame]:
    """
    The function allows the user to generate N repetitions of model training (given data generating process),
    to allow the user to compare the parameters stability and fitting time of the methods .

    Args:
        rep (int): number of repetitions to run the models
        n_patients (int): number of sample to generate for each repetition
        n_cov (int): number of covariates to generate for each repetition
        d_times (int): how many times T to generate (i.e. times would be $t \in [1,\ldots,d]$)
        j_events (int): number of events to generate for each repetition
        pid_col (str): the name of the id column
        test_size (float): the test size (percentage) for train/test splitting
        drop_cols (Iterable):
            Columns that shouldn't be visible for the model, from the generated sample.
            default is drop the real event time (T) and censoring time (C)
        model1 (ExpansionBasedFitter): Typically Lee et al. 2018 [1] model (DataExpansionFitter).
            but can be any of base type ExpansionBasedFitter
        model1_name (str): the name of the first model
        model2 (ExpansionBasedFitter): Typically our suggested method (TwoStagesFitter).
            but can be any of base type ExpansionBasedFitter
        model2_name  (str): the name of the second model
        allow_fails (int): number of allowed failed repetitions.
            I.e. the method would run up to rep + allow_fails times.
        verbose (int): verbosity level for the pandarallel module
        real_coef_dict (dict): dictionary which represent the real coefficients to be generated for each repetition
        censoring_prob (float): The probability to use the censoring method for each round of generation

    Returns:
        rep_dict (dict): Dictionary which contains for each round (key)
            its beta/alpha/real coefficients dataframe (value).
        times (dict): Dictionary which  contains for each model (key) a list of the training times (value).
        ret_df (pd.DataFrame): Dataframe which contains for each time in d times, the average amount of events for it.

    """

    from .examples_utils.plots import compare_beta_models_for_example
    from tqdm import trange
    from time import time
    assert real_coef_dict is not None, "The user should supply the coefficients of the experiment"
    rep_dict = {}
    times = {model1_name: [], model2_name: []}
    counts_df_list = []
    final = 0
    failed = 0
    for samp in trange(rep+allow_fails):
        try:
            patients_df = generate_quick_start_df(n_patients=n_patients, n_cov=n_cov, d_times=d_times,
                                                  j_events=j_events,
                                                  pid_col=pid_col, seed=samp, real_coef_dict=real_coef_dict,
                                                  censoring_prob=censoring_prob )
            counts_df = patients_df[patients_df['X'] <= d_times].groupby(['J', 'X']).size().to_frame(samp)
            assert not (counts_df.reset_index()['X'].value_counts() < (j_events + 1)).any(), "Not enough events"
            counts_df_list.append(counts_df)
            drop_cols = pd.Index(drop_cols)
            start_1 = time()
            fitter = model1()
            fitter.fit(df=patients_df.drop(drop_cols, axis=1))
            end_1 = time()
            start_2 = time()
            new_fitter = model2()
            if isinstance(new_fitter, TwoStagesFitter):
                new_fitter.fit(df=patients_df.drop(drop_cols, axis=1), verbose=verbose)
            else:
                new_fitter.fit(df=patients_df.drop(drop_cols, axis=1))
            end_2 = time()
            times[model1_name].append(end_1 - start_1)
            times[model2_name].append(end_2 - start_2)
            res_dict = compare_beta_models_for_example(fitter.event_models,
                                                       new_fitter.event_models, real_coef_dict=real_coef_dict)
            rep_dict[samp] = res_dict
            final += 1
            if final == rep:
                break
        except Exception as e:
            print(e)
            failed += 1
            print(f'Failed to fit sample {samp+1}, fail #{failed}')
            continue
    print(f'final: {final}')
    ret_df = pd.concat(counts_df_list, axis=1).fillna(0).mean(axis=1).apply(np.ceil).to_frame()
    return rep_dict, times, ret_df
