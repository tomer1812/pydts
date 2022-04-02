from typing import Iterable, Tuple, Union
from time import time

import matplotlib.pyplot as plt
import statsmodels.api as sm
from pydts.base_fitters import ExpansionBasedFitter
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from scipy.special import logit, expit
import numpy as np
import pandas as pd
from lifelines.fitters.coxph_fitter import CoxPHFitter
from pandarallel import pandarallel
from typing import Optional, List, Union
from matplotlib import colors as mcolors
from tqdm import tqdm
from joblib import Parallel, delayed
from pydts.examples_utils.generate_simulations_data import generate_quick_start_df


COLORS = list(mcolors.TABLEAU_COLORS.keys())


class DataExpansionFitter(ExpansionBasedFitter):
    """
    This class implements the fitter as described in Lee et al. 2018 [1]
    See also Simple Example section.

    Example:
        ```py linenums="1"
            from pydts.fitters import DataExpansionFitter
            fitter = DataExpansionFitter()
            fitter.fit(df=train_df, event_type_col='J', duration_col='X')
            fitter.print_summary()
        ```

    References:
        [1] "On the Analysis of Discrete Time Competing Risks Data", Lee et al., Biometrics, 2018, DOI: 10.1111/biom.12881
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
        # todo : make it more general for both classes
        t = self._validate_t(t, return_iter=True)
        assert event in self.events, \
            f"Cannot predict for event {event} - it was not included during .fit()"
        self._validate_covariates_in_df(df.head())

        _t = np.array([t_i for t_i in t if (f'hazard_j{event}_t{t_i}' not in df.columns)])
        if len(_t) == 0:
            return df

        temp_df = df.copy()  # todo make sure .copy() is required
        model = self.event_models[event]
        res = Parallel(n_jobs=n_jobs)(delayed(model.predict)(df[self.covariates].assign(X=c)) for c in t)
        temp_hazard_df = pd.concat(res, axis=1)
        temp_df[[f'hazard_j{event}_t{c_}' for c_ in t]] = temp_hazard_df.values
        return temp_df


class TwoStagesFitter(ExpansionBasedFitter):

    """
    This class implements the new approach for fitting model to discrete time survival data.
    See also Simple Example section.

    Example:
        ```py linenums="1"
            from pydts.fitters import TwoStagesFitter
            fitter = TwoStagesFitter()
            fitter.fit(df=train_df, event_type_col='J', duration_col='X')
            fitter.print_summary()
        ```
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
        strata_df[f'{self.duration_col}_copy'] = expanded_df[self.duration_col]
        beta_j_model = model(**model_kwargs)
        beta_j_model.fit(df=strata_df[self.covariates + [f'{self.duration_col}', f'{self.duration_col}_copy', f'j_{event}']],
                         duration_col=self.duration_col, event_col=f'j_{event}', strata=f'{self.duration_col}_copy',
                         **model_fit_kwargs)
        return beta_j_model

    def _fit_beta(self, expanded_df, events, model=CoxPHFitter, model_kwargs={}, model_fit_kwargs={}):
        # Model fitting for conditional estimation of Beta_j for all events
        beta_models = {}
        for event in events:
            beta_models[event] = self._fit_event_beta(expanded_df=expanded_df, event=event,
                                                      model=model, model_kwargs=model_kwargs,
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
            verbose: int = 2
            ) -> dict:
        """
        This method fits a model to the discrete data.

        Args:
            df (pd.DataFrame): training data for fitting the model
            covariates (list): list of covariates to be used in fitting the beta model
            event_type_col (str): The event type column name (must be a column in df),
                                  Right censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).
            x0 (Union[numpy.array, int], Optional): initial guess to pass to scipy.optimize.minimize function
            fit_beta_kwargs (dict, Optional): Keyword arguments to pass on to fit beta procedure.
                                              If different model for beta is desired, it can be defined here.
                                              For example:
                                              fit_beta_kwargs={
                                                    model=CoxPHFitter, # model object
                                                    model_kwargs={},  # keywords arguments to pass on to the model instance initiation
                                                    model_fit_kwargs={}  # keywords arguments to pass on to model.fit() method
                                              }
            verbose (int, Optional): The verbosity level of pandaallel
        Returns:
            event_models (dict): Fitted models dictionary. Keys - event names, Values - fitted models for the event.
        """

        self._validate_cols(df, event_type_col, duration_col, pid_col)
        if covariates is not None:
            cov_not_in_df = [cov for cov in covariates if cov not in df.columns]
            if len(cov_not_in_df) > 0:
                raise ValueError(f"Error during fit - missing covariates from df: {cov_not_in_df}")

        pandarallel.initialize(verbose=verbose)
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
                                    row[duration_col])), axis=1)
            n_et['success'] = n_et['opt_res'].parallel_apply(lambda val: val.success)
            n_et['alpha_jt'] = n_et['opt_res'].parallel_apply(lambda val: val.x[0])
            assert_fit(n_et, self.times)  # todo move basic input validation before any optimization
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
        for event, model in self.event_models.items():
            _summary_func = getattr(model[0], summary_func, None)
            if _summary_func is not None:
                print(f'\n\nModel summary for event: {event}')
                _summary_func(**summary_kwargs)
            else:
                print(f'Not {summary_func} function in event {event} model')
            from IPython.display import display
            display(model[1].drop('opt_res', axis=1).set_index([self.event_type_col, self.duration_col]))

    def plot_event_alpha(self, event: Union[str, int], ax: plt.Axes = None, scatter_kwargs: dict = {},
                         show=True, title=None, xlabel='t', ylabel=r'$\alpha_{jt}$', fontsize=18,
                         color: str = None, label: str = None) -> plt.Axes:
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
                              ylabel: str = r'$\alpha_{jt}$', fontsize: int = 18) -> plt.Axes:
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
        title = r'$\alpha_{jt}$' + f' for all events' if title is None else title
        for idx, (event, model) in enumerate(self.event_models.items()):
            label = f'{event}'
            color = colors[idx % len(colors)]
            self.plot_event_alpha(event=event, ax=ax, scatter_kwargs=scatter_kwargs, show=False, title=title,
                                  ylabel=ylabel, xlabel=xlabel, fontsize=fontsize, label=label, color=color)
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
        temp_df = df.copy()  # todo make sure .copy() is required
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

    def plot_all_events_beta(self, ax: plt.Axes = None, colors: list = COLORS, show: bool = True,
                             title: Union[str, None] = None, xlabel: str = 't',  ylabel: str = r'$\beta_{j}$',
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

        yt = list(se_df.index) * (len(se_df) // 2)
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
    # todo add assertions

    df_for_ploting = df.copy()  # todo make sure .copy() is required
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


def assert_fit(event_df, times):
    # todo: split to 2: one generic, one for new model
    if not event_df['success'].all():
        problematic_times = event_df.loc[~event_df['success'], "X"].tolist()
        event = event_df['J'].max()  # all the events in the dataframe are the same
        raise RuntimeError(f"In event J={event}, The method did not converged in D={problematic_times}."
                           f" Consider changing the problem definition."
                           f"\n See https://tomer1812.github.io/pydts/User%20Story/ for more details.")
    if event_df.shape[0] != len(times):
        event = event_df['J'].max()  # all the events in the dataframe are the same
        problematic_times = pd.Index(event_df['X']).symmetric_difference(times).tolist()
        raise RuntimeError(f"In event J={event}, The method didn't have events D={problematic_times}."
                           f" Consider changing the problem definition."
                           f"\n See https://tomer1812.github.io/pydts/User%20Story/ for more details.")


def repetitive_fitters(rep, n_patients, n_cov, d_times, j_events, pid_col, test_size,
                       drop_cols: Iterable = ("C", "T"),
                       model1=DataExpansionFitter,
                       model1_name="Lee",
                       model2=TwoStagesFitter,
                       model2_name: str = "Ours",
                       allow_fails: int = 20,
                       verbose: int = 2
                       ) -> Tuple[dict, dict]:
    # todo docstrings
    # todo assertions
    # todo move to utils?
    # todo try catch

    from pydts.examples_utils.plots import compare_beta_models_for_example
    rep_dict = {}
    times = {model1_name: [], model2_name: []}
    counts_df_list = []
    final = 0
    for samp in tqdm(range(rep+allow_fails)):
        try:
            patients_df = generate_quick_start_df(n_patients=n_patients, n_cov=n_cov, d_times=d_times, j_events=j_events,
                                                  pid_col=pid_col, seed=samp)
            train_df, test_df = train_test_split(patients_df, test_size=test_size)
            counts_df_list.append(train_df.groupby(['J', 'X']).size().to_frame(samp))
            drop_cols = pd.Index(drop_cols)
            start_1 = time()
            fitter = model1()
            fitter.fit(df=train_df.drop(drop_cols, axis=1))
            end_1 = time()
            start_2 = time()
            new_fitter = model2()
            if isinstance(new_fitter, TwoStagesFitter):
                new_fitter.fit(df=train_df.drop(drop_cols, axis=1), verbose=verbose)
            else:
                new_fitter.fit(df=train_df.drop(drop_cols, axis=1))
            end_2 = time()
            times[model1_name].append(end_1 - start_1)
            times[model2_name].append(end_2 - start_2)
            res_dict = compare_beta_models_for_example(fitter.event_models, new_fitter.event_models)
            rep_dict[samp] = res_dict
            final += 1
            if final == rep:
                break
        except:
            print(f'Failed to fit sample {samp}')
            continue
    print(f'final: {final}')
    ret_df = pd.concat(counts_df_list, axis=1).fillna(0).mean(axis=1).apply(np.ceil).to_frame()
    return rep_dict, times, ret_df


def get_real_hazard(df, real_coef_dict, times, events):
    """

    Args:
        df:
        real_coef_dict:
        times:
        events:

    Returns:

    """
    # todo docstrings
    # todo assertions
    # todo move to utils?

    a_t = {event: {t: real_coef_dict['alpha'][event](t) for t in times} for event in events}
    b = pd.concat([df.dot(real_coef_dict['beta'][j]) for j in events], axis=1, keys=events)

    for j in events:
        df[[f'hazard_j{j}_t{t}' for t in times]] = pd.concat([expit(a_t[j][t] + b[j]) for t in times],
                                                             axis=1).values
    return df


if __name__ == "__main__":
    from pydts.examples_utils.generate_simulations_data import generate_quick_start_df
    n_patients = 50000
    n_cov = 5
    patients_df = generate_quick_start_df(n_patients=n_patients, n_cov=n_cov, d_times=30, j_events=2,
                                          pid_col='pid', seed=0)
    train_df, test_df = train_test_split(patients_df, test_size=0.25)
    # m = DataExpansionFitter()
    # m.fit(df=df.drop(['C', 'T'], axis=1))
    # m.print_summary()

    m2 = TwoStagesFitter()
    m2.fit(train_df.drop(['C', 'T'], axis=1))
    #m2.plot_all_events_alpha()
    #pred_df = m2.predict_hazard_all(test_df)
    #pred_df = m2.predict_overall_survival(test_df, t=5)
    #pred_prob = m2.predict_prob_event_j_at_t(test_df, event=1, t=2)
    #m2.predict_event_cumulative_incident_function(test_df, event=1)
    tdf = test_df[[f'Z{i+1}' for i in range(n_cov)]]
    m2.predict_cumulative_incident_function(tdf)
    # m2.predict(test_df)
    # print(m2.get_beta_SE())
    # m2.plot_all_events_beta()
    print('x')

