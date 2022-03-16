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


# DEFAULT_MODELS_KWARGS = dict(family=sm.families.Binomial())
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
        self.formula = None
        self.expanded_df = None
        self.events = None
        self.covariates = None
        self.models_kwargs = dict(family=sm.families.Binomial())
        self.event_models = dict()

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
        self.events = [c for c in sorted(df[event_type_col].unique()) if c != 0]
        self.covariates = [col for col in df if col not in [event_type_col, duration_col, pid_col]]

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
        self.events = None
        self.covariates = None
        self.event_type_col = None
        self.duration_col = None
        self.pid_col = None
        self.times = None

    def _alpha_jt(self, x, df, y_t, beta_j, n_jt, t):
        partial_df = df[df[self.duration_col] >= t]
        expit_add = np.dot(partial_df[self.covariates], beta_j)
        return ((1 / y_t) * np.sum(expit(x + expit_add)) - (n_jt / y_t)) ** 2

    def _fit_event_beta(self, expanded_df, event, model=CoxPHFitter, model_kwargs={}, model_fit_kwargs={}):
        strata_df = expanded_df[self.covariates + [f'j_{event}', self.duration_col]]
        strata_df[f'{self.duration_col}_copy'] = expanded_df[self.duration_col]
        beta_j_model = model(**model_kwargs)
        beta_j_model.fit(df=strata_df[self.covariates + [f'{self.duration_col}', f'{self.duration_col}_copy', f'j_{event}']],
                         duration_col=self.duration_col, event_col=f'j_{event}', strata=f'{self.duration_col}_copy',
                         **model_fit_kwargs)
        return beta_j_model

    def _fit_beta(self, expanded_df, events, model=CoxPHFitter, model_kwargs={}, model_fit_kwargs={}):
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
            fit_beta_kwargs: dict = {}) -> dict:
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
        Returns:
            event_models (dict): Fitted models dictionary. Keys - event names, Values - fitted models for the event.
        """
        pandarallel.initialize()
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

        # y_t = len(df[duration_col]) - df[duration_col].value_counts().sort_index().cumsum()
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
            assert_fit(n_et)
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

    def predict_hazard_jt(self, df: pd.DataFrame, event: Union[str, int],  t: np.array) -> pd.DataFrame:
        '''
        This function calculates the hazard for the given event at the given time values if they were included in
        the training set of the event.

        Args:
            df (pd.DataFrame): samples to predict for
            event (Union[str, int]): event name
            t (np.array): times to calculate the hazard for

        Returns:
            df (pd.DataFrame): samples with the prediction columns
        '''

        if f'{self.duration_col}_copy' not in df.columns:
            drop = True
            df[f'{self.duration_col}_copy'] = df[self.duration_col]
        else:
            drop = False

        model = self.event_models[event]

        beta_j_x = (df[self.covariates] * model[0].params_).sum(axis=1)
        alpha_df = model[1].set_index(self.duration_col)
        _t = np.array([t_i for t_i in t
                       if ((t_i in alpha_df.index) and (f'hazard_j{event}_t{t_i}' not in df.columns))])
        if len(_t) == 0:
            if drop:
                df.drop(f'{self.duration_col}_copy', axis=1, inplace=True)
            return df

        if len(_t) > 1:
            beta_j_x = pd.concat([beta_j_x] * len(_t), ignore_index=True, axis=1)
        alpha_jt_t = pd.concat([alpha_df.loc[_t, 'alpha_jt'] * _t] * len(beta_j_x), axis=1).T
        alpha_jt_t.index = beta_j_x.index
        alpha_jt_t.columns = [f'hazard_j{event}_t{c}' for c in alpha_jt_t.columns]

        hazard_df = self.hazard_inverse_transformation(alpha_jt_t + beta_j_x.values)

        # todo validate this hazard imputation!

        for t_i in self.times[1:]:
            if f'hazard_j{event}_t{t_i}' not in hazard_df.columns:
                print(f'Imputing column hazard_j{event}_t{t_i}')
                hazard_df[f'hazard_j{event}_t{t_i}'] = hazard_df[f'hazard_j{event}_t{t_i-1}']
        hazard_df = hazard_df[[f'hazard_j{event}_t{t_i}' for t_i in self.times]]

        if drop:
            df.drop(f'{self.duration_col}_copy', axis=1, inplace=True)
        df = pd.concat([df, hazard_df], axis=1)
        return df

    def predict_hazard_t(self, df: pd.DataFrame, t: np.array) -> pd.DataFrame:
        """
        This function calculates the hazard for all the events at the requested time values if they were included in
        the training set of each event.

        Args:
            df (pd.DataFrame): samples to predict for
            t (np.array): times to calculate the hazard for

        Returns:
            df (pd.DataFrame): samples with the prediction columns
        """

        if f'{self.duration_col}_copy' not in df.columns:
            df[f'{self.duration_col}_copy'] = df[self.duration_col]

        for event, model in self.event_models.items():
            df = self.predict_hazard_jt(df=df, event=event, t=t)
        df.drop(f'{self.duration_col}_copy', axis=1, inplace=True)
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
        df = self.predict_hazard_t(df, t=self.times)
        return df

    def hazard_transformation(self, a: Union[int, np.array, pd.Series, pd.DataFrame]) -> \
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

    def hazard_inverse_transformation(self, a: Union[int, np.array, pd.Series, pd.DataFrame]) -> \
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

    def predict_overall_survival(self, df: pd.DataFrame, t: int = None, return_hazards: bool = False) -> pd.DataFrame:
        """
        This function adds columns of the overall survival until time t.
        Args:
            df (pandas.DataFrame): dataframe with covariates columns
            t (int): time
            return_hazards (bool): if to keep the hazard columns

        Returns:
            df (pandas.DataFrame): dataframe with the additional overall survival columns

        """
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
            overall = pd.concat([overall, all_hazards[[c for c in all_hazards.columns
                                                       if c[:7] == 'hazard_']]], axis=1)
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
        This function adds columns of a specific event occurance probabilities.

        Args:
            df (pandas.DataFrame): dataframe with covariates columns
            event (Union[str, int]): event name

        Returns:
            df (pandas.DataFrame): dataframe with probabilities columns

        """

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
        for event in self.events:
            if f'cif_j{event}_at_t{self.times[-1]}' not in df.columns:
                df = self.predict_event_cumulative_incident_function(df=df, event=event)
        return df

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
                             fontsize: int = 18) -> plt.Axes:
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

        Returns:
            ax (matplotlib.pyplot.Axes): output figure
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        title = r'$\beta_{j}$' + f' for all events' if title is None else title

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

    def predict_marginal_prob_event_j(self, df: pd.DataFrame, event: Union[str, int]) -> pd.DataFrame:
        """
        This function calculates the marginal probability of an event given the covariates.

        Args:
            df (pandas.DataFrame): dataframe with covariates columns included
            event (Union[str, int]): event name

        Returns:
            df (pandas.DataFrame): dataframe with additional prediction columns
        """
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
        for event in self.events:
            df = self.predict_marginal_prob_event_j(df=df, event=event)
        return df


def assert_fit(event_df):
    if not event_df['success'].all():
        problematic_times = event_df.loc[~event_df['success'], "X"].tolist()
        event = event_df['J'].max()  # all the events in the dataframe are the same
        print(f"In event J={event}, The method did not converged in D={problematic_times}."
              f" Consider changing the "
              f"problem definition. \n See TBD for more details.")
        # todo: add user example
        # raise RuntimeError("")



if __name__ == "__main__":
    from pydts.examples_utils.generate_simulations_data import generate_quick_start_df
    n_patients = 1000
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
    # m2.predict_cumulative_incident_function(test_df)
    # m2.predict(test_df)
    print(m2.get_beta_SE())
    m2.plot_all_events_beta()
    print('x')

