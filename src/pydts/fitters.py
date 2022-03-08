import statsmodels.api as sm
from pydts.examples_utils.generate_simulations_data import generate_quick_start_df
from pydts.base_fitters import ExpansionBasedFitter
from sklearn.model_selection import train_test_split
from scipy.special import expit
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from lifelines.fitters.coxph_fitter import CoxPHFitter
from pandarallel import pandarallel
from typing import Optional


DEFAULT_MODELS_KWARGS = dict(family=sm.families.Binomial())


class DataExpansionFitter(ExpansionBasedFitter):
    """
    This class implements the fitter as described in Lee et al. 2018
    """

    def _fit_event(self, df, formula, models_kwargs=DEFAULT_MODELS_KWARGS, model_fit_kwargs={}):
        model = sm.GLM.from_formula(formula=formula, data=df, **models_kwargs)
        return model.fit(**model_fit_kwargs)

    def fit(self,
            df: pd.DataFrame,
            event_type_col: str = 'J',
            duration_col: str = 'X',
            pid_col: str = 'pid',
            formula: Optional[str] = None,
            models_kwargs: Optional[dict] = DEFAULT_MODELS_KWARGS,
            model_fit_kwargs: Optional[dict] = {}):
        """
        This method fits a model to the discrete data.

        Args:
            df (pd.DataFrame): training data for fitting the model
            event_type_col (str): The event type column name (must be a column in df),
                                  Right censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).
            formula:
            models_kwargs:
            model_fit_kwargs:

        Returns:

        """

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
            self.event_models[event] = self._fit_event(df=self.expanded_df, formula=self.formula,
                    models_kwargs=models_kwargs, model_fit_kwargs=model_fit_kwargs)
        return self.event_models

    def print_summary(self, summary_func="summary", summary_kwargs={}):
        for event, model in self.event_models.items():
            _summary_func = getattr(model, summary_func, None)
            if _summary_func is not None:
                print(f'\n\nModel summary for event: {event}')
                print(_summary_func(**summary_kwargs))
            else:
                print(f'Not {summary_func} function in event {event} model')


class TwoStagesFitter(ExpansionBasedFitter):

    def _alpha_jt(self, x, df, y_t, beta_j, n_jt, t):
        partial_df = df[df[self.duration_col] >= t]
        expit_add = (partial_df[self.covariates] * beta_j).sum(axis=1)
        return ((1 / y_t) * np.sum(expit(x + expit_add)) - (n_jt / y_t)) ** 2

    def _fit_event_beta(self, expanded_df, event, model=CoxPHFitter,
                        model_kwargs={}, model_fit_kwargs={}):
        strata_df = expanded_df[self.covariates + [f'j_{event}', self.duration_col]]
        strata_df[f'{self.duration_col}_copy'] = expanded_df[self.duration_col]
        beta_j_model = model(**model_kwargs)
        beta_j_model.fit(df=strata_df[self.covariates + [f'{self.duration_col}', f'{self.duration_col}_copy', f'j_{event}']],
                         duration_col=self.duration_col, event_col=f'j_{event}', strata=f'{self.duration_col}_copy',
                         **model_fit_kwargs)
        return beta_j_model

    def _fit_beta(self, expanded_df, events, model=CoxPHFitter, model_kwargs={},
                  model_fit_kwargs={}):
        beta_models = {}
        for event in events:
            beta_models[event] = self._fit_event_beta(expanded_df=expanded_df, event=event,
                model=model, model_kwargs=model_kwargs, model_fit_kwargs=model_fit_kwargs)
        return beta_models

    def fit(self, df, covariates=None, event_type_col='J', duration_col='X', pid_col='pid', x0=0, fit_beta_kwargs={}):
        pandarallel.initialize()
        events = [c for c in sorted(df[event_type_col].unique()) if c != 0]
        if covariates is None:
            covariates = [col for col in df if col not in [event_type_col, duration_col, pid_col]]
        self.covariates = covariates
        self.event_type_col = event_type_col
        self.duration_col = duration_col
        self.pid_col = pid_col

        expanded_df = self._expand_data(df=df, event_type_col=event_type_col, duration_col=duration_col,
                                        pid_col=pid_col)

        beta_models = self._fit_beta(expanded_df, events, **fit_beta_kwargs)

        y_t = len(df[duration_col]) - df[duration_col].value_counts().sort_index().cumsum()
        n_jt = df.groupby([event_type_col, duration_col]).size().to_frame().reset_index()
        n_jt.columns = [event_type_col, duration_col, 'n_jt']

        alpha_df = pd.DataFrame()
        for event in events:
            n_et = n_jt[n_jt[event_type_col] == event]
            n_et['opt_res'] = n_et.parallel_apply(lambda row: minimize(self._alpha_jt, x0=x0,
                                    args=(df, y_t.loc[row[duration_col]], beta_models[event].params_, row['n_jt'],
                                    row[duration_col])), axis=1)
            n_et['success'] = n_et['opt_res'].parallel_apply(lambda val: val.success)
            n_et['alpha_jt'] = n_et['opt_res'].parallel_apply(lambda val: val.x[0])
            self.event_models[event] = [beta_models[event], n_et]
            alpha_df = pd.concat([alpha_df, n_et], ignore_index=True)

        return self.event_models


if __name__ == "__main__":
    n_patients = 2000
    n_cov = 5
    patients_df = generate_quick_start_df(n_patients=n_patients, n_cov=n_cov, d_times=30, j_events=2,
                                          pid_col='pid', seed=0)
    df, test_df = train_test_split(patients_df, test_size=0.25)
    m = DataExpansionFitter()
    m.fit(df=df.drop(['C', 'T'], axis=1))
    m.print_summary()

    m2 = TwoStagesFitter()
    m2.fit(df)
