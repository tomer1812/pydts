import unittest

import matplotlib.pyplot as plt
from unittest.mock import patch
from src.pydts.examples_utils.generate_simulations_data import generate_quick_start_df
from src.pydts.fitters import TwoStagesFitterExact
from src.pydts.data_generation import EventTimesSampler
from src.pydts.utils import get_real_hazard
import numpy as np
import pandas as pd


class TestTwoStagesFitterExact(unittest.TestCase):

    def setUp(self):
        self.real_coef_dict = {
            "alpha": {
                1: lambda t: -1. + 0.4 * np.log(t),
                2: lambda t: -1. + 0.4 * np.log(t),
            },
            "beta": {
                1: -0.4*np.log([0.8, 3, 3, 2.5, 2]),
                2: -0.3*np.log([1, 3, 4, 3, 2]),
            }
        }
        self.df = generate_quick_start_df(n_patients=300, n_cov=5, d_times=4, j_events=2, pid_col='pid', seed=0,
                                          real_coef_dict=self.real_coef_dict, censoring_prob=0.1)
        self.df = self.df.apply(pd.to_numeric, errors="raise")

        self.m = TwoStagesFitterExact()
        self.fitted_model = TwoStagesFitterExact()
        self.fitted_model.fit(df=self.df.drop(['C', 'T'], axis=1))

    def test_fit_case_event_col_not_in_df(self):
        # Event column (here 'J') must be passed in df to .fit()
        print(self.fitted_model.get_beta_SE())


    def test_fit_case_event_col_not_in_df(self):
        # Event column (here 'J') must be passed in df to .fit()
        with self.assertRaises(AssertionError):
            self.m.fit(df=self.df.drop(['C', 'J', 'T'], axis=1))

    def test_fit_case_duration_col_not_in_df(self):
        # Duration column (here 'X') must be passed in df to .fit()
        with self.assertRaises(AssertionError):
            self.m.fit(df=self.df.drop(['C', 'X', 'T'], axis=1))

    def test_fit_case_pid_col_not_in_df(self):
        # Duration column (here 'pid') must be passed in df to .fit()
        with self.assertRaises(AssertionError):
            self.m.fit(df=self.df.drop(['C', 'pid', 'T'], axis=1))

    def test_fit_case_cov_col_not_in_df(self):
        # Covariates columns (here ['Z1','Z2','Z3','Z4','Z5']) must be passed in df to .fit()
        with self.assertRaises(ValueError):
            self.m.fit(df=self.df.drop(['C', 'T'], axis=1), covariates=['Z6'])

    def test_fit_case_missing_jt(self):
        # df to .fit() should contain observations to all (event, times)

        # drop (events[1], times[2])
        tmp_df = self.df[
            ~((self.df[self.fitted_model.duration_col] == self.fitted_model.times[2]) &
              (self.df[self.fitted_model.event_type_col] == self.fitted_model.events[1]))
        ]

        with self.assertRaises(RuntimeError):
            self.m.fit(df=tmp_df.drop(['C', 'T'], axis=1))

    def test_fit_case_correct_fit(self):
        # Fit should be successful
        m = TwoStagesFitterExact()
        m.fit(df=self.df.drop(['C', 'T'], axis=1))

    def test_print_summary(self):
        self.fitted_model.print_summary()

    def test_plot_event_alpha_case_correct_event(self):
        self.fitted_model.plot_event_alpha(event=self.fitted_model.events[0], show=False)

    def test_plot_event_alpha_case_correct_event_and_show(self):
        with patch('matplotlib.pyplot.show') as p:
            self.fitted_model.plot_event_alpha(event=self.fitted_model.events[0], show=True)

    def test_plot_event_alpha_case_incorrect_event(self):
        with self.assertRaises(AssertionError):
            self.fitted_model.plot_event_alpha(event=100, show=False)

    def test_plot_all_events_alpha(self):
        self.fitted_model.plot_all_events_alpha(show=False)

    def test_plot_all_events_alpha_case_show(self):
        with patch('matplotlib.pyplot.show') as p:
            self.fitted_model.plot_all_events_alpha(show=True)

    def test_plot_all_events_alpha_case_axis_exits(self):
        fig, ax = plt.subplots()
        self.fitted_model.plot_all_events_alpha(show=False, ax=ax)

    def test_get_beta_SE(self):
        self.fitted_model.get_beta_SE()

    def test_get_alpha_df(self):
        self.fitted_model.get_alpha_df()

    # def test_plot_all_events_beta(self):
    #     self.fitted_model.plot_all_events_beta(show=False)
    #
    # def test_plot_all_events_beta_case_show(self):
    #     with patch('matplotlib.pyplot.show') as p:
    #         self.fitted_model.plot_all_events_beta(show=True)
    #
    # def test_plot_all_events_beta_case_ax_exists(self):
    #     fig, ax = plt.subplots()
    #     self.fitted_model.plot_all_events_beta(show=False, ax=ax)

    def test_predict_hazard_jt_case_covariate_not_in_df(self):
        # Covariates columns used in fit (here ['Z1','Z2','Z3','Z4','Z5']) must be passed in df to .predict()
        with self.assertRaises(AssertionError):
            self.fitted_model.predict_hazard_jt(
                df=self.df.drop(['C', 'T', 'Z1'], axis=1),
                event=self.fitted_model.events[0],
                t=self.fitted_model.times[0])

    def test_predict_hazard_jt_case_hazard_already_on_df(self):
        # print(self.df.drop(['C', 'T', 'X', 'J'], axis=1).set_index('pid').copy())
        df_temp = get_real_hazard(self.df.drop(['C', 'T', 'X', 'J'], axis=1).set_index('pid').copy(),
                                  real_coef_dict=self.real_coef_dict,
                                  times=self.fitted_model.times,
                                  events=self.fitted_model.events)
        assert (df_temp == self.fitted_model.predict_hazard_jt(df=df_temp,
                                                               event=self.fitted_model.events[0],
                                                               t=self.fitted_model.times
                                                               )
                ).all().all()

    def test_hazard_transformation_result(self):
        from scipy.special import logit
        num = np.array([0.5])
        a = logit(num)
        print(a)
        assert (a == self.fitted_model._hazard_transformation(num)).all()

    def test_predict_hazard_jt_case_event_not_in_events(self):
        # Event passed to .predict() must be in fitted events
        with self.assertRaises(AssertionError):
            self.fitted_model.predict_hazard_jt(
                df=self.df.drop(['C', 'T'], axis=1), event=100, t=self.fitted_model.times[0])

    def test_predict_hazard_jt_case_time_not_in_times(self):
        # Event passed to .predict() must be in fitted events
        with self.assertRaises(AssertionError):
            self.fitted_model.predict_hazard_jt(
                df=self.df.drop(['C', 'T'], axis=1), event=self.fitted_model.events[0], t=1000)

    def test_predict_hazard_jt_case_successful_predict(self):
        self.fitted_model.predict_hazard_jt(
            df=self.df.drop(['C', 'T'], axis=1),
            event=self.fitted_model.events[0], t=self.fitted_model.times[0])

    def test_predict_hazard_t_case_successful_predict(self):
        self.fitted_model.predict_hazard_t(df=self.df.drop(['C', 'T'], axis=1), t=self.fitted_model.times[:3])

    def test_predict_hazard_all_case_successful_predict(self):
        self.fitted_model.predict_hazard_all(df=self.df.drop(['C', 'T'], axis=1))

    def test_predict_overall_survival_case_successful_predict(self):
        self.fitted_model.predict_overall_survival(
            df=self.df.drop(['C', 'T'], axis=1), t=self.fitted_model.times[3], return_hazards=True)

    def test_predict_prob_event_j_at_t_case_successful_predict(self):
        self.fitted_model.predict_prob_event_j_at_t(df=self.df.drop(['C', 'T'], axis=1),
                                                    event=self.fitted_model.events[0],
                                                    t=self.fitted_model.times[3])

    def test_predict_prob_event_j_all_case_successful_predict(self):
        self.fitted_model.predict_prob_event_j_all(df=self.df.drop(['C', 'T'], axis=1),
                                                   event=self.fitted_model.events[0])

    def test_predict_prob_events_case_successful_predict(self):
        self.fitted_model.predict_prob_events(df=self.df.drop(['C', 'T'], axis=1))

    def test_predict_event_cumulative_incident_function_case_successful_predict(self):
        self.fitted_model.predict_event_cumulative_incident_function(df=self.df.drop(['C', 'T'], axis=1),
                                                                     event=self.fitted_model.events[0])

    def test_predict_cumulative_incident_function_case_successful_predict(self):
        self.fitted_model.predict_cumulative_incident_function(df=self.df.drop(['C', 'T'], axis=1))

    def test_predict_full_case_successful_predict(self):
        self.fitted_model.predict_full(df=self.df.drop(['C', 'T'], axis=1))

    def test_predict_marginal_prob_function_case_successful(self):
        self.fitted_model.predict_marginal_prob_event_j(df=self.df.drop(columns=['C', 'T']),
                                                        event=1)

    def test_predict_marginal_prob_function_case_event_not_exists(self):
        # Event passed to .predict_margnial() must be in fitted events
        with self.assertRaises(AssertionError):
            self.fitted_model.predict_marginal_prob_event_j(
                df=self.df.drop(['C', 'T'], axis=1), event=100)

    def test_predict_marginal_prob_function_cov_not_exists(self):
        # Covariates columns used in fit (here ['Z1','Z2','Z3','Z4','Z5']) must be passed in df to .predict()
        with self.assertRaises(AssertionError):
            self.fitted_model.predict_marginal_prob_event_j(
                df=self.df.drop(columns=['C', 'T', 'Z1']),
                event=self.fitted_model.events[0])

    def test_predict_marginal_prob_all_events_function_successful(self):
        self.fitted_model.predict_marginal_prob_all_events(df=self.df.drop(columns=['C', 'T']))

    def test_predict_marginal_prob_all_events_cov_not_exits(self):
        # Covariates columns used in fit (here ['Z1','Z2','Z3','Z4','Z5']) must be passed in df to .predict()
        with self.assertRaises(AssertionError):
            self.fitted_model.predict_marginal_prob_all_events(
                df=self.df.drop(columns=['C', 'T', 'Z1']))

    def test_alpha_jt_function_value(self):
        t = 1
        j = 1
        row = self.fitted_model.alpha_df.query("X == @t and J == @j")
        x = row['alpha_jt'].item()
        y_t = (self.df["X"]
               .value_counts()
               .sort_index(ascending=False)  # each event count for its occurring time and the times before
               .cumsum()
               .sort_index()
               )
        rel_y_t = y_t.loc[t]
        #rel_beta = self.fitted_model.beta_models[j].params_
        rel_beta = getattr(self.fitted_model.beta_models[j], self.fitted_model.beta_models_params_attr)
        n_jt = row['n_jt']
        df = self.df.drop(columns=['C', 'T'])
        partial_df = df[df["X"] >= t]
        expit_add = np.dot(partial_df[self.fitted_model.covariates], rel_beta)
        from scipy.special import expit
        a_jt = ((1 / rel_y_t) * np.sum(expit(x + expit_add)) - (n_jt / rel_y_t)) ** 2
        a_jt_from_func = self.fitted_model._alpha_jt(x=x, df=df,
                                                     y_t=rel_y_t, beta_j=rel_beta,
                                                     n_jt=n_jt, t=t, event=j)
        self.assertEqual(a_jt.item(), a_jt_from_func.item())

    def test_predict_event_jt_case_t1_not_hazard(self):
        self.fitted_model.predict_prob_event_j_at_t(df=self.df.drop(['C', 'T'], axis=1),
                                                    event=self.fitted_model.events[0],
                                                    t=self.fitted_model.times[0])

    def test_predict_event_jt_case_t3_not_hazard(self):
        temp_df = self.df.drop(['C', 'T'], axis=1)
        temp_df = self.fitted_model.predict_overall_survival(df=temp_df,
                                                             t=self.fitted_model.times[3],
                                                             return_hazards=False)
        self.fitted_model.predict_prob_event_j_at_t(df=temp_df,
                                                    event=self.fitted_model.events[0],
                                                    t=self.fitted_model.times[3])

    def test_regularization_same_for_all_beta_models(self):
        L1_regularized_fitter = TwoStagesFitterExact()

        fit_beta_kwargs = {
            'model_kwargs': {
                'alpha': 0.03,
                'L1_wt': 1
            }
        }

        L1_regularized_fitter.fit(df=self.df.drop(['C', 'T'], axis=1), fit_beta_kwargs=fit_beta_kwargs)
        print(L1_regularized_fitter.get_beta_SE())

    def test_regularization_different_to_each_beta_model(self):
        L1_regularized_fitter = TwoStagesFitterExact()

        fit_beta_kwargs = {
            'model_fit_kwargs': {
                1: {
                        'alpha': 0.003,
                        'L1_wt': 1
                },
                2: {
                        'alpha': 0.005,
                        'L1_wt': 1
                }
            }
        }

        L1_regularized_fitter.fit(df=self.df.drop(['C', 'T'], axis=1), fit_beta_kwargs=fit_beta_kwargs)
        # print(L1_regularized_fitter.get_beta_SE())

        L2_regularized_fitter = TwoStagesFitterExact()

        fit_beta_kwargs = {
            'model_fit_kwargs': {
                1: {
                        'alpha': 0.003,
                        'L1_wt': 0
                },
                2: {
                        'alpha': 0.005,
                        'L1_wt': 0
                }
            }
        }

        L2_regularized_fitter.fit(df=self.df.drop(['C', 'T'], axis=1), fit_beta_kwargs=fit_beta_kwargs)
        # print(L2_regularized_fitter.get_beta_SE())

    def test_different_covariates_to_each_beta_model(self):
        twostages_fitter = TwoStagesFitterExact()
        covariates = {
            1: ['Z1', 'Z2', 'Z3'],
            2: ['Z2', 'Z3', 'Z4', 'Z5']
        }
        twostages_fitter.fit(df=self.df.drop(['C', 'T'], axis=1), covariates=covariates)

    def test_different_covariates_to_each_beta_model_prediction(self):
        twostages_fitter = TwoStagesFitterExact()
        covariates = {
            1: ['Z1', 'Z2', 'Z3'],
            2: ['Z2', 'Z3', 'Z4', 'Z5']
        }
        twostages_fitter.fit(df=self.df.drop(['C', 'T'], axis=1), covariates=covariates)
        twostages_fitter.predict_cumulative_incident_function(df=self.df.drop(['C', 'T'], axis=1))
