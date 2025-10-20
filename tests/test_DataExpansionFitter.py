import unittest
from src.pydts.examples_utils.generate_simulations_data import generate_quick_start_df
from src.pydts.fitters import DataExpansionFitter
import numpy as np
import pandas as pd
from src.pydts.utils import get_real_hazard


class TestDataExpansionFitter(unittest.TestCase):
    def setUp(self):
        self.real_coef_dict = {
            "alpha": {
                1: lambda t: -1 - 0.3 * np.log(t),
                2: lambda t: -1.75 - 0.15 * np.log(t)
            },
            "beta": {
                1: -np.log([0.8, 3, 3, 2.5, 2]),
                2: -np.log([1, 3, 4, 3, 2])
            }
        }
        self.df = generate_quick_start_df(n_patients=1000, n_cov=5, d_times=10, j_events=2, pid_col='pid', seed=0,
                                          real_coef_dict=self.real_coef_dict, censoring_prob=0.8)
        self.m = DataExpansionFitter()
        self.fitted_model = DataExpansionFitter()
        self.fitted_model.fit(df=self.df.drop(['C', 'T'], axis=1))

    def test_fit_case_C_in_df(self):
        # 'C' named column cannot be passed in df to .fit()
        with self.assertRaises(ValueError):
            m = DataExpansionFitter()
            m.fit(df=self.df)

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

    def test_fit_case_correct_fit(self):
        # Fit should be successful and produce valid models
        m = DataExpansionFitter()
        m.fit(df=self.df.drop(['C', 'T'], axis=1))
        
        # Validate that fitting produced expected attributes
        self.assertEqual(m.events, [1, 2])
        self.assertEqual(m.times[:-1], list(range(1, 11, 1)))
        self.assertEqual(m.covariates, ['Z1', 'Z2', 'Z3', 'Z4', 'Z5'])
        self.assertIsNotNone(m.event_models)
        coef1 = m.get_beta_SE()[(1, '   coef   ')].astype(float)
        coef2 = m.get_beta_SE()[(2, '   coef   ')].astype(float)
        self.assertTrue((coef1.min() > -1.6))
        self.assertTrue((coef1.max() < 0.4))
        self.assertTrue((coef2.min() > -1.7))
        self.assertTrue((coef2.max() < 0.3))
        coef1 = m.get_alpha_df()[(1, '   coef   ')].astype(float)
        coef2 = m.get_alpha_df()[(2, '   coef   ')].astype(float)
        self.assertTrue((coef1.min() > -2.3))
        self.assertTrue((coef1.max() < 0))
        self.assertTrue((coef2.min() > -3.7))
        self.assertTrue((coef2.max() < -1.8))

    def test_fit_with_kwargs(self):
        import statsmodels.api as sm
        m = DataExpansionFitter()
        m.fit(df=self.df.drop(columns=['C', 'T']), models_kwargs=dict(family=sm.families.Binomial()))
        
        # Validate that fitting with kwargs produced expected attributes
        self.assertIsNotNone(m.events)
        self.assertIsNotNone(m.times)
        self.assertIsNotNone(m.covariates)
        self.assertEqual(len(m.events), 2)  # Should have 2 events
        self.assertEqual(len(m.times[:-1]), 10)  # Should have 10 event time points
        self.assertEqual(len(m.covariates), 5)  # Should have 5 covariates

    def test_print_summary(self):
        self.fitted_model.print_summary()

    def test_get_beta_SE(self):
        result = self.fitted_model.get_beta_SE()
        
        # Validate output structure and content
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    def test_get_alpha_SE(self):
        result = self.fitted_model.get_alpha_df()
        
        # Validate output structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    def test_predict_hazard_jt_case_covariate_not_in_df(self):
        # Covariates columns used in fit (here ['Z1','Z2','Z3','Z4','Z5']) must be passed in df to .predict()
        with self.assertRaises(AssertionError):
            self.fitted_model.predict_hazard_jt(
                df=self.df.drop(['C', 'T', 'Z1'], axis=1),
                event=self.fitted_model.events[0],
                t=self.fitted_model.times[0])

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
        result = self.fitted_model.predict_hazard_jt(
            df=self.df.drop(['C', 'T'], axis=1),
            event=self.fitted_model.events[0], t=self.fitted_model.times[0])
        
        # Validate output
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertEqual(len(result), len(self.df))  # Should have same number of rows as input

    def test_predict_hazard_t_case_successful_predict(self):
        result = self.fitted_model.predict_hazard_t(df=self.df.drop(['C', 'T'], axis=1), t=self.fitted_model.times[:3])
        
        # Validate output
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertEqual(len(result), len(self.df))  # Should have same number of rows as input

    def test_predict_hazard_all_case_successful_predict(self):
        result = self.fitted_model.predict_hazard_all(df=self.df.drop(['C', 'T'], axis=1))
        
        # Validate output structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.df))

    def test_predict_overall_survival_case_successful_predict(self):
        result = self.fitted_model.predict_overall_survival(
            df=self.df.drop(['C', 'T'], axis=1), t=self.fitted_model.times[5], return_hazards=True)
        
        # Validate output structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.df))

    def test_predict_prob_event_j_at_t_case_successful_predict(self):
        result = self.fitted_model.predict_prob_event_j_at_t(df=self.df.drop(['C', 'T'], axis=1),
                                                    event=self.fitted_model.events[0],
                                                    t=self.fitted_model.times[3])
        
        # Validate output structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.df))

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
        result = self.fitted_model.predict_full(df=self.df.drop(['C', 'T'], axis=1))
        for time in range(1, 11, 1):
            self.assertTrue(result[f"overall_survival_t{time}"].between(0, 1, inclusive='both').all())

        for event in [1, 2]:
            for time in range(1, 11, 1):
                self.assertTrue(result[f"prob_j{event}_at_t{time}"].between(0, 1, inclusive='both').all())
                self.assertTrue(result[f"cif_j{event}_at_t{time}"].between(0, 1, inclusive='both').all())
                self.assertTrue(result[f"hazard_j{event}_t{time}"].between(0, 1, inclusive='both').all())


    def test_predict_hazard_jt_case_hazard_already_on_df(self):
        df_temp = get_real_hazard(self.df.drop(['C', 'T', 'X', 'J'], axis=1).set_index('pid').copy(),
                                  real_coef_dict=self.real_coef_dict,
                                  times=self.fitted_model.times,
                                  events=self.fitted_model.events)
        assert (df_temp == self.fitted_model.predict_hazard_jt(df=df_temp,
                                                               event=self.fitted_model.events[0],
                                                               t=self.fitted_model.times
                                                               )
                ).all().all()