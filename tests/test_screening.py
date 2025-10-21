import unittest
from src.pydts.data_generation import EventTimesSampler
from src.pydts.screening import SISTwoStagesFitterExact, SISTwoStagesFitter, get_expanded_df
import numpy as np
import pandas as pd


class TestScreening(unittest.TestCase):

    def setUp(self):
        n_cov = 50
        beta1 = np.zeros(n_cov)
        beta1[:5] = np.array([-0.6, 0.5, -0.5, 0.6, -0.6])
        beta2 = np.zeros(n_cov)
        beta2[:5] = np.array([0.5, -0.7, 0.7, -0.5, -0.7])

        real_coef_dict = {
            "alpha": {
                1: lambda t: -2.6 + 0.1 * np.log(t),
                2: lambda t: -2.7 + 0.2 * np.log(t)
            },
            "beta": {
                1: beta1,
                2: beta2
            }
        }

        n_patients = 400
        d_times = 7
        j_events = 2

        ets = EventTimesSampler(d_times=d_times, j_event_types=j_events)

        seed = 2
        means_vector = np.zeros(n_cov)
        covariance_matrix = np.identity(n_cov)

        clip_value = 3

        covariates = [f'Z{i + 1}' for i in range(n_cov)]

        patients_df = pd.DataFrame(data=pd.DataFrame(data=np.random.multivariate_normal(means_vector, covariance_matrix,
                                                                                        size=n_patients),
                                                     columns=covariates))
        patients_df.clip(lower=-1 * clip_value, upper=clip_value, inplace=True)
        patients_df = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
        patients_df = ets.sample_independent_lof_censoring(patients_df, prob_lof_at_t=0.01 * np.ones(d_times),
                                                           seed=seed + 1)
        patients_df = ets.update_event_or_lof(patients_df)
        patients_df.index.name = 'pid'
        self.patients_df = patients_df.reset_index()
        self.patients_df = self.patients_df.apply(pd.to_numeric, errors="raise")
        self.covariates = covariates
        self.fitter = SISTwoStagesFitter()

    def test_psis_permute_df(self):
        result = self.fitter.permute_df(df=self.patients_df)
        
        # Validate output structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), len(self.patients_df))
        self.assertEqual(set(result.columns), set(self.patients_df.columns))
        
        # Check that permutation actually changed the data (with high probability)
        # At least some rows should be different after permutation
        different_rows = (result != self.patients_df).any(axis=1).sum()
        self.assertGreater(different_rows, len(self.patients_df) * 0.5)  # At least 50% should be different

    def test_psis_fit_marginal_model(self):
        expanded_df = get_expanded_df(self.patients_df.drop(['C', 'T'], axis=1))
        result = self.fitter.fit_marginal_model(expanded_df, covariate='Z1')
        
        # Validate that marginal model returns beta estimates DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        
        # Should have one row for the single covariate 'Z1'
        self.assertEqual(len(result), 1)
        self.assertIn('Z1', result.index)
        
        # Check that we have columns for both events (structure may vary)
        if hasattr(result.columns, 'levels'):
            # MultiIndex columns case
            self.assertEqual(len(result.columns.levels[0]), 2)  # Should have 2 events
        else:
            # Regular columns case - should have at least some columns
            self.assertGreater(len(result.columns), 0)
        
        # Check that all values are numeric and finite
        numeric_data = result.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            self.assertTrue(numeric_data.notna().all().all())
            self.assertTrue(np.isfinite(numeric_data).all().all())

    def test_psis_get_marginal_estimates(self):
        expanded_df = get_expanded_df(self.patients_df.drop(['C', 'T'], axis=1))
        result = self.fitter.get_marginal_estimates(expanded_df)
        
        # Validate output structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        
        # Should have estimates for all covariates
        self.assertEqual(len(result), len(self.covariates))
        
        # Check that estimates are numeric
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        self.assertGreater(len(numeric_cols), 0)  # Should have at least some numeric columns
        
        for col in numeric_cols:
            self.assertTrue(result[col].notna().all())
            self.assertTrue(np.isfinite(result[col]).all())

    def test_psis_get_data_driven_treshold(self):
        result = self.fitter.get_data_driven_threshold(df=self.patients_df.drop(['C', 'T'], axis=1))
        
        # Validate threshold properties
        self.assertIsInstance(result, (float, np.float64))
        self.assertGreater(result, 0)  # Threshold should be positive
        self.assertTrue(np.isfinite(result))
        
        # Check that the fitter now has null model data
        self.assertIsInstance(self.fitter.null_model_df, pd.DataFrame)
        self.assertFalse(self.fitter.null_model_df.empty)
        
        # Check that permuted data was created
        self.assertIsInstance(self.fitter.permuted_df, pd.DataFrame)
        self.assertEqual(len(self.fitter.permuted_df), len(self.patients_df))

    def test_psis_fit_data_driven_threshold(self):
        result = self.fitter.fit(df=self.patients_df.drop(['C', 'T'], axis=1), quantile=0.95)
        
        # Validate that fitting was successful and returns a fitted model
        self.assertIsNotNone(result)
        # The result should be a TwoStagesFitter instance
        from src.pydts.fitters import TwoStagesFitter
        self.assertIsInstance(result, TwoStagesFitter)
        
        # Check that the model has been fitted with expected attributes
        self.assertEqual(result.events, [1, 2])  # Should have 2 events
        self.assertEqual(result.times[:-1], list(range(1, 8, 1)))
        
        # Check that screening was performed
        self.assertIsInstance(self.fitter.chosen_covariates, (list, np.ndarray))
        
        # Check that threshold was set
        self.assertIsInstance(self.fitter.threshold, (float, np.float64))
        self.assertGreater(self.fitter.threshold, 0)
        
        # Check that some covariates were chosen
        self.assertGreater(len(self.fitter.chosen_covariates), 0)  # Should select at least some

    def test_psis_fit_user_defined_threshold(self):
        result = self.fitter.fit(df=self.patients_df.drop(['C', 'T'], axis=1), threshold=0.15)
        
        # Validate that fitting with user-defined threshold was successful
        self.assertIsNotNone(result)
        from src.pydts.fitters import TwoStagesFitter
        self.assertIsInstance(result, TwoStagesFitter)
        
        # Check that the user-defined threshold was used
        self.assertEqual(self.fitter.threshold, 0.15)
        
        # Check that screening was performed with the threshold
        self.assertIsInstance(self.fitter.chosen_covariates, (list, np.ndarray))
        
        # Check that the final model was fitted
        self.assertIsNotNone(result.covariates)
        self.assertEqual(result.events, [1, 2])  # Should have 2 events
        self.assertEqual(result.times[:-1], list(range(1, 8, 1)))

    def test_psis_covs_dict(self):
        with self.assertRaises(ValueError):
            self.fitter.fit(df=self.patients_df.drop(['C', 'T'], axis=1),
                            covariates={1: self.covariates[:-3], 2: self.covariates[:-8]})


class TestScreeningExact(TestScreening):

    def setUp(self):
        n_cov = 30
        beta1 = np.zeros(n_cov)
        beta1[:5] = np.array([-0.6, 0.5, -0.5, 0.6, -0.6])
        beta2 = np.zeros(n_cov)
        beta2[:5] = np.array([0.5, -0.7, 0.7, -0.5, -0.7])

        real_coef_dict = {
            "alpha": {
                1: lambda t: -3.1 + 0.1 * np.log(t),
                2: lambda t: -3.2 + 0.2 * np.log(t)
            },
            "beta": {
                1: beta1,
                2: beta2
            }
        }

        n_patients = 400
        d_times = 7
        j_events = 2

        ets = EventTimesSampler(d_times=d_times, j_event_types=j_events)

        seed = 2
        means_vector = np.zeros(n_cov)
        covariance_matrix = np.identity(n_cov)

        clip_value = 3

        covariates = [f'Z{i + 1}' for i in range(n_cov)]

        patients_df = pd.DataFrame(data=pd.DataFrame(data=np.random.multivariate_normal(means_vector, covariance_matrix,
                                                                                        size=n_patients),
                                                     columns=covariates))
        patients_df.clip(lower=-1 * clip_value, upper=clip_value, inplace=True)
        patients_df = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
        patients_df = ets.sample_independent_lof_censoring(patients_df, prob_lof_at_t=0.01 * np.ones(d_times),
                                                           seed=seed + 1)
        patients_df = ets.update_event_or_lof(patients_df)
        patients_df.index.name = 'pid'
        self.patients_df = patients_df.reset_index()
        self.covariates = covariates
        self.fitter = SISTwoStagesFitterExact()
