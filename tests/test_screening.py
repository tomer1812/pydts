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
        self.patients_df = pd.to_numeric(self.patients_df)
        self.covariates = covariates
        self.fitter = SISTwoStagesFitter()

    def test_psis_permute_df(self):
        self.fitter.permute_df(df=self.patients_df)

    def test_psis_fit_marginal_model(self):
        expanded_df = get_expanded_df(self.patients_df.drop(['C', 'T'], axis=1))
        self.fitter.fit_marginal_model(expanded_df, covariate='Z1')

    def test_psis_get_marginal_estimates(self):
        expanded_df = get_expanded_df(self.patients_df.drop(['C', 'T'], axis=1))
        self.fitter.get_marginal_estimates(expanded_df)

    def test_psis_get_data_driven_treshold(self):
        self.fitter.get_data_driven_threshold(df=self.patients_df.drop(['C', 'T'], axis=1))

    def test_psis_fit_data_driven_threshold(self):
        self.fitter.fit(df=self.patients_df.drop(['C', 'T'], axis=1), quantile=0.95)

    def test_psis_fit_user_defined_threshold(self):
        self.fitter.fit(df=self.patients_df.drop(['C', 'T'], axis=1), threshold=0.15)

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
