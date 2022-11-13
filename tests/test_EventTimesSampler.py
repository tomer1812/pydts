import unittest
from src.pydts.data_generation import EventTimesSampler
import numpy as np
import pandas as pd


class TestEventTimesSampler(unittest.TestCase):

    def test_sample_2_events_5_covariates(self):
        n_cov = 5
        n_patients = 1000
        seed = 0
        covariates = [f'Z{i + 1}' for i in range(n_cov)]
        patients_df = pd.DataFrame(data=np.random.uniform(low=0.0, high=1.0, size=[n_patients, n_cov]),
                                   columns=covariates)

        ets = EventTimesSampler(d_times=15, j_event_types=2)
        real_coef_dict = {
            "alpha": {
                1: lambda t: -1 - 0.3 * np.log(t),
                2: lambda t: -1.75 - 0.15 * np.log(t),
            },
            "beta": {
                1: -np.log([0.8, 3, 3, 2.5, 2]),
                2: -np.log([1, 3, 4, 3, 2]),
            }
        }
        ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)

    def test_sample_3_events_4_covariates(self):
        n_cov = 4
        n_patients = 1000
        seed = 0
        covariates = [f'Z{i + 1}' for i in range(n_cov)]
        patients_df = pd.DataFrame(data=np.random.uniform(low=0.0, high=1.0, size=[n_patients, n_cov]),
                                   columns=covariates)

        ets = EventTimesSampler(d_times=15, j_event_types=2)
        real_coef_dict = {
            "alpha": {
                1: lambda t: -1 - 0.3 * np.log(t),
                2: lambda t: -1.75 - 0.15 * np.log(t),
                3: lambda t: -1.5 - 0.2 * np.log(t),
            },
            "beta": {
                1: -np.log([0.8, 3, 3, 2.5]),
                2: -np.log([1, 3, 4, 3]),
                3: -np.log([0.5, 2, 3, 4]),
            }
        }
        ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)

    def test_sample_hazard_censoring(self):
        seed = 0
        n_cov = 5
        n_patients = 10000
        np.random.seed(seed)
        covariates = [f'Z{i + 1}' for i in range(n_cov)]
        patients_df = pd.DataFrame(data=np.random.uniform(low=0.0, high=1.0, size=[n_patients, n_cov]),
                                   columns=covariates)

        ets = EventTimesSampler(d_times=15, j_event_types=4)
        censoring_coef_dict = {
            "alpha": {
                1: lambda t: -1 - 0.3 * np.log(t),
            },
            "beta": {
                1: -np.log([0.8, 3, 3, 2.5, 2]),
            }
        }
        ets.sample_hazard_lof_censoring(patients_df, censoring_coef_dict, seed)

    def test_sample_independent_censoring(self):
        n_cov = 4
        n_patients = 1000
        covariates = [f'Z{i + 1}' for i in range(n_cov)]
        patients_df = pd.DataFrame(data=np.random.uniform(low=0.0, high=1.0, size=[n_patients, n_cov]),
                                   columns=covariates)

        ets = EventTimesSampler(d_times=15, j_event_types=2)
        ets.sample_independent_lof_censoring(patients_df, prob_los_at_t=0.03 * np.ones_like(ets.times))

    def test_update_event_or_lof(self):
        n_cov = 5
        n_patients = 1000
        seed = 0
        covariates = [f'Z{i + 1}' for i in range(n_cov)]
        patients_df = pd.DataFrame(data=np.random.uniform(low=0.0, high=1.0, size=[n_patients, n_cov]),
                                   columns=covariates)

        ets = EventTimesSampler(d_times=15, j_event_types=3)
        real_coef_dict = {
            "alpha": {
                1: lambda t: -1 - 0.3 * np.log(t),
                2: lambda t: -1.75 - 0.15 * np.log(t),
                3: lambda t: -1.75 - 0.15 * np.log(t),
            },
            "beta": {
                1: -np.log([0.8, 3, 3, 2.5, 2]),
                2: -np.log([1, 3, 4, 3, 2]),
                3: -np.log([1, 3, 4, 3, 2]),
            }
        }
        patients_df = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
        censoring_coef_dict = {
            "alpha": {
                1: lambda t: -1 - 0.3 * np.log(t),
            },
            "beta": {
                1: -np.log([0.8, 3, 3, 2.5, 2]),
            }
        }

        patients_df = ets.sample_hazard_lof_censoring(patients_df, censoring_coef_dict, seed=seed)
        patients_df = ets.update_event_or_lof(patients_df)

    def test_update_event_or_lof_T_assertion(self):
        with self.assertRaises(AssertionError):
            seed = 0
            n_cov = 5
            n_patients = 10000
            np.random.seed(seed)
            covariates = [f'Z{i + 1}' for i in range(n_cov)]
            patients_df = pd.DataFrame(data=np.random.uniform(low=0.0, high=1.0, size=[n_patients, n_cov]),
                                       columns=covariates)

            ets = EventTimesSampler(d_times=15, j_event_types=3)
            censoring_coef_dict = {
                "alpha": {
                    1: lambda t: -1 - 0.3 * np.log(t),
                },
                "beta": {
                    1: -np.log([0.8, 3, 3, 2.5, 2]),
                }
            }

            patients_df = ets.sample_hazard_lof_censoring(patients_df, censoring_coef_dict, seed=seed)
            patients_df = ets.update_event_or_lof(patients_df)

    def test_update_event_or_lof_C_assertion(self):
        with self.assertRaises(AssertionError):
            n_cov = 5
            n_patients = 1000
            seed = 0
            covariates = [f'Z{i + 1}' for i in range(n_cov)]
            patients_df = pd.DataFrame(data=np.random.uniform(low=0.0, high=1.0, size=[n_patients, n_cov]),
                                       columns=covariates)

            ets = EventTimesSampler(d_times=15, j_event_types=3)
            real_coef_dict = {
                "alpha": {
                    1: lambda t: -1 - 0.3 * np.log(t),
                    2: lambda t: -1.75 - 0.15 * np.log(t),
                    3: lambda t: -1.75 - 0.15 * np.log(t),
                },
                "beta": {
                    1: -np.log([0.8, 3, 3, 2.5, 2]),
                    2: -np.log([1, 3, 4, 3, 2]),
                    3: -np.log([1, 3, 4, 3, 2]),
                }
            }
            patients_df = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
            patients_df = ets.update_event_or_lof(patients_df)