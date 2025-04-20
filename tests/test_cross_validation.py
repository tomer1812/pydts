import unittest
import numpy as np
import pandas as pd
from src.pydts.data_generation import EventTimesSampler
from src.pydts.cross_validation import TwoStagesCV, PenaltyGridSearchCV, TwoStagesCVExact, PenaltyGridSearchCVExact


class TestCrossValidation(unittest.TestCase):

    def setUp(self):
        n_cov = 6
        beta1 = np.zeros(n_cov)
        beta1[:5] = (-0.5 * np.log([0.8, 3, 3, 2.5, 2]))
        beta2 = np.zeros(n_cov)
        beta2[:5] = (-0.5 * np.log([1, 3, 4, 3, 2]))

        real_coef_dict = {
            "alpha": {
                1: lambda t: -2.0 - 0.2 * np.log(t),
                2: lambda t: -2.2 - 0.2 * np.log(t)
            },
            "beta": {
                1: beta1,
                2: beta2
            }
        }
        n_patients = 1000
        d_times = 5
        j_events = 2

        ets = EventTimesSampler(d_times=d_times, j_event_types=j_events)

        seed = 0
        means_vector = np.zeros(n_cov)
        covariance_matrix = 0.5 * np.identity(n_cov)
        clip_value = 1

        covariates = [f'Z{i + 1}' for i in range(n_cov)]

        patients_df = pd.DataFrame(data=pd.DataFrame(data=np.random.multivariate_normal(means_vector, covariance_matrix,
                                                                                        size=n_patients),
                                                     columns=covariates))
        patients_df.clip(lower=-1 * clip_value, upper=clip_value, inplace=True)
        patients_df = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
        patients_df.index.name = 'pid'
        patients_df = patients_df.reset_index()
        patients_df = ets.sample_independent_lof_censoring(patients_df, prob_lof_at_t=0.01 * np.ones(d_times))
        self.patients_df = ets.update_event_or_lof(patients_df)
        self.tscv = TwoStagesCV()

    def test_cross_validation_bs(self):
        self.tscv.cross_validate(self.patients_df, metrics='BS', n_splits=2)

    def test_cross_validation_auc(self):
        self.tscv.cross_validate(self.patients_df, metrics='AUC', n_splits=2)

    def test_cross_validation_iauc(self):
        self.tscv.cross_validate(self.patients_df, metrics='IAUC', n_splits=2)

    def test_cross_validation_gauc(self):
        self.tscv.cross_validate(self.patients_df, metrics='GAUC', n_splits=2)

    def test_cross_validation_ibs(self):
        self.tscv.cross_validate(self.patients_df, metrics='IBS', n_splits=2)

    def test_cross_validation_gbs(self):
        self.tscv.cross_validate(self.patients_df, metrics='GBS', n_splits=3)


class TestCrossValidationExact(TestCrossValidation):

    def setUp(self):
        n_cov = 6
        beta1 = np.zeros(n_cov)
        beta1[:5] = (-0.25 * np.log([0.8, 3, 3, 2.5, 2]))
        beta2 = np.zeros(n_cov)
        beta2[:5] = (-0.25 * np.log([1, 3, 4, 3, 2]))

        real_coef_dict = {
            "alpha": {
                1: lambda t: -1.7 - 0.2 * np.log(t),
                2: lambda t: -1.8 - 0.2 * np.log(t)
            },
            "beta": {
                1: beta1,
                2: beta2
            }
        }
        n_patients = 500
        d_times = 4
        j_events = 2

        ets = EventTimesSampler(d_times=d_times, j_event_types=j_events)

        seed = 0
        means_vector = np.zeros(n_cov)
        covariance_matrix = 0.5 * np.identity(n_cov)
        clip_value = 1

        covariates = [f'Z{i + 1}' for i in range(n_cov)]

        patients_df = pd.DataFrame(data=pd.DataFrame(data=np.random.multivariate_normal(means_vector, covariance_matrix,
                                                                                        size=n_patients),
                                                     columns=covariates))
        patients_df.clip(lower=-1 * clip_value, upper=clip_value, inplace=True)
        patients_df = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
        patients_df.index.name = 'pid'
        patients_df = patients_df.reset_index()
        patients_df = ets.sample_independent_lof_censoring(patients_df, prob_lof_at_t=0.01 * np.ones(d_times))
        self.patients_df = ets.update_event_or_lof(patients_df)
        self.tscv = TwoStagesCVExact()


class TestPenaltyGridSearchCV(unittest.TestCase):

    def setUp(self):
        n_cov = 6
        beta1 = np.zeros(n_cov)
        beta1[:5] = (-0.5 * np.log([0.8, 3, 3, 2.5, 2]))
        beta2 = np.zeros(n_cov)
        beta2[:5] = (-0.5 * np.log([1, 3, 4, 3, 2]))

        real_coef_dict = {
            "alpha": {
                1: lambda t: -2.0 - 0.2 * np.log(t),
                2: lambda t: -2.2 - 0.2 * np.log(t)
            },
            "beta": {
                1: beta1,
                2: beta2
            }
        }
        n_patients = 2000
        d_times = 5
        j_events = 2

        ets = EventTimesSampler(d_times=d_times, j_event_types=j_events)

        seed = 0
        means_vector = np.zeros(n_cov)
        covariance_matrix = 0.5 * np.identity(n_cov)
        clip_value = 1

        covariates = [f'Z{i + 1}' for i in range(n_cov)]

        patients_df = pd.DataFrame(data=pd.DataFrame(data=np.random.multivariate_normal(means_vector, covariance_matrix,
                                                                                        size=n_patients),
                                                     columns=covariates))
        patients_df.clip(lower=-1 * clip_value, upper=clip_value, inplace=True)
        patients_df = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
        patients_df.index.name = 'pid'
        patients_df = patients_df.reset_index()
        patients_df = ets.sample_independent_lof_censoring(patients_df, prob_lof_at_t=0.01 * np.ones(d_times))
        self.patients_df = ets.update_event_or_lof(patients_df)
        self.pgscv = PenaltyGridSearchCV()

    def test_penalty_grid_search_cross_validate(self):
        self.pgscv.cross_validate(full_df=self.patients_df,
                             l1_ratio=1,
                             n_splits=2,
                             penalizers=[0.0001, 0.02],
                             seed=0)


class TestPenaltyGridSearchCVExact(TestPenaltyGridSearchCV):

    def setUp(self):
        n_cov = 6
        beta1 = np.zeros(n_cov)
        beta1[:5] = (-0.3 * np.log([0.8, 3, 3, 2.5, 2]))
        beta2 = np.zeros(n_cov)
        beta2[:5] = (-0.3 * np.log([1, 3, 4, 3, 2]))

        real_coef_dict = {
            "alpha": {
                1: lambda t: -1.9 + 0.2 * np.log(t),
                2: lambda t: -1.9 + 0.2 * np.log(t)
            },
            "beta": {
                1: beta1,
                2: beta2
            }
        }
        n_patients = 400
        d_times = 4
        j_events = 2

        ets = EventTimesSampler(d_times=d_times, j_event_types=j_events)

        seed = 0
        means_vector = np.zeros(n_cov)
        covariance_matrix = 0.5 * np.identity(n_cov)
        clip_value = 1

        covariates = [f'Z{i + 1}' for i in range(n_cov)]

        patients_df = pd.DataFrame(data=pd.DataFrame(data=np.random.multivariate_normal(means_vector, covariance_matrix,
                                                                                        size=n_patients),
                                                     columns=covariates))
        patients_df.clip(lower=-1 * clip_value, upper=clip_value, inplace=True)
        patients_df = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
        patients_df.index.name = 'pid'
        patients_df = patients_df.reset_index()
        patients_df = ets.sample_independent_lof_censoring(patients_df, prob_lof_at_t=0.01 * np.ones(d_times))
        self.patients_df = ets.update_event_or_lof(patients_df)
        self.pgscv = PenaltyGridSearchCVExact()
