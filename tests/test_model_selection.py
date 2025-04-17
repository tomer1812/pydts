import unittest
import numpy as np
import pandas as pd
from src.pydts.data_generation import EventTimesSampler
from src.pydts.model_selection import PenaltyGridSearch, PenaltyGridSearchExact
from sklearn.model_selection import train_test_split
from src.pydts.fitters import TwoStagesFitter, TwoStagesFitterExact


class TestPenaltyGridSearch(unittest.TestCase):

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
        self.pgs = PenaltyGridSearch()

    def test_get_mixed_two_stages_fitter(self):
        train_df, test_df = train_test_split(self.patients_df.drop(['C', 'T'], axis=1), random_state=0)
        self.pgs.evaluate(train_df=train_df,
                     test_df=test_df,
                     l1_ratio=1,
                     metrics=[],
                     penalizers=[0.005, 0.02],
                     seed=0)
        mixed_two_stages = self.pgs.get_mixed_two_stages_fitter([0.005, 0.02])

        mixed_params = pd.concat([mixed_two_stages.beta_models[1].params_,
                                  mixed_two_stages.beta_models[2].params_], axis=1)

        fit_beta_kwargs = {
            'model_kwargs': {
                1: {'penalizer': 0.005, 'l1_ratio': 1},
                2: {'penalizer': 0.02, 'l1_ratio': 1},
            }
        }

        two_stages_fitter = TwoStagesFitter()
        two_stages_fitter.fit(df=train_df, fit_beta_kwargs=fit_beta_kwargs)

        two_stages_params = pd.concat([two_stages_fitter.beta_models[1].params_,
                                       two_stages_fitter.beta_models[2].params_], axis=1)

        pd.testing.assert_frame_equal(mixed_params, two_stages_params)
        pd.testing.assert_frame_equal(mixed_two_stages.alpha_df, two_stages_fitter.alpha_df)

    def test_assertion_get_mixed_two_stages_fitter_not_included(self):
        with self.assertRaises(AssertionError):

            train_df, test_df = train_test_split(self.patients_df.drop(['C', 'T'], axis=1), random_state=0)
            self.pgs.evaluate(train_df=train_df,
                         test_df=test_df,
                         l1_ratio=1,
                         metrics=[],
                         penalizers=[0.005, 0.02],
                         seed=0)

            mixed_two_stages = self.pgs.get_mixed_two_stages_fitter([0.1, 0.02])

    def test_assertion_get_mixed_two_stages_fitter_empty(self):
        with self.assertRaises(AssertionError):
            mixed_two_stages = self.pgs.get_mixed_two_stages_fitter([0.001, 0.02])

    def test_evaluate(self):
        train_df, test_df = train_test_split(self.patients_df.drop(['C', 'T'], axis=1), random_state=0)
        idx_max = self.pgs.evaluate(train_df=train_df, test_df=test_df, l1_ratio=1,
                               penalizers=[0.0001, 0.005, 0.02],
                               seed=0)

    def test_convert_results_dict_to_df(self):
        train_df, test_df = train_test_split(self.patients_df.drop(['C', 'T'], axis=1), random_state=0)
        idx_max = self.pgs.evaluate(train_df=train_df, test_df=test_df, l1_ratio=1,
                               penalizers=[0.0001, 0.005],
                               seed=0)
        self.pgs.convert_results_dict_to_df(self.pgs.global_bs)


class TestPenaltyGridSearchExact(TestPenaltyGridSearch):

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
        n_patients = 300
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
        self.pgs = PenaltyGridSearchExact()

    def test_get_mixed_two_stages_fitter(self):
        train_df, test_df = train_test_split(self.patients_df.drop(['C', 'T'], axis=1), random_state=0)
        self.pgs.evaluate(train_df=train_df,
                     test_df=test_df,
                     l1_ratio=1,
                     metrics=[],
                     penalizers=[0.005, 0.02],
                     seed=0)
        mixed_two_stages = self.pgs.get_mixed_two_stages_fitter([0.005, 0.02])

        mixed_params = pd.concat([mixed_two_stages.beta_models[1].params,
                                  mixed_two_stages.beta_models[2].params], axis=1)


        fit_beta_kwargs = {
            'model_fit_kwargs': {
                1: {
                        'alpha': 0.005,
                        'L1_wt': 1
                },
                2: {
                        'alpha': 0.02,
                        'L1_wt': 1
                }
            }
        }

        two_stages_fitter = TwoStagesFitterExact()
        two_stages_fitter.fit(df=train_df, fit_beta_kwargs=fit_beta_kwargs)

        two_stages_params = pd.concat([two_stages_fitter.beta_models[1].params,
                                       two_stages_fitter.beta_models[2].params], axis=1)

        pd.testing.assert_frame_equal(mixed_params, two_stages_params)
        pd.testing.assert_frame_equal(mixed_two_stages.alpha_df, two_stages_fitter.alpha_df)
