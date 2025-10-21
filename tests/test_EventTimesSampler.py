import unittest
from src.pydts.data_generation import EventTimesSampler
from src.pydts.fitters import TwoStagesFitter, DataExpansionFitter
from time import time
import numpy as np
import pandas as pd
slicer = pd.IndexSlice
COEF_COL = '   coef   '
STDERR_COL = ' std err '


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
        result = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
        
        # Validate output structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), n_patients)
        
        # Check required columns exist
        self.assertIn('T', result.columns)  # Event time
        self.assertIn('J', result.columns)  # Event type
        
        # Validate event types are within expected range
        self.assertTrue(result['J'].isin([0, 1, 2]).all())
        
        # Validate event times are positive and within reasonable range
        self.assertTrue((result['T'] > 0).all())
        self.assertTrue((result['T'] <= (ets.d_times+1)).all())

    def test_sample_3_events_4_covariates(self):
        n_cov = 4
        n_patients = 1000
        seed = 0
        covariates = [f'Z{i + 1}' for i in range(n_cov)]
        patients_df = pd.DataFrame(data=np.random.uniform(low=0.0, high=1.0, size=[n_patients, n_cov]),
                                   columns=covariates)

        ets = EventTimesSampler(d_times=15, j_event_types=3)  # Fixed: should be 3 events, not 2
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
        result = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
        
        # Validate output structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), n_patients)
        
        # Check required columns exist
        self.assertIn('T', result.columns)  # Event time
        self.assertIn('J', result.columns)  # Event type
        
        # Validate event types are within expected range (1, 2, 3)
        self.assertTrue(result['J'].isin([0, 1, 2, 3]).all())
        
        # Validate event times are positive and within reasonable range
        self.assertTrue((result['T'] > 0).all())
        self.assertTrue((result['T'] <= (ets.d_times+1)).all())

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
                0: lambda t: -1 - 0.3 * np.log(t),
            },
            "beta": {
                0: -np.log([0.8, 3, 3, 2.5, 2]),
            }
        }
        result = ets.sample_hazard_lof_censoring(patients_df, censoring_coef_dict, seed)
        
        # Validate hazard censoring was applied
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), n_patients)
        self.assertIn('C', result.columns)  # Censoring time
        
        # Check censoring times are valid
        self.assertTrue((result['C'] > 0).all())
        self.assertTrue((result['C'] <= (ets.d_times+1)).all())
        
        # Check that original covariates are preserved
        for cov in covariates:
            self.assertIn(cov, result.columns)

    def test_sample_independent_censoring(self):
        n_cov = 4
        n_patients = 1000
        covariates = [f'Z{i + 1}' for i in range(n_cov)]
        patients_df = pd.DataFrame(data=np.random.uniform(low=0.0, high=1.0, size=[n_patients, n_cov]),
                                   columns=covariates)

        d_times = 15
        ets = EventTimesSampler(d_times=15, j_event_types=2)
        result = ets.sample_independent_lof_censoring(patients_df, prob_lof_at_t=0.03 * np.ones(d_times))
        
        # Validate censoring was applied
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), n_patients)
        self.assertIn('C', result.columns)  # Censoring time
        
        # Check censoring times are valid
        self.assertTrue((result['C'] > 0).all())
        self.assertTrue((result['C'] <= (d_times+1)).all())

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
                0: lambda t: -1 - 0.3 * np.log(t),
            },
            "beta": {
                0: -np.log([0.8, 3, 3, 2.5, 2]),
            }
        }

        patients_df = ets.sample_hazard_lof_censoring(patients_df, censoring_coef_dict, seed=seed)
        result = ets.update_event_or_lof(patients_df)
        
        # Validate output structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), n_patients)
        
        # Check that X column exists (observed time)
        self.assertIn('X', result.columns)
        
        # Validate observed times are positive and within reasonable range
        self.assertTrue((result['X'] > 0).all())
        self.assertTrue((result['X'] <= (ets.d_times+1)).all())
        
        # Check that event indicators are valid
        if 'J' in result.columns:
            # J should be 0 for censored observations or valid event types
            valid_events = list(range(ets.j_event_types + 1))  # 0 for censoring, 1-j for events
            self.assertTrue(result['J'].isin(valid_events).all())

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
                    0: lambda t: -1 - 0.3 * np.log(t),
                },
                "beta": {
                    0: -np.log([0.8, 3, 3, 2.5, 2]),
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

    # def test_sample_and_fit_from_multinormal(self):
    #     # real_coef_dict = {
    #     #     "alpha": {
    #     #         1: lambda t: -3 - 3 * np.log(t),
    #     #         2: lambda t: -3 - 0.15 * np.log(t)
    #     #     },
    #     #     "beta": {
    #     #         # 1: [0.3, -1, -0.5, 0.7, 0.9],
    #     #         # 2: [-0.5, 0.5, 0.7, -0.7, 0.5]
    #     #         1: -np.log([0.8, 3, 3, 2.5, 2]),
    #     #         2: -np.log([1, 3, 4, 3, 2])
    #     #     }
    #     # }
    #
    #     real_coef_dict = {
    #         "alpha": {
    #             1: lambda t: -1.75 - 0.3 * np.log(t),
    #             2: lambda t: -1.5 - 0.15 * np.log(t)
    #         },
    #         "beta": {
    #             1: -0.5*np.log([0.8, 3, 3, 2.5, 2]),
    #             2: -0.5*np.log([1, 3, 4, 3, 2])
    #         }
    #     }
    #
    #     censoring_hazard_coef_dict = {
    #         "alpha": {
    #             0: lambda t: -1.75 - 0.3 * np.log(t),
    #         },
    #         "beta": {
    #             0: -0.5*np.log([1, 3, 4, 3, 2]),
    #         }
    #     }
    #
    #     n_patients = 15000
    #     n_cov = 5
    #     d_times = 50
    #     j_events = 2
    #     clip_value = 1
    #     means_vector = np.zeros(n_cov)
    #     covariance_matrix = 0.4*np.identity(n_cov)
    #
    #     ets = EventTimesSampler(d_times=d_times, j_event_types=j_events)
    #     seed = 0
    #     covariates = [f'Z{i + 1}' for i in range(n_cov)]
    #
    #     np.random.seed(seed)
    #
    #     patients_df = pd.DataFrame(data=pd.DataFrame(
    #         data=np.random.multivariate_normal(means_vector, covariance_matrix, size=n_patients),
    #         columns=covariates))
    #     # patients_df = pd.DataFrame(data=pd.DataFrame(
    #     #     data=np.random.uniform(0, 1, size=[n_patients, n_cov]),
    #     #     columns=covariates))
    #
    #     patients_df.clip(lower=-1*clip_value, upper=clip_value, inplace=True)
    #     patients_df = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
    #     patients_df = ets.sample_hazard_lof_censoring(patients_df,
    #                                                  censoring_hazard_coefs=censoring_hazard_coef_dict,
    #                                                  seed=seed + 1, events=[0])
    #     patients_df = ets.update_event_or_lof(patients_df)
    #     patients_df.index.name = 'pid'
    #     patients_df = patients_df.reset_index()
    #
    #     # Two step fitter
    #     new_fitter = TwoStagesFitter()
    #     two_step_start = time()
    #     new_fitter.fit(df=patients_df.drop(['C', 'T'], axis=1), nb_workers=1)
    #     two_step_end = time()
    #
    #     # Lee et al fitter
    #     lee_fitter = DataExpansionFitter()
    #     lee_start = time()
    #     lee_fitter.fit(df=patients_df.drop(['C', 'T'], axis=1))
    #     lee_end = time()
    #     lee_alpha_results = lee_fitter.get_alpha_df().loc[:,
    #                     slicer[:, [COEF_COL, STDERR_COL]]].unstack().to_frame()
    #     lee_beta_results = lee_fitter.get_beta_SE().loc[:, slicer[:, [COEF_COL, STDERR_COL]]].unstack().to_frame()
    #
    #     # Save results only if both fitters were successful
    #     two_step_fit_time = two_step_end - two_step_start
    #     lee_fit_time = lee_end - lee_start
    #
    #     two_step_alpha_results = new_fitter.alpha_df[['J', 'X', 'alpha_jt']].set_index(['J', 'X'])
    #     two_step_beta_results = new_fitter.get_beta_SE().unstack().to_frame()
    #     print('x')

    # def test_sample_and_fit_normal(self):
    #
    #     real_coef_dict = {
    #         "alpha": {
    #             1: lambda t: -3 - 0.3 * np.log(t),
    #             2: lambda t: -3 - 0.15 * np.log(t)
    #         },
    #         "beta": {
    #             1: -np.log([0.8, 3, 3, 2.5, 2]),
    #             2: -np.log([1, 3, 4, 3, 2])
    #         }
    #     }
    #
    #     n_patients = 10000
    #     n_cov = 5
    #     d_times = 50
    #     j_events = 2
    #
    #     ets = EventTimesSampler(d_times=d_times, j_event_types=j_events)
    #     seed = 0
    #     covariates = [f'Z{i + 1}' for i in range(n_cov)]
    #
    #     np.random.seed(seed)
    #
    #     patients_df = pd.DataFrame(data=pd.DataFrame(
    #         data=np.random.uniform(0, 1, size=[n_patients, n_cov]),
    #         columns=covariates))
    #     patients_df = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
    #     patients_df['X'] = patients_df['T']
    #     patients_df['C'] = 51
    #     patients_df.index.name = 'pid'
    #     patients_df = patients_df.reset_index()
    #
    #     # Two step fitter
    #     new_fitter = TwoStagesFitter()
    #     two_step_start = time()
    #     new_fitter.fit(df=patients_df.drop(['C', 'T'], axis=1), nb_workers=1)  # , x0=-3
    #     two_step_end = time()
    #
    #     # Lee et al fitter
    #     lee_fitter = DataExpansionFitter()
    #     lee_start = time()
    #     lee_fitter.fit(df=patients_df.drop(['C', 'T'], axis=1))
    #     lee_end = time()
    #     lee_alpha_results = lee_fitter.get_alpha_df().loc[:,
    #                     slicer[:, [COEF_COL, STDERR_COL]]].unstack().to_frame()
    #     lee_beta_results = lee_fitter.get_beta_SE().loc[:, slicer[:, [COEF_COL, STDERR_COL]]].unstack().to_frame()
    #
    #     # Save results only if both fitters were successful
    #     two_step_fit_time = two_step_end - two_step_start
    #     lee_fit_time = lee_end - lee_start
    #
    #     two_step_alpha_results = new_fitter.alpha_df[['J', 'X', 'alpha_jt']].set_index(['J', 'X'])
    #     two_step_beta_results = new_fitter.get_beta_SE().unstack().to_frame()
    #     print('x')


    def test_raise_negative_values_overall_survival_assertion(self):
        with self.assertRaises(ValueError):
            real_coef_dict = {
                "alpha": {
                    1: lambda t: -9 + 3 * np.log(t),
                    2: lambda t: -7 + 2.5 * np.log(t)
                },
                "beta": {
                    1: [1.3, 1.7, -1.5, 0.5, 1.6],
                    2: [-1.5, 1.5, 1.8, -1, 1.2]
                }
            }

            censoring_hazard_coef_dict = {
                "alpha": {
                    0: lambda t: -8 + 2.1 * np.log(t),
                },
                "beta": {
                    0: [2, 1, -1.5, 1.5, -1.3],
                }
            }

            n_patients = 25000
            n_cov = 5
            d_times = 12
            j_events = 2
            means_vector = np.zeros(n_cov)
            covariance_matrix = np.identity(n_cov)

            ets = EventTimesSampler(d_times=d_times, j_event_types=j_events)
            seed = 0
            covariates = [f'Z{i + 1}' for i in range(n_cov)]

            np.random.seed(seed)

            patients_df = pd.DataFrame(data=pd.DataFrame(
                data=np.random.multivariate_normal(means_vector, covariance_matrix, size=n_patients),
                columns=covariates))

            patients_df = ets.sample_event_times(patients_df, hazard_coefs=real_coef_dict, seed=seed)
            
            # This should raise ValueError due to negative survival probabilities
            patients_df = ets.sample_hazard_lof_censoring(patients_df, 
                                                         censoring_hazard_coefs=censoring_hazard_coef_dict, 
                                                         seed=seed + 1)
            ets.update_event_or_lof(patients_df)

