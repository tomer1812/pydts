import unittest

import numpy as np

from src.pydts.fitters import repetitive_fitters


class TestRepFitters(unittest.TestCase):
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

        self.n_patients = 10000
        self.n_cov = 5
        self.d = 15

    # def test_fit_function_case_successful(self):
    #     _ = repetitive_fitters(rep=5, n_patients=self.n_patients, n_cov=self.n_cov,
    #                            d_times=self.d, j_events=2, pid_col='pid', verbose=0,
    #                            allow_fails=20, real_coef_dict=self.real_coef_dict,
    #                            censoring_prob=.8)
    #
    # def test_fit_not_sending_coef(self):
    #     # event where fit are sent without real coefficient dict
    #     with self.assertRaises(AssertionError):
    #         _ = repetitive_fitters(rep=5, n_patients=self.n_patients, n_cov=self.n_cov,
    #                                d_times=self.d, j_events=2, pid_col='pid',  verbose=0,
    #                                allow_fails=20, censoring_prob=.8)
    #
    # def test_fit_repetitive_function_case_j_event_not_equal_to_real_coef(self):
    #     # event where fit are sent with wrong j_events, causing except to print it,
    #     # but not deal with value error in the end
    #     with self.assertRaises(ValueError):
    #         _ = repetitive_fitters(rep=2, n_patients=self.n_patients, n_cov=self.n_cov,
    #                                d_times=self.d, j_events=3, pid_col='pid', verbose=0,
    #                                allow_fails=0, real_coef_dict=self.real_coef_dict,
    #                                censoring_prob=.8)
    #
    # def test_fit_function_case_second_model_is_not_twoStages(self):
    #     from src.pydts.fitters import DataExpansionFitter
    #     _ = repetitive_fitters(rep=2, n_patients=self.n_patients, n_cov=self.n_cov,
    #                            d_times=self.d, j_events=2, pid_col='pid',
    #                            model2=DataExpansionFitter, verbose=0,
    #                            allow_fails=20, real_coef_dict=self.real_coef_dict,
    #                            censoring_prob=.8)