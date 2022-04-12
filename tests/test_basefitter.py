import unittest

import numpy as np

from src.pydts.base_fitters import BaseFitter, ExpansionBasedFitter
from src.pydts.examples_utils.generate_simulations_data import generate_quick_start_df


class TestBaseFitter(unittest.TestCase):
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

        self.base_fitter = BaseFitter()
        self.expansion_fitter = ExpansionBasedFitter()

    def test_base_fit(self):
        with self.assertRaises(NotImplementedError):
            self.base_fitter.fit(self.df)

    def test_base_predict(self):
        with self.assertRaises(NotImplementedError):
            self.base_fitter.predict(self.df)

    def test_base_evaluate(self):
        with self.assertRaises(NotImplementedError):
            self.base_fitter.evaluate(self.df)

    def test_base_print_summary(self):
        with self.assertRaises(NotImplementedError):
            self.base_fitter.print_summary()

    def test_expansion_predict_hazard_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.expansion_fitter.predict_hazard_jt(self.df, event=1, t=100)