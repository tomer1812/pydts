import unittest
from pydts.examples_utils.generate_simulations_data import generate_quick_start_df
from pydts.fitters import DataExpansionFitter, TwoStagesFitter


class TestFitters(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = generate_quick_start_df(n_patients=1000, n_cov=5, d_times=30, j_events=2, pid_col='pid', seed=0)

    def test_e2e_DataExpansionFitter(cls):
        m = DataExpansionFitter()
        m.fit(df=cls.df.drop(['C', 'T'], axis=1))
        m.print_summary()

    def test_e2e_TwoStagesFitter(cls):
        m2 = TwoStagesFitter()
        try:
            m2.fit(cls.df.drop(['C', 'T'], axis=1))
        except RuntimeError as e:
            temp_df = cls.df.copy()
            temp_df['X'].clip(upper=25, inplace=True)
            m2.fit(temp_df.drop(['C', 'T'], axis=1))
        m2.print_summary()


if __name__ == '__main__':
    unittest.main()
