import unittest
import numpy as np
import pandas as pd
from src.pydts.data_generation import EventTimesSampler
from src.pydts.evaluation import prediction_error
from src.pydts.fitters import TwoStagesFitter


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        n_cov = 15
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
        n_patients = 5000
        d_times = 15
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
        patients_df = ets.sample_independent_lof_censoring(patients_df, prob_lof_at_t=0.01 * np.ones_like(ets.times))
        self.patients_df = ets.update_event_or_lof(patients_df)

    def test_prediction_error(self):
        L1_regularized_fitter = TwoStagesFitter()

        fit_beta_kwargs = {
            'model_kwargs': {
                'penalizer': 0.01,
                'l1_ratio': 1
            }
        }

        L1_regularized_fitter.fit(df=self.patients_df.drop(['C', 'T'], axis=1),
                                  fit_beta_kwargs=fit_beta_kwargs,
                                  nb_workers=1)

        pred_df = L1_regularized_fitter.predict_cumulative_incident_function(self.patients_df)
        pe = prediction_error(pred_df)
        print(pe)
