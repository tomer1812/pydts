import unittest
from src.pydts.data_generation import EventTimesSampler
from src.pydts.evaluation import *
from src.pydts.fitters import TwoStagesFitter
import numpy as np
import pandas as pd


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        n_cov = 5
        beta1 = (-0.5 * np.log([0.8, 2, 3, 2.5, 1.2]))
        beta2 = (-0.5 * np.log([1, 3, 2, 1.5, 2.7]))

        self.real_coef_dict = {
            "alpha": {
                1: lambda t: -2.8 - 0.1 * np.log(t),
                2: lambda t: -3 - 0.1 * np.log(t)
            },
            "beta": {
                1: beta1,
                2: beta2
            }
        }
        n_patients = 15000
        d_times = 15
        j_events = 2

        self.ets = EventTimesSampler(d_times=d_times, j_event_types=j_events)

        seed = 0
        means_vector = np.zeros(n_cov)
        covariance_matrix = 0.5 * np.identity(n_cov)
        clip_value = 1

        self.covariates = [f'Z{i + 1}' for i in range(n_cov)]

        patients_df = pd.DataFrame(data=pd.DataFrame(data=np.random.multivariate_normal(means_vector, covariance_matrix,
                                                                                        size=n_patients),
                                                     columns=self.covariates))
        patients_df.clip(lower=-1 * clip_value, upper=clip_value, inplace=True)
        patients_df = self.ets.sample_event_times(patients_df,
                                                  hazard_coefs=self.real_coef_dict,
                                                  seed=seed)
        patients_df.index.name = 'pid'
        patients_df = patients_df.reset_index()
        patients_df = self.ets.sample_independent_lof_censoring(patients_df,
                                                                prob_lof_at_t=0.01 * np.ones(d_times))
        self.patients_df = self.ets.update_event_or_lof(patients_df)
        self.patients_df = self.patients_df.apply(pd.to_numeric, errors="raise")

    def test_event_specific_auc(self):
        fitter = TwoStagesFitter()
        fitter.fit(df=self.patients_df.drop(['C', 'T'], axis=1), nb_workers=1)
        pred_df = fitter.predict_prob_event_j_all(self.patients_df, event=1)
        esauc = event_specific_auc_at_t(pred_df, event=1, t=9)
        print(esauc)

    def test_event_specific_auc_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t], axis=1)
        esauc = event_specific_auc_at_t(pred_df, event=event, t=9)
        print(esauc)

    def test_event_specific_brier_score_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t], axis=1)
        esauc = event_specific_brier_score_at_t(pred_df, event=event, t=9)
        print(esauc)

    def test_event_specific_auc_all_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t], axis=1)
        esauc = event_specific_auc_at_t_all(pred_df, event=event)
        print(esauc)

    def test_event_specific_brier_score_all_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t], axis=1)
        esbs = event_specific_brier_score_at_t_all(pred_df, event=event)
        print(esbs)

    def test_event_specific_integrated_auc_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t], axis=1)
        esauc = event_specific_integrated_auc(pred_df, event=event)
        print(esauc)

    def test_event_specific_integrated_brier_score_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t], axis=1)
        esbs = event_specific_integrated_brier_score(pred_df, event=event)
        print(esbs)

    def test_event_specific_integrated_auc_with_weights_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t], axis=1)
        weights = pd.Series((1/self.ets.d_times)*np.ones(self.ets.d_times),
                            index=range(1, self.ets.d_times+1))
        esauc = event_specific_integrated_auc(pred_df, event=event, weights=weights)
        print(esauc)

    def test_event_specific_integrated_brier_score_with_weights_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t], axis=1)
        weights = pd.Series((1/self.ets.d_times)*np.ones(self.ets.d_times),
                            index=range(1, self.ets.d_times+1))
        esbs = event_specific_integrated_brier_score(pred_df, event=event, weights=weights)
        print(esbs)

    def test_global_auc_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t_1 = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        event = 2
        probs_j_at_t_2 = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t_1, probs_j_at_t_2], axis=1)
        esauc = global_auc(pred_df)
        print(esauc)

    def test_global_brier_score_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t_1 = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        event = 2
        probs_j_at_t_2 = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t_1, probs_j_at_t_2], axis=1)
        esbs = global_brier_score(pred_df)
        print(esbs)

    def test_events_integrated_brier_score_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t_1 = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        event = 2
        probs_j_at_t_2 = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t_1, probs_j_at_t_2], axis=1)
        esbs = events_integrated_brier_score(pred_df)
        print(esbs)

    def test_events_integrated_auc_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t_1 = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        event = 2
        probs_j_at_t_2 = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t_1, probs_j_at_t_2], axis=1)
        esbs = events_integrated_auc(pred_df)
        print(esbs)

    def test_events_auc_at_t_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t_1 = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        event = 2
        probs_j_at_t_2 = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t_1, probs_j_at_t_2], axis=1)
        esauc = events_auc_at_t(pred_df)
        print(esauc)

    def test_events_brier_score_at_t_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t_1 = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        event = 2
        probs_j_at_t_2 = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t_1, probs_j_at_t_2], axis=1)
        esbs = events_brier_score_at_t(pred_df)
        print(esbs)

    def test_event_specific_weights_perfect_model(self):
        cov_df = self.patients_df[self.covariates]
        hazards = self.ets.calculate_hazards(cov_df, self.real_coef_dict, events=self.ets.events)
        overall_survival = self.ets.calculate_overall_survival(hazards)
        probs_j_at_t = self.ets.calculate_prob_event_at_t(hazards, overall_survival)
        event = 1
        probs_j_at_t = pd.DataFrame(probs_j_at_t[event-1].values,
                                    columns=[f'prob_j{event}_at_t{t}'
                                             for t in range(1, self.ets.d_times+1)])
        pred_df = pd.concat([self.patients_df, probs_j_at_t], axis=1)
        weights = event_specific_weights(pred_df, event=event)
        print(weights)

    def test_event_specific_auc_simple_case_j1_t5(self):
        # d=6 times, n=7, J=2
        d = 6
        pred_df = pd.DataFrame(data=[
            [1, 3, 0.1, 0.1, 0.1, 0.07, 0.07, 0.01],
            [1, 5, 0.05, 0.05, 0.05, 0.07, 0.1, 0.15],
            [0, 7, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0, 2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0, 7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [2, 4, 0.04, 0.04, 0.04, 0.07, 0.08, 0.1],
            [2, 6, 0.04, 0.04, 0.07, 0.07, 0.12, 0.12],
        ], columns=['J', 'X'] + [f'prob_j1_at_t{t}' for t in range(1, d+1)])
        esauc = event_specific_auc_at_t(pred_df, event=1, t=5)
        assert esauc == 0.5

    def test_event_specific_auc_simple_case_j1_t3(self):
        # d=6 times, n=7, J=2
        d = 6
        pred_df = pd.DataFrame(data=[
            [1, 3, 0.1, 0.1, 0.1, 0.07, 0.07, 0.01],
            [1, 5, 0.05, 0.05, 0.05, 0.07, 0.1, 0.15],
            [0, 7, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0, 2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0, 7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [2, 4, 0.04, 0.04, 0.04, 0.07, 0.08, 0.1],
            [2, 6, 0.04, 0.04, 0.07, 0.07, 0.12, 0.12],
        ], columns=['J', 'X'] + [f'prob_j1_at_t{t}' for t in range(1, d+1)])
        esauc = event_specific_auc_at_t(pred_df, event=1, t=3)
        assert esauc == 0.9

    def test_cause_specific_auc_weights(self):
        # d=6 times, n=7, J=2
        d = 6
        pred_df = pd.DataFrame(data=[
            [0, 1, 0.1, 0.1, 0.1, 0.07, 0.07, 0.01],
            [1, 3, 0.1, 0.1, 0.1, 0.07, 0.07, 0.01],
            [1, 5, 0.05, 0.05, 0.05, 0.07, 0.1, 0.15],
            [0, 7, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            [0, 2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0, 7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [2, 4, 0.04, 0.04, 0.04, 0.07, 0.08, 0.1],
            [2, 6, 0.04, 0.04, 0.07, 0.07, 0.12, 0.12],
        ], columns=['J', 'X'] + [f'prob_j1_at_t{t}' for t in range(1, d+1)])

        event_specific_weights(pred_df, event=1)
        print('x')
