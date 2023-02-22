import pandas as pd
import numpy as np
from itertools import product
from .fitters import TwoStagesFitter
import warnings
from copy import deepcopy
pd.set_option("display.max_rows", 500)
warnings.filterwarnings('ignore')
from time import time
from typing import Union
slicer = pd.IndexSlice
from .evaluation import events_integrated_brier_score, global_brier_score, events_integrated_auc, global_auc


class PenaltyGridSearch(object):

    """ This class implements the penalty parameter grid search. """

    def __init__(self):
        self.l1_ratio = None
        self.penalizers = []
        self.seed = None
        self.meta_models = {}
        self.train_df = None
        self.test_df = None
        self.global_auc = {}
        self.integrated_auc = {}
        self.global_bs = {}
        self.integrated_bs = {}

    def evaluate(self,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 l1_ratio: float,
                 penalizers: list,
                 metrics: Union[list, str] = ['IBS', 'GBS', 'IAUC', 'GAUC'],
                 seed: Union[None, int] = None,
                 event_type_col: str = 'J',
                 duration_col: str = 'X',
                 pid_col: str = 'pid',
                 twostages_fit_kwargs: dict = {}) -> tuple:

        """
        This function implements model estimation using train_df and evaluation of the metrics using test_df to all the possible combinations of penalizers.

        Args:
            train_df (pd.DataFrame): training data for fitting the model.
            test_df (pd.DataFrame): testing data for evaluating the estimated model's performance.
            l1_ratio (float): regularization ratio for the CoxPHFitter (see lifelines.fitters.coxph_fitter.CoxPHFitter documentation).
            penalizers (list): penalizer options for each event (see lifelines.fitters.coxph_fitter.CoxPHFitter documentation).
            metrics (str, list): Evaluation metrics. Available metrics:
                                                    'IAUC': Integrated AUC (will be in PenaltyGridSearch.integrated_auc),
                                                    'GAUC': Global AUC (will be in PenaltyGridSearch.global_auc).
                                                    'IBS': Integrated Brier Score (will be in PenaltyGridSearch.integrated_bs),
                                                    'GBS': Global Brier Score (will be in PenaltyGridSearch.global_bs).
            seed (int): pseudo random seed number for numpy.random.seed()
            event_type_col (str): The event type column name (must be a column in df),
                                  Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in df).
            pid_col (str): Sample ID column name (must be a column in df).
            twostages_fit_kwargs (dict): keyword arguments to pass to the TwoStagesFitter.

        Returns:
            output (Tuple): Penalizers with best performance in terms of Global-AUC, if 'GAUC' is in metrics.

        """

        self.l1_ratio = l1_ratio
        self.penalizers = penalizers
        self.seed = seed
        np.random.seed(seed)

        for idp, penalizer in enumerate(penalizers):
            fit_beta_kwargs = {
                'model_kwargs': {
                    'penalizer': penalizer,
                    'l1_ratio': l1_ratio
                },
            }
            self.meta_models[penalizer] = TwoStagesFitter()
            print(f"Started estimating the coefficients for penalizer {penalizer} ({idp+1}/{len(penalizers)})")
            start = time()
            self.meta_models[penalizer].fit(df=train_df, fit_beta_kwargs=fit_beta_kwargs,
                                            pid_col=pid_col, event_type_col=event_type_col, duration_col=duration_col,
                                            **twostages_fit_kwargs)
            end = time()
            print(f"Finished estimating the coefficients for penalizer {penalizer} ({idp+1}/{len(penalizers)}), {int(end - start)} seconds")

        events = [j for j in sorted(train_df[event_type_col].unique()) if j != 0]
        grid = [penalizers for e in events]
        penalizers_combinations = list(product(*grid))

        for idc, combination in enumerate(penalizers_combinations):
            mixed_two_stages = self.get_mixed_two_stages_fitter(combination)

            pred_df = mixed_two_stages.predict_prob_events(test_df)

            for metric in metrics:
                if metric == 'IAUC':
                    self.integrated_auc[combination] = events_integrated_auc(pred_df, event_type_col=event_type_col,
                                                                                      duration_col=duration_col)
                elif metric == 'GAUC':
                    self.global_auc[combination] = global_auc(pred_df, event_type_col=event_type_col,
                                                                       duration_col=duration_col)
                elif metric == 'IBS':
                    self.integrated_bs[combination] = events_integrated_brier_score(pred_df,
                                                                                    event_type_col=event_type_col,
                                                                                    duration_col=duration_col)
                elif metric == 'GBS':
                    self.global_bs[combination] = global_brier_score(pred_df, event_type_col=event_type_col,
                                                                              duration_col=duration_col)

        output = self.convert_results_dict_to_df(self.global_auc).idxmax().values[0] if 'GAUC' in metrics else []
        return output

    def convert_results_dict_to_df(self, results_dict):
        """
        This function converts a results dictionary to a pd.DataFrame format.
        Args:
            results_dict: one of the class attributes: global_auc, integrated_auc, global_bs, integrated_bs.

        Returns:
            df (pd.DataFrame): Results in a pd.DataFrame format.
        """
        df = pd.DataFrame(results_dict.values(), index=pd.MultiIndex.from_tuples(results_dict.keys()))
        return df

    def get_mixed_two_stages_fitter(self, penalizers_combination: list) -> TwoStagesFitter:
        """
        This function creates a mixed TwoStagesFitter from the estimated meta models for a specific penalizers combination.

        Args:
            penalizers_combination (list): List with length equals to the number of competing events. The penalizers value to each of the events.
                                           Each of the values must be one of the values that was previously passed to the evaluate() method.

        Returns:
            mixed_two_stages (pydts.fitters.TwoStagesFitter): TwoStagesFitter for the required penalty combination.
        """
        _validate_estimated_value = [p for p in penalizers_combination if p not in list(self.meta_models.keys())]
        assert len(_validate_estimated_value) == 0, \
               f"Values {_validate_estimated_value} were note estimated. All the penalizers in penalizers_combination must be estimated using evaluate() before a mixed combination can be generated."

        events = self.meta_models[penalizers_combination[0]].events
        event_type_col = self.meta_models[penalizers_combination[0]].event_type_col
        mixed_two_stages = TwoStagesFitter()
        for ide, event in enumerate(sorted(events)):
            if ide == 0:
                mixed_two_stages.covariates = self.meta_models[penalizers_combination[ide]].covariates
                mixed_two_stages.duration_col = self.meta_models[penalizers_combination[ide]].duration_col
                mixed_two_stages.event_type_col = self.meta_models[penalizers_combination[ide]].event_type_col
                mixed_two_stages.events = self.meta_models[penalizers_combination[ide]].events
                mixed_two_stages.pid_col = self.meta_models[penalizers_combination[ide]].pid_col
                mixed_two_stages.times = self.meta_models[penalizers_combination[ide]].times

            mixed_two_stages.beta_models[event] = deepcopy(self.meta_models[penalizers_combination[ide]].beta_models[event])
            mixed_two_stages.event_models[event] = []
            mixed_two_stages.event_models[event].append(deepcopy(self.meta_models[penalizers_combination[ide]].beta_models[event]))

            event_alpha = self.meta_models[penalizers_combination[ide]].alpha_df.copy()
            event_alpha = event_alpha[event_alpha[event_type_col] == event]
            mixed_two_stages.alpha_df = pd.concat([mixed_two_stages.alpha_df, event_alpha])
            mixed_two_stages.event_models[event].append(event_alpha)

        return mixed_two_stages
