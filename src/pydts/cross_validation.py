import pandas as pd
import numpy as np
from .fitters import TwoStagesFitter
import warnings
from copy import deepcopy
from sklearn.model_selection import KFold
pd.set_option("display.max_rows", 500)
warnings.filterwarnings('ignore')
slicer = pd.IndexSlice
from typing import Optional, List, Union
import psutil
from .evaluation import events_brier_score_at_t, events_integrated_brier_score, global_brier_score, \
    events_integrated_auc, global_auc, events_auc_at_t
from .model_selection import PenaltyGridSearch
from time import time

WORKERS = psutil.cpu_count(logical=False)


class TwoStagesCV(object):
    """
    This class implements K-fold cross-validation using TwoStagesFitters
    """

    def __init__(self):
        self.models = {}
        self.test_pids = {}
        self.results = pd.DataFrame()
        self.global_auc = {}
        self.integrated_auc = {}
        self.global_bs = {}
        self.integrated_bs = {}

    def cross_validate(self,
                       full_df: pd.DataFrame,
                       n_splits: int = 5,
                       shuffle: bool = True,
                       seed: Union[int, None] = None,
                       fit_beta_kwargs: dict = {},
                       covariates=None,
                       event_type_col: str = 'J',
                       duration_col: str = 'X',
                       pid_col: str = 'pid',
                       x0: Union[np.array, int] = 0,
                       verbose: int = 2,
                       nb_workers: int = WORKERS,
                       metrics=['BS', 'IBS', 'GBS', 'AUC', 'IAUC', 'GAUC']):

        """
        This method implements K-fold cross-validation using TwoStagesFitters and full_df data.
        Args:
            full_df (pd.DataFrame): Data to cross validate.
            n_splits (int): Number of folds, defaults to 5.
            shuffle (boolean): Shuffle samples before splitting to folds. Defaults to True.
            seed: Pseudo-random seed to KFold instance. Defaults to None.
            fit_beta_kwargs (dict, Optional): Keyword arguments to pass on to the estimation procedure.
                                              If different model for beta is desired, it can be defined here.
            covariates (list): list of covariates to be used in estimating the regression coefficients.
            event_type_col (str): The event type column name (must be a column in df),
                                  Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in full_df).
            pid_col (str): Sample ID column name (must be a column in full_df).
            x0 (Union[numpy.array, int], Optional): initial guess to pass to scipy.optimize.minimize function
            verbose (int, Optional): The verbosity level of pandaallel
            nb_workers (int, Optional): The number of workers to pandaallel. If not sepcified, defaults to the number of workers available.
            metrics (str, list): Evaluation metrics. Available metrics:
                                                    'AUC': AUC at t (will be added to TwoStagesCV.results),
                                                    'IAUC': Integrated AUC (will be in TwoStagesCV.integrated_auc),
                                                    'GAUC': Global AUC (will be in TwoStagesCV.global_auc).
                                                    'BS': Brier score at t (will be added to TwoStagesCV.results),
                                                    'IBS': Integrated Brier Score (will be in TwoStagesCV.integrated_bs),
                                                    'GBS': Global Brier Score (will be in TwoStagesCV.global_bs).

        Returns:
            Results (pd.DataFrame): Cross validation metrics results
        """

        if isinstance(metrics, str):
            metrics = [metrics]

        self.models = {}
        self.kfold_cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

        if 'C' in full_df.columns:
            full_df = full_df.drop(['C'], axis=1)
        if 'T' in full_df.columns:
            full_df = full_df.drop(['T'], axis=1)

        for i_fold, (train_index, test_index) in enumerate(self.kfold_cv.split(full_df)):
            self.test_pids[i_fold] = full_df.iloc[test_index][pid_col].values
            train_df, test_df = full_df.iloc[train_index], full_df.iloc[test_index]
            fold_fitter = TwoStagesFitter()
            print(f'Fitting fold {i_fold+1}/{n_splits}')
            fold_fitter.fit(df=train_df,
                            covariates=covariates,
                            event_type_col=event_type_col,
                            duration_col=duration_col,
                            pid_col=pid_col,
                            x0=x0,
                            fit_beta_kwargs=fit_beta_kwargs,
                            verbose=verbose,
                            nb_workers=nb_workers)

            self.models[i_fold] = deepcopy(fold_fitter)

            pred_df = self.models[i_fold].predict_prob_events(test_df)

            for metric in metrics:
                if metric == 'IAUC':
                    self.integrated_auc[i_fold] = events_integrated_auc(pred_df, event_type_col=event_type_col,
                                                                        duration_col=duration_col)
                elif metric == 'GAUC':
                    self.global_auc[i_fold] = global_auc(pred_df, event_type_col=event_type_col,
                                                                  duration_col=duration_col)
                elif metric == 'IBS':
                    self.integrated_bs[i_fold] = events_integrated_brier_score(pred_df, event_type_col=event_type_col,
                                                                                        duration_col=duration_col)
                elif metric == 'GBS':
                    self.global_bs[i_fold] = global_brier_score(pred_df, event_type_col=event_type_col,
                                                                         duration_col=duration_col)
                elif metric == 'AUC':
                    tmp_res = events_auc_at_t(pred_df, event_type_col=event_type_col,
                                                       duration_col=duration_col)
                    tmp_res = pd.concat([tmp_res], keys=[i_fold], names=['fold'])
                    tmp_res = pd.concat([tmp_res], keys=[metric], names=['metric'])
                    self.results = pd.concat([self.results, tmp_res], axis=0)
                elif metric == 'BS':
                    tmp_res = events_brier_score_at_t(pred_df, event_type_col=event_type_col,
                                                               duration_col=duration_col)
                    tmp_res = pd.concat([tmp_res], keys=[i_fold], names=['fold'])
                    tmp_res = pd.concat([tmp_res], keys=[metric], names=['metric'])
                    self.results = pd.concat([self.results, tmp_res], axis=0)

        return self.results


class PenaltyGridSearchCV(object):
    """
    This class implements K-fold cross-validation of the PenaltyGridSearch
    """

    def __init__(self):
        self.folds_grids = {}
        self.test_pids = {}
        self.global_auc = {}
        self.integrated_auc = {}
        self.global_bs = {}
        self.integrated_bs = {}

    def cross_validate(self,
                       full_df: pd.DataFrame,
                       l1_ratio: float,
                       penalizers: list,
                       n_splits: int = 5,
                       shuffle: bool = True,
                       seed: Union[int, None] = None,
                       event_type_col: str = 'J',
                       duration_col: str = 'X',
                       pid_col: str = 'pid',
                       twostages_fit_kwargs: dict = {'nb_workers': WORKERS},
                       metrics=['IBS', 'GBS', 'IAUC', 'GAUC']) -> pd.DataFrame:

        """
        This method implements K-fold cross-validation using PenaltyGridSearch and full_df data.

        Args:
            full_df (pd.DataFrame): Data to cross validate.
            l1_ratio (float): regularization ratio for the CoxPHFitter (see lifelines.fitters.coxph_fitter.CoxPHFitter documentation).
            penalizers (list): penalizer options for each event (see lifelines.fitters.coxph_fitter.CoxPHFitter documentation).
            n_splits (int): Number of folds, defaults to 5.
            shuffle (boolean): Shuffle samples before splitting to folds. Defaults to True.
            seed: Pseudo-random seed to KFold instance. Defaults to None.
            event_type_col (str): The event type column name (must be a column in df),
                                  Right-censored sample (i) is indicated by event value 0, df.loc[i, event_type_col] = 0.
            duration_col (str): Last follow up time column name (must be a column in full_df).
            pid_col (str): Sample ID column name (must be a column in full_df).
            twostages_fit_kwargs (dict): keyword arguments to pass to each TwoStagesFitter.
            metrics (str, list): Evaluation metrics. Available metrics:
                                                    'IAUC': Integrated AUC (will be in PenaltyGridSearchCV.integrated_auc),
                                                    'GAUC': Global AUC (will be in PenaltyGridSearchCV.global_auc).
                                                    'IBS': Integrated Brier Score (will be in PenaltyGridSearchCV.integrated_bs),
                                                    'GBS': Global Brier Score (will be in PenaltyGridSearchCV.global_bs).

        Returns:
            gauc_output_df (pd.DataFrame): Global AUC k-fold mean and standard error for all possible combination of the penalizers.
        """

        if isinstance(metrics, str):
            metrics = [metrics]

        self.folds_grids = {}
        self.kfold_cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

        if 'C' in full_df.columns:
            full_df = full_df.drop(['C'], axis=1)
        if 'T' in full_df.columns:
            full_df = full_df.drop(['T'], axis=1)

        for i_fold, (train_index, test_index) in enumerate(self.kfold_cv.split(full_df)):
            print(f'Starting fold {i_fold+1}/{n_splits}')
            start = time()
            self.test_pids[i_fold] = full_df.iloc[test_index][pid_col].values
            train_df, test_df = full_df.iloc[train_index], full_df.iloc[test_index]
            fold_pgs = PenaltyGridSearch()

            fold_pgs.evaluate(train_df=train_df,
                              test_df=test_df,
                              l1_ratio=l1_ratio,
                              penalizers=penalizers,
                              metrics=metrics,
                              seed=seed,
                              event_type_col=event_type_col,
                              duration_col=duration_col,
                              pid_col=pid_col,
                              twostages_fit_kwargs=twostages_fit_kwargs)

            self.folds_grids[i_fold] = deepcopy(fold_pgs)

            for metric in metrics:
                if metric == 'GAUC':
                    self.global_auc[i_fold] = fold_pgs.convert_results_dict_to_df(fold_pgs.global_auc)
                elif metric == 'IAUC':
                    self.integrated_auc[i_fold] = fold_pgs.convert_results_dict_to_df(fold_pgs.integrated_auc)
                elif metric == 'GBS':
                    self.global_bs[i_fold] = fold_pgs.convert_results_dict_to_df(fold_pgs.global_bs)
                elif metric == 'IBS':
                    self.integrated_bs[i_fold] = fold_pgs.convert_results_dict_to_df(fold_pgs.integrated_bs)

            end = time()
            print(f'Finished fold {i_fold+1}/{n_splits}, {int(end-start)} seconds')

        if 'GAUC' in metrics:
            res = [v for k, v in self.global_auc.items()]
            gauc_output_df = pd.concat([pd.concat(res, axis=1).mean(axis=1),
                                        pd.concat(res, axis=1).std(axis=1)],
                                       keys=['Mean', 'SE'], axis=1)
        else:
            gauc_output_df = pd.DataFrame()
        return gauc_output_df