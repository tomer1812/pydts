import pandas as pd
import numpy as np
from pydts.fitters import TwoStagesFitter, DataExpansionFitter
import warnings
from copy import deepcopy
from sklearn.model_selection import KFold
pd.set_option("display.max_rows", 500)
warnings.filterwarnings('ignore')
slicer = pd.IndexSlice
from typing import Optional, List, Union
import psutil
from .evaluation import prediction_error

WORKERS = psutil.cpu_count(logical=False)


class TwoStagesCV(object):
    """
    This class implements K-fold cross-validation using TwoStagesFitters
    """

    def __init__(self):
        self.models = {}
        self.test_idx = {}
        self.results = pd.DataFrame()

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
                       metrics='PE'):

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
            metrics (str, list): Evaluation metrics. Currently only prediction error ('PE') is available

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
            self.test_idx[i_fold] = test_index
            train_df, test_df = full_df.loc[train_index], full_df.loc[test_index]
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
            for metric in metrics:
                if metric == 'PE':
                    pred_df = self.models[i_fold].predict_cumulative_incident_function(test_df)
                    tmp_res = prediction_error(pred_df, event_type_col=event_type_col,
                                                        duration_col=duration_col)
                    tmp_res = pd.concat([tmp_res], keys=[i_fold], names=['fold'])
                    tmp_res = pd.concat([tmp_res], keys=[metric], names=['metric'])
                    self.results = pd.concat([self.results, tmp_res], axis=0)

        return self.results