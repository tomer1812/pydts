import numpy as np
from scipy.special import logit
import pandas as pd


class BaseDiscreteModel:
    """
    This class specifies the discrete model structure
    """

    def fit(self):
        raise NotImplemented

    def predict(self):
        raise NotImplemented

    def print_summary(self):
        raise NotImplemented

    def _hazard(self, params, t, event_type):
        raise NotImplemented

    def _cumulative_hazard(self, params, t):
        raise NotImplemented

    def _h_func(self, a):
        raise NotImplemented

    def _log_likelihood(self):
        raise NotImplemented


class DiscreteModel(BaseDiscreteModel):

    alpha_df: pd.DataFrame
    beta_df: pd.DataFrame  # columns

    def _hazard(self, params, t, event_type):
        self._h_func()

    def _h_func(self, a):
        return logit(a)



# class BaseH1DiscreteModel(BaseDiscreteModel):
#     r"""
#     This class implements fitting discrete models with cumulative hazard of the form:
#
#     .. math::  H_{1}(t) = \sum_{i}^{t} h(t_{i})
#
#     """
#
#     def _hazard(self, params, t):
#         return self._cumulative_hazard(params, t) - self._cumulative_hazard(params, t-1)
#
#     def _cumulative_hazard(self, params, t):
#         return np.sum(self._h_func(self.event_times[:t]))
#
#
# class BaseH2DiscreteModel(BaseDiscreteModel):
#     r"""
#     This class implements fitting discrete models with cumulative hazard of the form:
#
#     .. math::  H_{2}(t) = 1 - exp \sum_{i}^{t} h(t_{i})
#
#     """
#
#     def _hazard(self, params, t):
#         return 1 - np.exp(self._cumulative_hazard(params, t) - self._cumulative_hazard(params, t + 1))
#
#     def _cumulative_hazard(self, params, t):
#         # -np.log(S(t))
#         return -np.log(self._overall_survival_proba(params, t))
#
#     def _overall_survival_proba(self, params, t):
#         return 1