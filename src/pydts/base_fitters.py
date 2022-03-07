import numpy as np
from scipy.special import logit
import pandas as pd
from typing import Callable, Iterator, List, Optional, Tuple, Union, Any, Iterable
import statsmodels.api as sm


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