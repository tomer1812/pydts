
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def partial_hazard(model, X):
    """

    Args:
        model:
        X:

    Returns:

    """
    coef = model.params_.values
    inner_exponent = np.dot(X, coef)
    return np.exp(inner_exponent)


def hazard_func(models: dict, X: pd.DataFrame):
    """

    Args:
        models:
        X:

    Returns:

    """
    hazards = {}
    for risk, model in models.items():
        partial = partial_hazard(model, X)
        baseline_hazard = model.baseline_hazard_['baseline hazard']
        hazard = np.outer(partial, baseline_hazard.values)
        hazards.update({risk: hazard})
    return hazards


def cumulative_hazard(model, rel_times, X):
    """

    Args:
        model:
        rel_times:
        X:

    Returns:

    """
    cum_haz = model.baseline_cumulative_hazard_['baseline cumulative hazard'].values
    cum_haz_step = stepfun(rel_times, cum_haz)
    cum_haz = cum_haz_step(rel_times)
    partial = partial_hazard(model, X)
    return np.outer(partial, cum_haz)


def survival_func(models, rel_times, X):
    """

    Args:
        models:
        rel_times:
        X:

    Returns:

    """
    exponent = np.zeros_like(rel_times)
    for risk, model in models.items():
        cum_haz = cumulative_hazard(model, rel_times.values, X)
        exponent = exponent - cum_haz
    surv_func = np.exp(exponent)
    return surv_func


def stepfun(t, h):
    """

    Args:
        t:
        h:

    Returns:

    """
    zeros = np.zeros(1)
    t = np.concatenate((zeros, t))
    h = np.concatenate((zeros, h))
    return interp1d(t, h, kind='previous', fill_value=np.nan, bounds_error=False)