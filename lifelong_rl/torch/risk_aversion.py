import numpy as np
import torch

import lifelong_rl.torch.pytorch_util as ptu

"""
Risk aversion interface. Risk aversion (or seeking) can be represented as w^T x where x is
a sorted vector of the values from the distribution drawn iid and w is a mask. This
module handles the generation of these masks via defining the inverse beta function.
"""


def get_mask(mask_type, n_quantiles, risk_kwargs):
    """
    Return a torch mask corresponding to the input parameters.
    """
    if mask_type in _masks:
        if n_quantiles in _masks[mask_type]:
            return _masks[mask_type][n_quantiles]
    else:
        _masks[mask_type] = dict()

    if mask_type in _inverse_beta_funcs:
        _masks[mask_type][n_quantiles] = create_mask(
            _inverse_beta_funcs[mask_type],
            n_quantiles=n_quantiles,
            risk_kwargs=risk_kwargs,
        )
    else:
        raise NotImplementedError('mask_type not recognized')

    return _masks[mask_type][n_quantiles]


"""
Utility functions
"""


_masks = dict()


def create_mask(inverse_beta_func, n_quantiles, risk_kwargs):
    """
    x in [0, 1] represents the CDF of the input.
    beta(x) represents the cumulative weight assigned to the lower x% of
        values, e.g. it is analogous to the CDF. This is typically easier
        to represent via the inverse of the beta function, so we take the
        inverse of the inverse beta function to get the original function.
    The reweighted function becomes:
        R(f, beta) = sum_i f(i/n) * (beta((i+1)/(n+1)) - beta(i/(n+1))
    """

    tau = np.linspace(0, 1, n_quantiles + 1)
    betas = np.zeros(n_quantiles + 1)
    mask = np.zeros(n_quantiles)

    # TODO: there are some issues with mask and risk_kwarg caching

    for i in range(n_quantiles + 1):
        betas[i] = inverse_beta_func(tau[i], risk_kwargs)
    for i in range(n_quantiles):
        mask[i] = betas[i+1] - betas[i]

    return ptu.from_numpy(mask)


def get_inverse(func, x, n_bins=1024, risk_kwargs=None):
    # assumes domain/range is (0, 1), and function is monotonically increasing

    # assume we don't need things finer than 1024 for now, just
    # going to use a slow linear search
    # TODO: this function can be rewritten much better

    risk_kwargs = risk_kwargs if risk_kwargs is not None else dict()

    for i in range(n_bins):
        new_val = func(i / n_bins, risk_kwargs)
        if x <= new_val:
            return i / n_bins
    return 1.


"""
Types of risk aversion
"""


def neutral_func(tau, risk_kwargs):
    # Neutral risk preference / expected value / identity function
    return tau


def cvar_func(tau, risk_kwargs):
    # Conditional Value at Risk (only consider bottom alpha% of outcomes)
    alpha = risk_kwargs['alpha']
    if tau < alpha:
        return tau / alpha
    else:
        return 1.


def _cpw_weight(tau, risk_kwargs):
    eta = risk_kwargs['eta']
    return (tau ** eta) / (((tau ** eta) + (1 - tau) ** eta) ** (1 / eta))


def cpw_func(tau, risk_kwargs):
    # Cumulative Probability Weighting (from prospect theory)
    return get_inverse(_cpw_weight, tau, risk_kwargs=risk_kwargs)


_inverse_beta_funcs = dict(
    neutral=neutral_func,
    cvar=cvar_func,
    cpw=cpw_func,
)
