import numpy as np
import torch
from torch.distributions import Distribution as TorchDistribution
from torch.distributions import Normal as TorchNormal
from torch.distributions import Independent as TorchIndependent

from collections import OrderedDict

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.util.eval_util import create_stats_ordered_dict


class Distribution(TorchDistribution):

    def sample_and_logprob(self):
        s = self.sample()
        log_p = self.log_prob(s)
        return s, log_p

    def rsample_and_logprob(self):
        s = self.rsample()
        log_p = self.log_prob(s)
        return s, log_p

    def mle_estimate(self):
        return self.mean

    def get_diagnostics(self):
        return {}


class Independent(Distribution, TorchIndependent):

    def get_diagnostics(self):
        return self.base_dist.get_diagnostics()


class TorchDistributionWrapper(Distribution):

    def __init__(self, distribution: TorchDistribution):
        self.distribution = distribution

    @property
    def batch_shape(self):
        return self.distribution.batch_shape

    @property
    def event_shape(self):
        return self.distribution.event_shape

    @property
    def arg_constraints(self):
        return self.distribution.arg_constraints

    @property
    def support(self):
        return self.distribution.support

    @property
    def mean(self):
        return self.distribution.mean

    @property
    def variance(self):
        return self.distribution.variance

    @property
    def stddev(self):
        return self.distribution.stddev

    def sample(self, sample_size=torch.Size()):
        return self.distribution.sample(sample_shape=sample_size)

    def rsample(self, sample_size=torch.Size()):
        return self.distribution.rsample(sample_shape=sample_size)

    def log_prob(self, value):
        return self.distribution.log_prob(value)

    def cdf(self, value):
        return self.distribution.cdf(value)

    def icdf(self, value):
        return self.distribution.icdf(value)

    def enumerate_support(self, expand=True):
        return self.distribution.enumerate_support(expand=expand)

    def entropy(self):
        return self.distribution.entropy()

    def perplexity(self):
        return self.distribution.perplexity()

    def __repr__(self):
        return 'Wrapped ' + self.distribution.__repr__()


class MultivariateDiagonalNormal(TorchDistributionWrapper):

    from torch.distributions import constraints
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}

    def __init__(self, loc, scale_diag, reinterpreted_batch_ndims=1):
        dist = Independent(TorchNormal(loc, scale_diag),
                           reinterpreted_batch_ndims=reinterpreted_batch_ndims)
        super().__init__(dist)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'mean',
            ptu.get_numpy(self.mean),
            # exclude_max_min=True,
        ))
        stats.update(create_stats_ordered_dict(
            'std',
            ptu.get_numpy(self.distribution.stddev),
        ))
        return stats

    def __repr__(self):
        return self.distribution.base_dist.__repr__()


class TanhNormal(Distribution):

    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.

    """
    def __init__(self, normal_mean, normal_std, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = torch.distributions.Normal(normal_mean, normal_std)
        self.epsilon = epsilon

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def _log_prob_from_pre_tanh(self, pre_tanh_value):
        """
        Adapted from
        https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73
        This formula is mathematically equivalent to log(1 - tanh(x)^2).
        Derivation:
        log(1 - tanh(x)^2)
         = log(sech(x)^2)
         = 2 * log(sech(x))
         = 2 * log(2e^-x / (e^-2x + 1))
         = 2 * (log(2) - x - log(e^-2x + 1))
         = 2 * (log(2) - x - softplus(-2x))
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        log_prob = self.normal.log_prob(pre_tanh_value)
        correction = - 2. * (
            ptu.from_numpy(np.log([2.]))
            - pre_tanh_value
            - torch.nn.functional.softplus(-2. * pre_tanh_value)
        )
        return log_prob + correction

    def log_prob(self, value, pre_tanh_value=None):
        if pre_tanh_value is None:
            # errors or instability at values near 1
            value = torch.clamp(value, -0.999999, 0.999999)
            pre_tanh_value = torch.log(1+value) / 2 - torch.log(1-value) / 2
        return self._log_prob_from_pre_tanh(pre_tanh_value)

    def rsample_with_pretanh(self):
        z = (
                self.normal_mean +
                self.normal_std *
                MultivariateDiagonalNormal(
                    ptu.zeros(self.normal_mean.size()),
                    ptu.ones(self.normal_std.size())
                ).sample()
        )
        return torch.tanh(z), z

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        if return_pretanh_value:
            return value.detach(), pre_tanh_value
        else:
            return value.detach()

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        value, pre_tanh_value = self.rsample_with_pretanh()
        if return_pretanh_value:
            return value, pre_tanh_value
        else:
            return value

    def sample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        value, pre_tanh_value = value.detach(), pre_tanh_value.detach()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    def rsample_and_logprob(self):
        value, pre_tanh_value = self.rsample_with_pretanh()
        log_p = self.log_prob(value, pre_tanh_value)
        return value, log_p

    @property
    def mean(self):
        return torch.tanh(self.normal_mean)

    def get_diagnostics(self):
        stats = OrderedDict()
        stats.update(create_stats_ordered_dict(
            'mean',
            ptu.get_numpy(self.mean),
        ))
        stats.update(create_stats_ordered_dict(
            'normal/std',
            ptu.get_numpy(self.normal_std)
        ))
        stats.update(create_stats_ordered_dict(
            'normal/log_std',
            ptu.get_numpy(torch.log(self.normal_std)),
        ))
        return stats
