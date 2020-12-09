import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.models.networks import ParallelizedEnsemble


class ProbabilisticEnsemble(ParallelizedEnsemble):

    """
    Probabilistic ensemble (Chua et al. 2018).
    Implementation is parallelized such that every model uses one forward call.
    Each member predicts the mean and variance of the next state.
    Sampling is done either uniformly or via trajectory sampling.
    """

    def __init__(
            self,
            ensemble_size,        # Number of members in ensemble
            obs_dim,              # Observation dim of environment
            action_dim,           # Action dim of environment
            hidden_sizes,         # Hidden sizes for each model
            spectral_norm=False,  # Apply spectral norm to every hidden layer
            **kwargs
    ):
        super().__init__(
            ensemble_size=ensemble_size,
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=2*(obs_dim + 2),  # We predict (reward, done, next_state - state)
            hidden_activation=torch.tanh,
            spectral_norm=spectral_norm,
            **kwargs
        )

        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.output_size = obs_dim + 2

        # Note: we do not learn the logstd here, but some implementations do
        self.max_logstd = nn.Parameter(
            ptu.ones(obs_dim + 2), requires_grad=False)
        self.min_logstd = nn.Parameter(
            -ptu.ones(obs_dim + 2) * 5, requires_grad=False)

    def forward(self, input, deterministic=False, return_dist=False):
        output = super().forward(input)
        mean, logstd = torch.chunk(output, 2, dim=-1)

        # Variance clamping to prevent poor numerical predictions
        logstd = self.max_logstd - F.softplus(self.max_logstd - logstd)
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)

        if deterministic:
            return mean, logstd if return_dist else mean

        std = torch.exp(logstd)
        eps = ptu.randn(std.shape)
        samples = mean + std * eps

        if return_dist:
            return samples, mean, logstd
        else:
            return samples

    def get_loss(self, x, y, split_by_model=False, return_l2_error=False):
        # Note: we assume y here already accounts for the delta of the next state

        mean, logstd = self.forward(x, deterministic=True, return_dist=True)
        if len(y.shape) < 3:
            y = y.unsqueeze(0).repeat(self.ensemble_size, 1, 1)

        # Maximize log-probability of transitions
        inv_var = torch.exp(-2 * logstd)
        sq_l2_error = (mean - y)**2
        if return_l2_error:
            l2_error = torch.sqrt(sq_l2_error).mean(dim=-1).mean(dim=-1)

        loss = (sq_l2_error * inv_var + 2 * logstd).sum(dim=-1).mean(dim=-1)

        if split_by_model:
            losses = [loss[i] for i in range(self.ensemble_size)]
            if return_l2_error:
                l2_errors = [l2_error[i] for i in range(self.ensemble_size)]
                return losses, l2_errors
            else:
                return losses
        else:
            if return_l2_error:
                return loss.mean(), l2_error.mean()
            else:
                return loss.mean()

    def sample_with_disagreement(self, input, return_dist=False, disagreement_type='mean'):
        preds, mean, logstd = self.forward(input, deterministic=False, return_dist=True)

        # Standard uniformly from the ensemble
        inds = torch.randint(0, preds.shape[0], input.shape[:-1])

        # Ensure we don't use the same member to estimate disagreement
        inds_b = torch.randint(0, mean.shape[0], input.shape[:-1])
        inds_b[inds == inds_b] = torch.fmod(inds_b[inds == inds_b] + 1, mean.shape[0])

        # Repeat for multiplication
        inds = inds.unsqueeze(dim=-1).to(device=ptu.device)
        inds = inds.repeat(1, preds.shape[2])
        inds_b = inds_b.unsqueeze(dim=-1).to(device=ptu.device)
        inds_b = inds_b.repeat(1, preds.shape[2])

        # Uniformly sample from ensemble
        samples = (inds == 0).float() * preds[0]
        for i in range(1, preds.shape[0]):
            samples += (inds == i).float() * preds[i]

        if disagreement_type == 'mean':
            # Disagreement = mean squared difference in mean predictions (Kidambi et al. 2020)
            means_a = (inds == 0).float() * mean[0]
            means_b = (inds_b == 0).float() * mean[0]
            for i in range(1, preds.shape[0]):
                means_a += (inds == i).float() * mean[i]
                means_b += (inds_b == i).float() * mean[i]

            disagreements = torch.mean((means_a - means_b) ** 2, dim=-1, keepdim=True)

        elif disagreement_type == 'var':
            # Disagreement = max Frobenius norm of covariance matrix (Yu et al. 2020)
            vars = (2 * logstd).exp()
            frobenius = torch.sqrt(vars.sum(dim=-1))
            disagreements, *_ = frobenius.max(dim=0)
            disagreements = disagreements.reshape(-1, 1)

        else:
            raise NotImplementedError

        if return_dist:
            return samples, disagreements, mean, logstd
        else:
            return samples, disagreements
