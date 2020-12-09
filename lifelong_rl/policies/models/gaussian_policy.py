import numpy as np
import torch
from torch import nn as nn

from lifelong_rl.policies.base.base import ExplorationPolicy
from lifelong_rl.torch.pytorch_util import eval_np
from lifelong_rl.torch.distributions import TanhNormal
from lifelong_rl.models.networks import Mlp
import lifelong_rl.torch.pytorch_util as ptu


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class TanhGaussianPolicy(Mlp, ExplorationPolicy):

    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    action, mean, log_std, _ = policy(obs)
    action, mean, log_std, _ = policy(obs, deterministic=True)
    action, mean, log_std, log_prob = policy(obs, return_log_prob=True)
    ```

    Here, mean and log_std are the mean and log_std of the Gaussian that is
    sampled from.

    If deterministic is True, action = tanh(mean).
    If return_log_prob is False (default), log_prob = None
        This is done because computing the log_prob can be a bit expensive.
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            restrict_obs_dim=0,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        self.restrict_obs_dim = restrict_obs_dim

        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            init_logstd = ptu.ones(1, action_dim) * np.log(std)
            self.log_std = torch.nn.Parameter(init_logstd, requires_grad=True)

            # for NPG
            model_parameters = filter(lambda p: p.requires_grad, self.parameters())
            self.trainable_params = list(model_parameters) + [self.log_std]
            self.param_shapes = [p.cpu().data.numpy().shape for p in self.trainable_params]
            self.param_sizes = [p.cpu().data.numpy().size for p in self.trainable_params]

    def get_action(self, obs_np, deterministic=False):
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def forward(
            self,
            obs,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        if len(obs.shape) == 1:
            obs = obs[self.restrict_obs_dim:]
        else:
            obs = obs[:,self.restrict_obs_dim:]

        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            log_std = self.log_std * ptu.ones(*mean.shape)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = log_std.exp()

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

    def get_log_probs(self, obs, actions):
        _, mean, log_std, *_ = self.forward(obs, deterministic=True)
        tanh_normal = TanhNormal(mean, log_std.exp())
        return tanh_normal.log_prob(actions).sum(dim=-1, keepdim=True)

    def get_param_values(self):
        params = np.concatenate([p.contiguous().view(-1).cpu().data.numpy() for p in self.trainable_params])
        return params.copy()

    def set_param_values(self, new_params):
        current_idx = 0
        for idx, param in enumerate(self.trainable_params):
            vals = new_params[current_idx:current_idx + self.param_sizes[idx]]
            vals = vals.reshape(self.param_shapes[idx])
            param.data = ptu.from_numpy(vals).float()
            current_idx += self.param_sizes[idx]
        self.trainable_params[-1].data = torch.clamp(self.trainable_params[-1], LOG_SIG_MIN)
