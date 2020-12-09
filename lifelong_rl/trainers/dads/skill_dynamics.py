import numpy as np
import torch

import lifelong_rl.torch.pytorch_util as ptu


class SkillDynamics(torch.nn.Module):

    def __init__(
            self,
            observation_size,
            action_size,
            latent_size,
            normalize_observations=True,
            fc_layer_params=(256, 256),
            fix_variance=True,
            activation_func=torch.nn.ReLU,
    ):
        super().__init__()

        self._observation_size = observation_size
        self._action_size = action_size
        self._latent_size = latent_size
        self._normalize_observations = normalize_observations

        self._fc_layer_params = fc_layer_params
        self._fix_variance = fix_variance

        layers = []
        for i in range(len(fc_layer_params)-1):
            if i == 0:
                layers.append(activation_func())
            layers.append(torch.nn.Linear(fc_layer_params[i], fc_layer_params[i+1]))
            layers.append(activation_func())
        self.model = torch.nn.Sequential(*layers)

        in_layers = []
        if self._normalize_observations:
            in_layers.append(torch.nn.BatchNorm1d(observation_size + latent_size))
            self.out_preproc = torch.nn.BatchNorm1d(observation_size, affine=False)
        else:
            print('not normalization observations')
        in_layers.append(torch.nn.Linear(observation_size + latent_size, fc_layer_params[0]))

        self.in_func = torch.nn.Sequential(*in_layers)

        self.out_mean = torch.nn.Linear(fc_layer_params[-1], observation_size)
        if not self._fix_variance:
            self.out_std = torch.nn.Linear(fc_layer_params[-1], observation_size)
            # TODO: implement clipping
            raise NotImplementedError

        self._normalize_output = True

    def forward(self, obs, latents):
        x = torch.cat([obs, latents], dim=-1)
        x = self.in_func(x)
        x = self.model(x)
        if self._fix_variance:
            return self.out_mean(x)
        else:
            return self.out_mean(x), self.out_std(x)

    def _get_distribution(self, obs, latents):
        x = torch.cat([obs, latents], dim=-1)
        x = self.in_func(x)
        x = self.model(x)

        mean = self.out_mean(x)
        if self._fix_variance:
            std = ptu.ones(*mean.shape)
            dist = torch.distributions.independent.Independent(
                torch.distributions.Normal(mean, std), 1
            )
        else:
            raise NotImplementedError

        return dist

    def get_log_prob(self, obs, latents, next_obs):
        if self._normalize_observations:
            next_obs = self.out_preproc(next_obs)
        dist = self._get_distribution(obs, latents)
        return dist.log_prob(next_obs)

    def get_loss(self, obs, latents, next_obs, weights=None):
        log_probs = self.get_log_prob(obs, latents, next_obs)
        if weights is not None:
            log_probs = log_probs * weights
        return -log_probs.mean()
