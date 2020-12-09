import numpy as np
import torch

import lifelong_rl.torch.pytorch_util as ptu


"""
Various path utilities (mostly used by policy gradient experiment_configs)
"""


def calculate_baselines(paths, value_func):
    for path in paths:
        obs = ptu.from_numpy(np.concatenate(
            [path['observations'], path['next_observations'][-1:]], axis=0
        ))
        values = torch.squeeze(value_func(obs), dim=-1)
        path['baselines'] = ptu.get_numpy(values)
        if path['terminals'][-1]:
            path['baselines'][-1] = 0


def calculate_returns(paths, discount):
    for path in paths:
        rewards, dones = path['rewards'], path['terminals']
        if 'baselines' in path:
            terminal_value = path['baselines'][-1]
        else:
            terminal_value = 0
        rewards = np.append(rewards, terminal_value)
        path['returns'] = discount_cumsum(rewards, dones, discount)[:-1]
        assert len(path['returns']) == len(dones)


def calculate_advantages(paths, discount, gae_lambda=None, normalize=False):
    total_advs = []
    for path in paths:
        returns = path['returns']
        if 'baselines' not in path:
            advantages = returns
        elif gae_lambda is None:
            advantages = returns - path['baselines'][:-1]
        else:
            rewards, baselines, dones = path['rewards'], path['baselines'], path['terminals']
            assert len(baselines) == len(rewards)+1
            td_deltas = rewards + discount * baselines[1:] - baselines[:-1]
            assert td_deltas.shape == rewards.shape
            advantages = discount_cumsum(td_deltas, dones, gae_lambda * discount)
            assert advantages.shape == rewards.shape
        path['advantages'] = advantages
        if normalize:
            total_advs = np.append(total_advs, advantages)
    if normalize:
        mean, std = total_advs.mean(), total_advs.std()
        for path in paths:
            path['advantages'] = (path['advantages'] - mean) / (std + 1e-6)


def discount_cumsum(x, dones, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1] * (1-dones[t])
    return discount_cumsum
