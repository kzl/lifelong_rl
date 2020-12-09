import numpy as np
import torch
import torch.optim as optim

from collections import OrderedDict
import copy

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.envs.env_utils import get_dim
from lifelong_rl.util.eval_util import create_stats_ordered_dict
from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
import lifelong_rl.samplers.utils.path_functions as path_functions
import lifelong_rl.util.pythonplusplus as ppp


class PGTrainer(TorchTrainer):

    """
    Encapsulating base trainer for policy gradient methods trained from trajectories.
    By itself, trains using vanilla policy gradient with some tricks (GAE, early stopping, etc).
    """

    def __init__(
            self,
            env,                        # Associated environment
            policy,                     # Associated policy
            value_func,                 # Associated value function V(s)
            discount=0.99,              # Discount factor
            gae_lambda=0.95,            # Lambda to use for GAE for value estimation
            policy_lr=1e-3,             # Learning rate for policy
            value_lr=1e-3,              # Learning rate for value function
            target_kl=0.01,             # Can do early termination if KL is reached
            entropy_coeff=0.,           # Coefficient of entropy bonus
            num_epochs=10,              # Number of epochs for training per train call
            num_policy_epochs=None,     # Number of epochs for policy (can be < num_epochs)
            policy_batch_size=1024,     # Batch size for policy training
            value_batch_size=1024,      # Batch size for value function training
            normalize_advantages=True,  # Optionally, can normalize advantages
            input_normalization=True,   # Whether or not to normalize the inputs to policy & value
            max_grad_norm=10,           # Gradient norm clipping
            action_dim=None,
    ):
        super().__init__()

        self.env = env
        self.obs_dim = get_dim(self.env.observation_space)
        self.action_dim = self.env.action_space.shape[0] if action_dim is None else action_dim
        self.policy = policy
        self.value_func = value_func
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.target_kl = target_kl
        self.entropy_coeff = entropy_coeff
        self.num_epochs = num_epochs
        self.num_policy_epochs = num_policy_epochs if num_policy_epochs is not None else num_epochs
        self.policy_batch_size = policy_batch_size
        self.value_batch_size = value_batch_size
        self.normalize_advantages = normalize_advantages
        self.input_normalization = input_normalization
        self.max_grad_norm = max_grad_norm

        if policy_lr is not None:
            self.policy_optim = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optim = optim.Adam(self.value_func.parameters(), lr=value_lr)

        self._reward_std = 1

        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_from_paths(self, paths):

        """
        Path preprocessing; have to copy so we don't modify when paths are used elsewhere
        """

        paths = copy.deepcopy(paths)
        for path in paths:
            # Other places like to have an extra dimension so that all arrays are 2D
            path['terminals'] = np.squeeze(path['terminals'], axis=-1)
            path['rewards'] = np.squeeze(path['rewards'], axis=-1)

            # Reward normalization; divide by std of reward in replay buffer
            path['rewards'] = np.clip(path['rewards'] / (self._reward_std + 1e-3), -10, 10)

        obs, actions = [], []
        for path in paths:
            obs.append(path['observations'])
            actions.append(path['actions'])
        obs = np.concatenate(obs, axis=0)
        actions = np.concatenate(actions, axis=0)

        obs_tensor, act_tensor = ptu.from_numpy(obs), ptu.from_numpy(actions)

        """
        Policy training loop
        """

        old_policy = copy.deepcopy(self.policy)
        with torch.no_grad():
            log_probs_old = old_policy.get_log_probs(obs_tensor, act_tensor).squeeze(dim=-1)

        rem_value_epochs = self.num_epochs
        for epoch in range(self.num_policy_epochs):

            """
            Recompute advantages at the beginning of each epoch. This allows for advantages
                to utilize the latest value function.
            Note: while this is not present in most implementations, it is recommended
                  by Andrychowicz et al. 2020.
            """

            path_functions.calculate_baselines(paths, self.value_func)
            path_functions.calculate_returns(paths, self.discount)
            path_functions.calculate_advantages(
                paths, self.discount, self.gae_lambda, self.normalize_advantages,
            )

            advantages, returns, baselines = [], [], []
            for path in paths:
                advantages = np.append(advantages, path['advantages'])
                returns = np.append(returns, path['returns'])

            if epoch == 0 and self._need_to_update_eval_statistics:
                with torch.no_grad():
                    values = torch.squeeze(self.value_func(obs_tensor), dim=-1)
                    values_np = ptu.get_numpy(values)
                first_val_loss = ((returns - values_np) ** 2).mean()

            old_params = self.policy.get_param_values()

            num_policy_steps = len(advantages) // self.policy_batch_size
            for _ in range(num_policy_steps):
                if num_policy_steps == 1:
                    batch = dict(
                        observations=obs,
                        actions=actions,
                        advantages=advantages,
                    )
                else:
                    batch = ppp.sample_batch(
                        self.policy_batch_size,
                        observations=obs,
                        actions=actions,
                        advantages=advantages,
                    )
                policy_loss, kl = self.train_policy(batch, old_policy)

            with torch.no_grad():
                log_probs = self.policy.get_log_probs(obs_tensor, act_tensor).squeeze(dim=-1)
            kl = (log_probs_old - log_probs).mean()

            if (self.target_kl is not None and kl > 1.5 * self.target_kl) or (kl != kl):
                if epoch > 0 or kl != kl:  # nan check
                    self.policy.set_param_values(old_params)
                break

            num_value_steps = len(advantages) // self.value_batch_size
            for i in range(num_value_steps):
                batch = ppp.sample_batch(
                    self.value_batch_size,
                    observations=obs,
                    targets=returns,
                )
                value_loss = self.train_value(batch)
            rem_value_epochs -= 1

        # Ensure the value function is always updated for the maximum number
        # of epochs, regardless of if the policy wants to terminate early.
        for _ in range(rem_value_epochs):
            num_value_steps = len(advantages) // self.value_batch_size
            for i in range(num_value_steps):
                batch = ppp.sample_batch(
                    self.value_batch_size,
                    observations=obs,
                    targets=returns,
                )
                value_loss = self.train_value(batch)

        if self._need_to_update_eval_statistics:
            with torch.no_grad():
                _, _, _, log_pi, *_ = self.policy(obs_tensor, return_log_prob=True)
                values = torch.squeeze(self.value_func(obs_tensor), dim=-1)
                values_np = ptu.get_numpy(values)

            errors = returns - values_np
            explained_variance = 1 - (np.var(errors) / np.var(returns))
            value_loss = errors ** 2

            self.eval_statistics['Num Epochs'] = epoch + 1

            self.eval_statistics['Policy Loss'] = ptu.get_numpy(policy_loss).mean()
            self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl).mean()
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Advantages',
                advantages,
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Returns',
                returns,
            ))

            self.eval_statistics['Value Loss'] = value_loss.mean()
            self.eval_statistics['First Value Loss'] = first_val_loss
            self.eval_statistics['Value Explained Variance'] = explained_variance
            self.eval_statistics.update(create_stats_ordered_dict(
                'Values',
                ptu.get_numpy(values),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Value Squared Errors',
                value_loss,
            ))

    def fit_input_stats(self, replay_buffer):
        if self.input_normalization:
            transitions = replay_buffer.get_transitions()
            obs = transitions[:,:self.obs_dim]
            self.policy.fit_input_stats(obs)
            self.value_func.fit_input_stats(obs)
            self._reward_std = transitions[:,-(self.obs_dim+2)].std()
            if self._reward_std < 0.01:
                self._reward_std = transitions[:,-(self.obs_dim+2)].max()

    def policy_objective(self, obs, actions, advantages, old_policy):
        log_probs = torch.squeeze(self.policy.get_log_probs(obs, actions), dim=-1)
        log_probs_old = torch.squeeze(old_policy.get_log_probs(obs, actions), dim=-1)
        objective = (log_probs * advantages).mean()
        kl = (log_probs_old - log_probs).mean()
        return objective, kl

    def train_policy(self, batch, old_policy):
        obs = ptu.from_numpy(batch['observations'])
        actions = ptu.from_numpy(batch['actions'])
        advantages = ptu.from_numpy(batch['advantages'])

        objective, kl = self.policy_objective(obs, actions, advantages, old_policy)
        policy_loss = -objective

        self.policy_optim.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optim.step()

        return policy_loss, kl

    def train_value(self, batch):
        obs = ptu.from_numpy(batch['observations'])
        targets = ptu.from_numpy(batch['targets'])

        value_preds = torch.squeeze(self.value_func(obs), dim=-1)
        value_loss = 0.5 * ((value_preds - targets) ** 2).mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        return value_loss

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.value_func,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            value_func=self.value_func,
        )
