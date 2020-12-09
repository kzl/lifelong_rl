import gtimer as gt
import numpy as np
import torch

from collections import OrderedDict

from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.trainers.dads.empowerment_functions import calculate_contrastive_empowerment
from lifelong_rl.util.eval_util import create_stats_ordered_dict
import lifelong_rl.util.pythonplusplus as ppp


class DADSTrainer(TorchTrainer):

    """
    Sharma et al. 2019. "Dynamics-Aware Discovery of Skills".
    Learns skills using a contrastive lower bound to empowerment.
    """

    def __init__(
            self,
            control_policy,              # Associated low-level skill policy
            discriminator,               # Associated discriminator q(s' | s, z)
            replay_buffer,               # Associated replay buffer of transitions
            policy_trainer,              # Associated trainer for the control policy
            replay_size,                 # How many transitions to store for learning
            num_prior_samples=512,       # Number of samples from prior for estimating reward
            num_discrim_updates=32,      # Number of discriminator updates per train call
            num_policy_updates=64,       # Number of policy updates per train call
            discrim_learning_rate=1e-3,  # Learning arte for discriminator
            policy_batch_size=128,       # Batch size for training networks
            reward_bounds=(-10, 10),     # Clip intrinsic rewards for stability
            empowerment_horizon=1,       # Length of horizon for empowerment
            reward_scale=5.,             # Typically beneficial to multiply rewards
            restrict_input_size=0,       # If > 0, restrict input to discriminator
            relabel_rewards=True,        # Whether or not to relabel the rewards of buffer
            train_every=1,               # How often to train when train is called
            reward_mode=None,            # Which reward function to use (default: contrastive)
    ):
        super().__init__()

        self.control_policy = control_policy
        self.discriminator = discriminator
        self.replay_buffer = replay_buffer
        self.policy_trainer = policy_trainer

        self.obs_dim = replay_buffer.obs_dim()
        self.action_dim = replay_buffer.action_dim()
        self.latent_dim = self.control_policy.input_size - self.obs_dim

        self.num_prior_samples = num_prior_samples
        self.num_discrim_updates = num_discrim_updates
        self.num_policy_updates = num_policy_updates
        self.policy_batch_size = policy_batch_size
        self.reward_bounds = reward_bounds
        self.empowerment_horizon = empowerment_horizon
        self.reward_scale = reward_scale
        self.restrict_input_size = restrict_input_size
        self.relabel_rewards = relabel_rewards
        self.reward_mode = reward_mode

        self.discrim_optim = torch.optim.Adam(
            discriminator.parameters(), lr=discrim_learning_rate, betas=(0, 0.9),
        )

        self._obs = np.zeros((replay_size, self.obs_dim))
        self._next_obs = np.zeros((replay_size, self.obs_dim))       # obs + empowerment_horizon
        self._true_next_obs = np.zeros((replay_size, self.obs_dim))  # obs + 1 (normal next_obs)
        self._latents = np.zeros((replay_size, self.latent_dim))
        self._actions = np.zeros((replay_size, self.action_dim))
        self._rewards = np.zeros((replay_size, 1))
        self._logprobs = np.zeros((replay_size, 1))
        self._ptr = 0
        self.replay_size = replay_size
        self._cur_replay_size = 0

        self.obs_mean, self.obs_std = None, None

        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self._epoch_size = None
        self.eval_statistics = OrderedDict()

        self.train_every = train_every
        self._train_calls = 0

    def add_sample(self, obs, next_obs, true_next_obs, action, latent, logprob=None, **kwargs):
        self._obs[self._ptr] = obs
        self._next_obs[self._ptr] = next_obs
        self._true_next_obs[self._ptr] = true_next_obs
        self._actions[self._ptr] = action
        self._latents[self._ptr] = latent

        if logprob is not None:
            self._logprobs[self._ptr] = logprob

        self._ptr = (self._ptr + 1) % self.replay_size
        self._cur_replay_size = min(self._cur_replay_size+1, self.replay_size)

    def calculate_intrinsic_rewards(self, states, next_states, latents, *args, **kwargs):
        if self.restrict_input_size > 0:
            states = states[:,:self.restrict_input_size]
            next_states = next_states[:,:self.restrict_input_size]
        if self.reward_mode is None:
            reward_func = calculate_contrastive_empowerment
        else:
            raise NotImplementedError('reward_mode not recognized')
        rewards, (logp, logp_altz, denom), reward_diagnostics = reward_func(
            self.discriminator,
            states,
            next_states,
            latents,
            num_prior_samples=self.num_prior_samples,
            distribution_type='uniform',
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            return_diagnostics=True,
        )
        rewards[rewards != rewards] = -10  # check for NaN
        rewards = np.clip(rewards, self.reward_bounds[0], self.reward_bounds[1])
        return rewards, (logp, logp_altz, denom), reward_diagnostics

    def reward_postprocessing(self, rewards, *args, **kwargs):
        # Some scaling of the rewards can help; it is very finicky though
        rewards *= self.reward_scale
        rewards = np.clip(rewards, *self.reward_bounds)  # stabilizes training
        return rewards, dict()

    def train_from_paths(self, paths, train_discrim=True, train_policy=True):

        """
        Reading new paths: append latent to state
        Note that is equivalent to on-policy when latent buffer size = sum of paths length
        """

        epoch_obs, epoch_next_obs, epoch_latents = [], [], []

        for path in paths:
            obs = path['observations']
            next_obs = path['next_observations']
            actions = path['actions']
            latents = path.get('latents', None)
            path_len = len(obs) - self.empowerment_horizon + 1

            obs_latents = np.concatenate([obs, latents], axis=-1)
            log_probs = self.control_policy.get_log_probs(
                ptu.from_numpy(obs_latents),
                ptu.from_numpy(actions),
            )
            log_probs = ptu.get_numpy(log_probs)

            for t in range(path_len):
                self.add_sample(
                    obs[t],
                    next_obs[t+self.empowerment_horizon-1],
                    next_obs[t],
                    actions[t],
                    latents[t],
                    logprob=log_probs[t],
                )

                epoch_obs.append(obs[t:t+1])
                epoch_next_obs.append(next_obs[t+self.empowerment_horizon-1:t+self.empowerment_horizon])
                epoch_latents.append(np.expand_dims(latents[t], axis=0))

        epoch_obs = np.concatenate(epoch_obs, axis=0)
        epoch_next_obs = np.concatenate(epoch_next_obs, axis=0)
        epoch_latents = np.concatenate(epoch_latents, axis=0)

        self._epoch_size = len(epoch_obs)

        gt.stamp('policy training', unique=False)

        """
        The rest is shared, train from buffer
        """

        if train_discrim:
            self.train_discriminator(epoch_obs, epoch_next_obs, epoch_latents)
        if train_policy:
            self.train_from_buffer()

    def train_discriminator(self, obs, next_obs, latents):

        obs_deltas = next_obs - obs

        self.discriminator.train()
        start_discrim_loss = None

        for i in range(self.num_discrim_updates):
            batch = ppp.sample_batch(
                self.policy_batch_size,
                obs=obs,
                latents=latents,
                obs_deltas=obs_deltas,
            )
            batch = ptu.np_to_pytorch_batch(batch)

            if self.restrict_input_size > 0:
                batch['obs'] = batch['obs'][:, :self.restrict_input_size]
                batch['obs_deltas'] = batch['obs_deltas'][:, :self.restrict_input_size]

            # we embedded the latent in the observation, so (s, z) -> (delta s')
            discrim_loss = self.discriminator.get_loss(
                batch['obs'],
                batch['latents'],
                batch['obs_deltas'],
            )

            if i == 0:
                start_discrim_loss = discrim_loss

            self.discrim_optim.zero_grad()
            discrim_loss.backward()
            self.discrim_optim.step()

        if self._need_to_update_eval_statistics:
            self.eval_statistics['Discriminator Loss'] = ptu.get_numpy(discrim_loss).mean()
            self.eval_statistics['Discriminator Start Loss'] = ptu.get_numpy(start_discrim_loss).mean()

        gt.stamp('discriminator training', unique=False)

    def train_from_buffer(self, reward_kwargs=None):

        """
        Compute intrinsic reward: approximate lower bound to I(s'; z | s)
        """

        if self.relabel_rewards:

            rewards, (logp, logp_altz, denom), reward_diagnostics = self.calculate_intrinsic_rewards(
                self._obs[:self._cur_replay_size],
                self._next_obs[:self._cur_replay_size],
                self._latents[:self._cur_replay_size],
                reward_kwargs=reward_kwargs
            )
            orig_rewards = rewards.copy()
            rewards, postproc_dict = self.reward_postprocessing(rewards, reward_kwargs=reward_kwargs)
            reward_diagnostics.update(postproc_dict)
            self._rewards[:self._cur_replay_size] = np.expand_dims(rewards, axis=-1)

            gt.stamp('intrinsic reward calculation', unique=False)

        """
        Train policy
        """

        state_latents = np.concatenate([self._obs, self._latents], axis=-1)[:self._cur_replay_size]
        next_state_latents = np.concatenate(
            [self._true_next_obs, self._latents], axis=-1)[:self._cur_replay_size]

        for _ in range(self.num_policy_updates):
            batch = ppp.sample_batch(
                self.policy_batch_size,
                observations=state_latents,
                next_observations=next_state_latents,
                actions=self._actions[:self._cur_replay_size],
                rewards=self._rewards[:self._cur_replay_size],
            )
            batch = ptu.np_to_pytorch_batch(batch)
            self.policy_trainer.train_from_torch(batch)

        gt.stamp('policy training', unique=False)

        """
        Diagnostics
        """

        if self._need_to_update_eval_statistics:
            # self._need_to_update_eval_statistics = False
            self.eval_statistics.update(self.policy_trainer.eval_statistics)

            if self.relabel_rewards:
                self.eval_statistics.update(reward_diagnostics)

                self.eval_statistics.update(create_stats_ordered_dict(
                    'Discriminator Log Pis',
                    logp,
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Discriminator Alt Log Pis',
                    logp_altz,
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Intrinsic Reward Denominator',
                    denom,
                ))

                # Since replay buffer can be off-policy, want to give most up-to-date rewards
                if self._ptr < self._epoch_size:
                    if self._ptr == 0:
                        inds = np.r_[len(rewards)-self._epoch_size:len(rewards)]
                    else:
                        inds = np.r_[0:self._ptr,len(rewards)-self._ptr:len(rewards)]
                else:
                    inds = np.r_[self._ptr-self._epoch_size:self._ptr]

                self.eval_statistics.update(create_stats_ordered_dict(
                    'Intrinsic Rewards (Original)',
                    orig_rewards[inds],
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Intrinsic Rewards (Processed)',
                    rewards[inds],
                ))

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self.policy_trainer.end_epoch(epoch)

    @property
    def networks(self):
        return self.policy_trainer.networks + [self.discriminator]

    def get_snapshot(self):
        snapshot = dict(
            control_policy=self.control_policy,
            discriminator=self.discriminator,
        )

        for k, v in self.policy_trainer.get_snapshot().items():
            snapshot['policy_trainer/' + k] = v

        return snapshot
