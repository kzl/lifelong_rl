import gtimer as gt
import numpy as np
import torch

import lifelong_rl.samplers.utils.model_rollout_functions as mrf
import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.trainers.dads.dads import DADSTrainer
from lifelong_rl.util.eval_util import create_stats_ordered_dict


def always_one(train_steps):
    return 1


class MBSkillTrainer(DADSTrainer):

    """
    Model-Based Skill Learning (introduced in Lu et al. 2020).
    Uses model-generated rollouts for training skill policy.
    """

    def __init__(
            self,
            dynamics_model,                 # Associated dynamics model for generating rollouts
            rollout_len_func=always_one,    # Length of generated rollouts
            num_model_samples=400,          # Number of timesteps to generate per train call
            disagreement_threshold=None,    # Penalize rollouts higher than disagreement threshold
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.dynamics_model = dynamics_model
        self.rollout_len_func = rollout_len_func
        self.num_model_samples = num_model_samples
        self.disagreement_threshold = disagreement_threshold

        self._epoch_size = self.num_model_samples
        self._modeL_disagreements = np.zeros(self.replay_size)

    def add_sample(self, obs, next_obs, true_next_obs, action, latent, disagreement=0, **kwargs):
        self._modeL_disagreements[self._ptr] = disagreement
        super().add_sample(obs, next_obs, true_next_obs, action, latent)

    def generate_latents(self, obs):
        return np.random.uniform(low=-1, high=1, size=(obs.shape[0], self.latent_dim))

    def generate_paths(self, **kwargs):
        paths, path_disagreements = mrf.policy_latent_with_disagreement(
            kwargs['dynamics_model'],
            kwargs['control_policy'],
            kwargs['start_states'],
            kwargs['latents'],
            max_path_length=kwargs['rollout_len'],
            terminal_cutoff=None,
        )
        return paths, path_disagreements

    def reward_postprocessing(self, rewards, reward_kwargs=None, *args, **kwargs):
        if self.disagreement_threshold is None:
            return super().reward_postprocessing(rewards)
        rewards, diagnostics = super().reward_postprocessing(rewards)

        disagreements = reward_kwargs['disagreements']
        violated = disagreements > self.disagreement_threshold
        rewards[violated] = self.reward_bounds[0]

        if self._need_to_update_eval_statistics:
            diagnostics.update(create_stats_ordered_dict(
                'Model Disagreement',
                disagreements,
            ))
            diagnostics['Pct of Timesteps over Disagreement Cutoff'] = np.mean(violated)

        return rewards, diagnostics

    def train_from_torch(self, batch):

        self._train_calls += 1
        if self._train_calls % self.train_every > 0:
            return

        rollout_len = self.rollout_len_func(self._n_train_steps_total)
        num_model_rollouts = max(self.num_model_samples // rollout_len, 1)
        self.eval_statistics['Rollout Length'] = rollout_len

        real_batch = self.replay_buffer.random_batch(num_model_rollouts)
        start_states = real_batch['observations']
        latents = self.generate_latents(start_states)

        observations = np.zeros((self.num_model_samples, self.obs_dim))
        next_observations = np.zeros((self.num_model_samples, self.obs_dim))
        actions = np.zeros((self.num_model_samples, self.action_dim))
        unfolded_latents = np.zeros((self.num_model_samples, self.latent_dim))
        disagreements = np.zeros(self.num_model_samples)

        num_samples, b_ind, num_traj = 0, 0, 0
        while num_samples < self.num_model_samples:
            e_ind = b_ind + 4192 // rollout_len
            with torch.no_grad():
                paths, path_disagreements = self.generate_paths(
                    dynamics_model=self.dynamics_model,
                    control_policy=self.control_policy,
                    start_states=start_states[b_ind:e_ind],
                    latents=ptu.from_numpy(latents[b_ind:e_ind]),
                    rollout_len=rollout_len,
                )

            b_ind = e_ind

            path_disagreements = ptu.get_numpy(path_disagreements)
            for i, path in enumerate(paths):
                clipped_len = min(len(path['observations'] - (self.empowerment_horizon-1)),
                                  self.num_model_samples - num_samples)
                bi, ei = num_samples, num_samples + clipped_len

                if self.empowerment_horizon > 1:
                    path['observations'] = path['observations'][:-(self.empowerment_horizon-1)]
                    path['next_observations'] = path['next_observations'][(self.empowerment_horizon-1):
                                                                          (self.empowerment_horizon-1)+clipped_len]
                    path['actions'] = path['actions'][:-(self.empowerment_horizon-1)]

                observations[bi:ei] = path['observations'][:clipped_len]
                next_observations[bi:ei] = path['next_observations'][:clipped_len]
                actions[bi:ei] = path['actions'][:clipped_len]
                unfolded_latents[bi:ei] = latents[num_traj:num_traj + 1]
                disagreements[bi:ei] = path_disagreements[i,:clipped_len]

                num_samples += clipped_len
                num_traj += 1

                if num_samples >= self.num_model_samples:
                    break

        gt.stamp('generating rollouts', unique=False)

        if not self.relabel_rewards:
            rewards, (logp, logp_altz, denom), reward_diagnostics = self.calculate_intrinsic_rewards(
                observations, next_observations, unfolded_latents)
            orig_rewards = rewards.copy()
            rewards, postproc_dict = self.reward_postprocessing(
                rewards, reward_kwargs=dict(disagreements=disagreements))
            reward_diagnostics.update(postproc_dict)

            if self._need_to_update_eval_statistics:
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

                self.eval_statistics.update(create_stats_ordered_dict(
                    'Intrinsic Rewards (Original)',
                    orig_rewards,
                ))
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Intrinsic Rewards (Processed)',
                    rewards,
                ))

            gt.stamp('intrinsic reward calculation', unique=False)

        if self._need_to_update_eval_statistics:
            self.eval_statistics.update(create_stats_ordered_dict(
                'Latents',
                latents,
            ))

        for t in range(self.num_model_samples):
            self.add_sample(
                observations[t],
                next_observations[t],
                next_observations[t],  # fix this
                actions[t],
                unfolded_latents[t],
                disagreement=disagreements[t],
            )

        gt.stamp('policy training', unique=False)

        self.train_discriminator(observations, next_observations, unfolded_latents)

        reward_kwargs = dict(
            disagreements=self._modeL_disagreements[:self._cur_replay_size]
        )
        self.train_from_buffer(reward_kwargs=reward_kwargs)

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

        self._epoch_size = len(epoch_obs)

        gt.stamp('policy training', unique=False)

        self.train_from_torch(None)
