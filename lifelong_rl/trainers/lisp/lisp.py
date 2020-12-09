import numpy as np
import torch

import gtimer as gt

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.trainers.lisp.mb_skill import MBSkillTrainer
import lifelong_rl.util.pythonplusplus as ppp
from lifelong_rl.util.eval_util import create_stats_ordered_dict


class LiSPTrainer(MBSkillTrainer):

    """
    Lifelong Skill-Space Planning (Lu et al. 2020).
    Learning skills using model-based rollouts with a skill-practice distribution.
    Should be combined with an MPC planner for acting.
    """

    def __init__(
            self,
            skill_practice_dist,        # Associated skill-practice distribution for generating latents
            skill_practice_trainer,     # Associated trainer for skill-practice distribution (ex. SAC)
            practice_train_steps=32,    # Number of training steps for skill-practice distribution
            practice_batch_size=256,    # Batch size of training skill-practice distribution
            num_unif_train_calls=0,     # Optionally, don't use skill practice distribution early on
            epsilon_greedy=0.,          # Optionally, sample latents from uniform with probability eps
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.skill_practice_dist = skill_practice_dist
        self.skill_practice_trainer = skill_practice_trainer
        self.practice_train_steps = practice_train_steps
        self.practice_batch_size = practice_batch_size
        self.num_unif_train_calls = num_unif_train_calls
        self.epsilon_greedy = epsilon_greedy

    def generate_latents(self, obs):
        if self._train_calls < self.num_unif_train_calls:
            return super().generate_latents(obs)
        latents, *_ = self.skill_practice_dist(ptu.from_numpy(obs))
        latents = ptu.get_numpy(latents)
        if self.epsilon_greedy > 0:
            unif_r = np.random.uniform(0, 1, size=latents.shape[0])
            eps_replace = unif_r < self.epsilon_greedy
            unif_latents = super().generate_latents(obs[eps_replace])
            latents[eps_replace] = unif_latents
        return latents

    def train_from_torch(self, batch):
        super().train_from_torch(batch)

        if self._train_calls % self.train_every > 0:
            return

        for _ in range(self.practice_train_steps):
            batch = ppp.sample_batch(
                self.practice_batch_size,
                observations=self._obs[:self._cur_replay_size],
                next_observations=self._next_obs[:self._cur_replay_size],
                actions=self._latents[:self._cur_replay_size],
                rewards=self._rewards[:self._cur_replay_size],
            )
            batch = ptu.np_to_pytorch_batch(batch)
            self.skill_practice_trainer.train_from_torch(batch)

        for k, v in self.skill_practice_trainer.get_diagnostics().items():
            self.eval_statistics['prior_trainer/' + k] = v

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

                # Adjustment so intrinsic rewards are over last epoch
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

    def end_epoch(self, epoch):
        super().end_epoch(epoch)
        self.skill_practice_trainer.end_epoch(epoch)

    @property
    def networks(self):
        return self.skill_practice_trainer.networks + self.policy_trainer.networks + [
            self.discriminator, self.skill_practice_dist,
        ]

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot['skill_practice'] = self.skill_practice_dist

        for k, v in self.skill_practice_trainer.get_snapshot().items():
            snapshot['skill_practice_trainer/' + k] = v

        return snapshot
