from collections import OrderedDict
import gtimer as gt

import numpy as np
import torch

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
import lifelong_rl.samplers.utils.model_rollout_functions as mrf
from lifelong_rl.util.eval_util import create_stats_ordered_dict


def always_one(train_steps):
    return 1


class MBPOTrainer(TorchTrainer):

    """
    Model-based Policy Optimization (Janner et al. 2019).
    Policy optimization using synthetic model-based rollouts.
    Supports various types of policy optimization procedures using a model.
    """

    def __init__(
            self,
            policy_trainer,                 # Associated policy trainer to learn from generated data
            dynamics_model,                 # Note that MBPOTrainer is not responsible for training this
            replay_buffer,                  # The true replay buffer, used for generating starting states
            generated_data_buffer,          # Replay buffer solely consisting of synthetic transitions
            rollout_len_func=always_one,    # Rollout length as a function of number of train calls
            num_model_rollouts=400,         # Number of *transitions* to generate per env timestep
            rollout_generation_freq=1,      # Can save time by only generating data when model is updated
            num_policy_updates=20,          # Number of policy updates per env timestep (should be > 1)
            rollout_batch_size=int(1e3),    # Maximum batch size for generating rollouts (i.e. GPU memory limit)
            real_data_pct=0.05,             # Percentage of real data used for policy training
            sampling_mode='uniform',        # Type of sampling to perform (original MBPO is 'uniform')
            sampling_kwargs=None,           # Arguments required for specific sampling procedure
            **kwargs
    ):
        super().__init__()

        self.policy_trainer = policy_trainer
        self.policy = policy_trainer.policy
        self.dynamics_model = dynamics_model
        self.replay_buffer = replay_buffer
        self.generated_data_buffer = generated_data_buffer
        self.rollout_len_func = rollout_len_func

        self.num_model_rollouts = num_model_rollouts
        self.rollout_generation_freq = rollout_generation_freq
        self.num_policy_updates = num_policy_updates
        self.rollout_batch_size = rollout_batch_size
        self.real_data_pct = real_data_pct
        self.sampling_mode = sampling_mode
        self.sampling_kwargs = sampling_kwargs if sampling_kwargs is not None else dict()

        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

    def train_from_torch(self, batch):
        # We only use the original batch to get the batch size for policy training

        """
        Generate synthetic data using dynamics model
        """
        if self._n_train_steps_total % self.rollout_generation_freq == 0:
            rollout_len = self.rollout_len_func(self._n_train_steps_total)
            total_samples = self.rollout_generation_freq * self.num_model_rollouts

            num_samples, generated_rewards, terminated = 0, np.array([]), []
            while num_samples < total_samples:
                batch_samples = min(self.rollout_batch_size, total_samples - num_samples)
                real_batch = self.replay_buffer.random_batch(batch_samples)
                start_states = real_batch['observations']

                with torch.no_grad():
                    paths = self.sample_paths(start_states, rollout_len)

                for path in paths:
                    self.generated_data_buffer.add_path(path)
                    num_samples += len(path['observations'])
                    generated_rewards = np.concatenate([generated_rewards, path['rewards'][:,0]])
                    terminated.append(path['terminals'][-1,0] > 0.5)

                if num_samples >= total_samples:
                    break

            gt.stamp('generating rollouts', unique=False)

        """
        Update policy on both real and generated data
        """

        batch_size = batch['observations'].shape[0]
        n_real_data = int(self.real_data_pct * batch_size)
        n_generated_data = batch_size - n_real_data

        for _ in range(self.num_policy_updates):
            batch = self.replay_buffer.random_batch(n_real_data)
            generated_batch = self.generated_data_buffer.random_batch(
                n_generated_data)

            for k in ('rewards', 'terminals', 'observations',
                      'actions', 'next_observations'):
                batch[k] = np.concatenate((batch[k], generated_batch[k]), axis=0)
                batch[k] = ptu.from_numpy(batch[k])

            self.policy_trainer.train_from_torch(batch)

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics and self._n_train_steps_total % self.rollout_generation_freq == 0:
            self._need_to_update_eval_statistics = False

            self.eval_statistics['MBPO Rollout Length'] = rollout_len
            self.eval_statistics.update(create_stats_ordered_dict(
                'MBPO Reward Predictions',
                generated_rewards,
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'MBPO Rollout Terminations',
                np.array(terminated).astype(float),
            ))

        self._n_train_steps_total += 1

    def sample_paths(self, start_states, rollout_len):
        if self.sampling_mode == 'uniform':
            # Sample uniformly from a model of the ensemble (original MBPO; Janner et al. 2019)
            paths = mrf.policy(
                self.dynamics_model,
                self.policy_trainer.policy,
                start_states,
                max_path_length=rollout_len,
            )

        elif self.sampling_mode == 'mean_disagreement':
            # Sample with penalty for disagreement of the mean (MOReL; Kidambi et al. 2020)
            paths, disagreements = mrf.policy_with_disagreement(
                self.dynamics_model,
                self.policy_trainer.policy,
                start_states,
                max_path_length=rollout_len,
                disagreement_type='mean',
            )
            disagreements = ptu.get_numpy(disagreements)

            threshold, penalty = self.sampling_kwargs['threshold'], self.sampling_kwargs['penalty']
            total_penalized, total_transitions = 0, 0
            for i, path in enumerate(paths):
                mask = np.zeros(len(path['rewards']))
                disagreement_values = disagreements[i]
                for t in range(len(path['rewards'])):
                    cur_mask = disagreement_values[t] > threshold
                    if t == 0:
                        mask[t] = cur_mask
                    elif cur_mask or mask[t-1] > 0.5:
                        mask[t] = 1.
                    else:
                        mask[t] = 0.
                mask = mask.reshape(len(mask), 1)
                path['rewards'] = (1-mask) * path['rewards'] - mask * penalty
                total_penalized += mask.sum()
                total_transitions += len(path)

            self.eval_statistics['Percent of Transitions Penalized'] = total_penalized / total_transitions
            self.eval_statistics.update(create_stats_ordered_dict(
                'Disagreement Values',
                disagreements,
            ))

        elif self.sampling_mode == 'var_disagreement':
            # Sample with penalty for disagreement of the variance (MOPO; Yu et al. 2020)
            paths, disagreements = mrf.policy_with_disagreement(
                self.dynamics_model,
                self.policy_trainer.policy,
                start_states,
                max_path_length=rollout_len,
                disagreement_type='var',
            )
            disagreements = ptu.get_numpy(disagreements)

            reward_penalty = self.sampling_kwargs['reward_penalty']
            for i, path in enumerate(paths):
                path_disagreements = disagreements[i,:len(path['rewards'])].reshape(*path['rewards'].shape)
                path['rewards'] -= reward_penalty * path_disagreements

            self.eval_statistics.update(create_stats_ordered_dict(
                'Disagreement Values',
                disagreements,
            ))

        else:
            raise NotImplementedError

        return paths

    def get_diagnostics(self):
        self.eval_statistics.update(self.policy_trainer.eval_statistics)
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True
        self.policy_trainer.end_epoch(epoch)

    @property
    def networks(self):
        return self.policy_trainer.networks

    def get_snapshot(self):
        mbpo_snapshot = dict(
            dynamics_model=self.dynamics_model
        )
        mbpo_snapshot.update(self.policy_trainer.get_snapshot())
        return mbpo_snapshot
