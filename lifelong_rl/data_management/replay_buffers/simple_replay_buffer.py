from collections import OrderedDict

import numpy as np

from lifelong_rl.data_management.replay_buffers.replay_buffer import ReplayBuffer


def load_replay_buffer_from_snapshot(new_replay, snapshot, force_terminal_false=False):
    for t in range(len(snapshot['replay_buffer/actions'])):
        sample = dict(env_info=dict())
        for k in ['observation', 'action', 'reward',
                  'terminal', 'next_observation', 'env_info']:
            if len(snapshot['replay_buffer/%ss' % k]) == 0:
                continue
            if force_terminal_false and k == 'terminal':
                sample[k] = [False]
            else:
                sample[k] = snapshot['replay_buffer/%ss' % k][t]
        new_replay.add_sample(**sample)


class SimpleReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self._logprobs = np.zeros((max_replay_buffer_size, 1))
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

        self._top = 0
        self._size = 0

        self.total_entries = 0

    def obs_preproc(self, obs):
        return obs

    def obs_postproc(self, obs):
        return obs

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, env_info, **kwargs):
        self._observations[self._top] = self.obs_preproc(observation)
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = self.obs_preproc(next_observation)

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def add_sample_with_logprob(self, observation, action, reward, terminal,
                                next_observation, env_info, logprob, **kwargs):
        self._logprobs[self._top] = logprob
        self.add_sample(observation, action, reward, terminal, next_observation, env_info, **kwargs)

    def get_transitions(self):
        return np.concatenate([
            self._observations[:self._size],
            self._actions[:self._size],
            self._rewards[:self._size],
            self._terminals[:self._size],
            self._next_obs[:self._size],
        ], axis=1)

    def get_logprobs(self):
        return self._logprobs[:self._size].copy()

    def relabel_rewards(self, rewards):
        self._rewards[:len(rewards)] = rewards

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1
        self.total_entries += 1

    def random_batch(self, batch_size, min_pct=0, max_pct=1, include_logprobs=False, return_indices=False):
        indices = np.random.randint(
            int(min_pct * self._size),
            int(max_pct * self._size),
            batch_size,
        )
        batch = dict(
            observations=self.obs_postproc(self._observations[indices]),
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self.obs_postproc(self._next_obs[indices]),
        )
        if include_logprobs:
            batch['logprobs'] = self._logprobs[indices]
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        if return_indices:
            return batch, indices
        else:
            return batch

    def get_snapshot(self):
        return dict(
            observations=self._observations[:self._size],
            actions=self._actions[:self._size],
            rewards=self._rewards[:self._size],
            terminals=self._terminals[:self._size],
            next_observations=self._next_obs[:self._size],
            env_infos=self._env_infos,
        )

    def load_snapshot(self, snapshot):
        prev_info = snapshot['env_info']
        for t in range(snapshot['observations'].shape[0]):
            env_info = {key: prev_info[key][t] for key in prev_info}
            self.add_sample(
                observation=snapshot['observations'][t],
                action=snapshot['actions'][t],
                reward=snapshot['rewards'][t],
                next_observation=snapshot['next_observations'][t],
                terminal=snapshot['terminals'][t],
                env_info=env_info,
            )

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def top(self):
        return self._top

    def num_steps_can_sample(self):
        return self._size

    def max_replay_buffer_size(self):
        return self._max_replay_buffer_size

    def obs_dim(self):
        return self._observation_dim

    def action_dim(self):
        return self._action_dim

    def get_diagnostics(self):
        return OrderedDict([
            ('size', self._size)
        ])
