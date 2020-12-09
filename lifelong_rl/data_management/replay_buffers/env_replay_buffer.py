import gym
from gym.spaces import Discrete

from lifelong_rl.data_management.replay_buffers.simple_replay_buffer import SimpleReplayBuffer
from lifelong_rl.envs.env_utils import get_dim
import numpy as np


class EnvReplayBuffer(SimpleReplayBuffer):

    def __init__(
            self,
            max_replay_buffer_size,
            env,
            env_info_sizes=None,
    ):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        self.env = env
        self._ob_space = env.observation_space
        self._action_space = env.action_space
        self._meta_infos = []

        if env_info_sizes is None:
            if hasattr(env, 'info_sizes'):
                env_info_sizes = env.info_sizes
            else:
                env_info_sizes = dict()

        if isinstance(self._ob_space, gym.spaces.Box):
            self._ob_shape = self._ob_space.shape
        else:
            self._ob_shape = None

        super().__init__(
            max_replay_buffer_size=max_replay_buffer_size,
            observation_dim=get_dim(self._ob_space),
            action_dim=get_dim(self._action_space),
            env_info_sizes=env_info_sizes
        )

    def obs_preproc(self, obs):
        if len(obs.shape) > len(self._ob_space.shape):
            obs = np.reshape(obs, (obs.shape[0], self._observation_dim))
        else:
            obs = np.reshape(obs, (self._observation_dim,))
        return obs

    def obs_postproc(self, obs):
        if self._ob_shape is None:
            return obs
        if len(obs.shape) > 1:
            obs = np.reshape(obs, (obs.shape[0], *self._ob_shape))
        else:
            obs = np.reshape(obs, self._ob_shape)
        return obs

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, env_info=None, **kwargs):
        if hasattr(self.env, 'get_meta_infos'):
            self._meta_infos.append(self.env.get_meta_infos())
        if env_info is None:
            env_info = dict()
        if isinstance(self._action_space, Discrete):
            new_action = np.zeros(self._action_dim)
            new_action[action] = 1
        else:
            new_action = action
        return super().add_sample(
            observation=observation,
            action=new_action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            env_info=env_info,
            **kwargs
        )

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot['meta_infos'] = self._meta_infos
        return snapshot
