import numpy as np

import copy

from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
import lifelong_rl.torch.pytorch_util as ptu


class HERTrainer(TorchTrainer):

    """
    Hindsight Experience Replay (Andrychowicz et al. 2017).
    Duplicates transitions using different goals with particular reward function.
    """

    def __init__(
            self,
            policy_trainer,
            replay_buffer,
            state_dim,
            reward_func=None,
            relabel_goal_func=None,
            num_sampled_goals=1,
            relabel_method='future',
            policy_batch_size=256,
            num_policy_steps=1,
    ):
        super().__init__()

        self.policy_trainer = policy_trainer
        self.replay_buffer = replay_buffer
        self.state_dim = state_dim
        self.reward_func = reward_func
        self.relabel_goal_func = relabel_goal_func
        self.num_sampled_goals = num_sampled_goals
        self.relabel_method = relabel_method
        self.policy_batch_size = policy_batch_size
        self.num_policy_steps = num_policy_steps

        # Default goal methods: L2 distance & goal = desired state
        if self.reward_func is None:
            self.reward_func = lambda s, a, ns, g: np.linalg.norm(ns[:2] - g) < .1
        if self.relabel_goal_func is None:
            self.relabel_goal_func = lambda s, a, ns, g: ns

    def train_from_paths(self, paths):

        """
        Path processing
        """

        paths = copy.deepcopy(paths)
        for path in paths:
            obs, next_obs = path['observations'], path['next_observations']
            states, next_states = obs[:,:self.state_dim], next_obs[:,:self.state_dim]
            goals = obs[:,self.state_dim:2*self.state_dim]
            actions = path['actions']
            terminals = path['terminals']  # this is probably always False, but might want it?
            path_len = len(obs)

            # Relabel goals based on transitions taken
            relabeled_goals = []
            for t in range(len(obs)):
                relabeled_goals.append(self.relabel_goal_func(
                    states[t], actions[t], next_states[t], goals[t],
                ))
            relabeled_goals = np.array(relabeled_goals)

            # Add transitions & resampled goals to replay buffer
            for t in range(path_len):
                goals_t = goals[t:t+1]
                for _ in range(self.num_sampled_goals):
                    if self.relabel_method == 'future':
                        goal_inds = np.random.randint(t, path_len, self.num_sampled_goals)
                        goals_t = np.concatenate([goals_t, relabeled_goals[goal_inds]], axis=0)
                    else:
                        raise NotImplementedError

                for k in range(len(goals_t)):
                    if not self.learn_reward_func:
                        r = self.reward_func(states[t], actions[t], next_states[t], goals_t[k])
                    else:
                        r = ptu.get_numpy(
                            self.learned_reward_func(
                                ptu.from_numpy(
                                    np.concatenate([next_states[t], goals[t]])))).mean()
                    self.replay_buffer.add_sample(
                        observation=np.concatenate([states[t], goals_t[k], obs[t,2*self.state_dim:]]),
                        action=actions[t],
                        reward=r,
                        terminal=terminals[t],  # not obvious what desired behavior is
                        next_observation=np.concatenate(
                            [next_states[t,:self.state_dim], goals_t[k], obs[t,2*self.state_dim:]]),
                        env_info=None,
                    )

        """
        Off-policy training
        """

        for _ in range(self.num_policy_steps):
            train_data = self.replay_buffer.random_batch(self.policy_batch_size)
            self.policy_trainer.train(train_data)

    def get_diagnostics(self):
        return self.policy_trainer.get_diagnostics()

    def end_epoch(self, epoch):
        self.policy_trainer.end_epoch(epoch)

    @property
    def networks(self):
        return self.policy_trainer.networks

    def get_snapshot(self):
        return self.policy_trainer.get_snapshot()
