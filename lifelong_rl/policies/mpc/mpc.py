import numpy as np
import torch

from collections import OrderedDict
import copy

from lifelong_rl.optimizers.random_shooting.mppi import MPPIOptimizer
from lifelong_rl.policies.base.base import ExplorationPolicy
import lifelong_rl.torch.pytorch_util as ptu
import lifelong_rl.torch.risk_aversion as risk_aversion
from lifelong_rl.util.eval_util import create_stats_ordered_dict


class MPCPolicy(ExplorationPolicy):

    """
    General class MPC policy. Directly using MPCPolicy does planning over actions using
    trajectory sampling, but this can be modified as desired by subclasses.
    """

    def __init__(
            self,
            env,                                # Exploration env (note: this matters if we use gt model)
            dynamics_model,                     # Parallelized dynamics model ensemble
            plan_dim,                           # Dimension of the plan (i.e. action dim)
            discount=.99,                       # Discount factor to apply when planning
            horizon=25,                         # Horizon for plan; not necessarily equating to timesteps
            repeat_length=1,                    # Repeat same plan for repeat_length timesteps
            plan_every=1,                       # Do replanning every plan_every timesteps
            temperature=.01,                    # MPPI temperature; temperature -> 0 reduces to random shooting
            noise_std=1.,                       # Std of noise for the first iteration of planning
            num_rollouts=400,                   # Number of trajectories to roll out per iteration
            num_particles=5,                    # Number of particles *per model* of dynamics ensemble
            planning_iters=3,                   # Number of planning iterations
            sampling_mode='ts',                 # How to sample from the dynamics ensemble
            sampling_kwargs=None,               # Arguments of sampling mode
            polyak=0.,                          # Between iterations, polyak averaging/learning rate of plan
            filter_coefs=(1.,0.,0.),            # Smoothing filter coefficients for noise
            use_plan_mean=True,                 # If False, use best plan found rather than MPC output
            risk_mode='neutral',                # Method of combining particle returns
            risk_kwargs=None,                   # Parameters for risk function
            predict_terminal=False,             # Whether or not to predict if a trajectory terminates
            use_gt_model=False,                 # Can optionally plan with ground truth model
            reward_func=None,                   # Can optionally use a given reward function
            terminal_func=None,                 # Can optionally use a given terminal function
            value_func=None,                    # Can optionally use a terminal value function
            reward_func_kwargs=None,            # Additional kwargs for the reward func when it is called
            value_func_kwargs=None,             # Additional kwargs for the value func when it is called
    ):
        self.env = env
        self.dynamics_model = dynamics_model
        self.num_models = self.dynamics_model.ensemble_size if self.dynamics_model is not None else 1
        self.plan_dim = plan_dim
        self.discount = discount
        self.horizon = horizon
        self.temperature = temperature
        self.noise_std = noise_std
        self.num_rollouts = num_rollouts
        self.num_particles = num_particles
        self.planning_iters = planning_iters
        self.polyak = polyak
        self.use_plan_mean = use_plan_mean
        self.repeat_length = repeat_length
        self.plan_every = plan_every
        self.sampling_mode = sampling_mode
        self.sampling_kwargs = sampling_kwargs if sampling_kwargs is not None else dict()
        self.risk_mode = risk_mode
        self.risk_kwargs = risk_kwargs if risk_kwargs is not None else dict()
        self.predict_terminal = predict_terminal
        self.use_gt_model = use_gt_model
        self.filter_coefs = filter_coefs
        self.reward_func = reward_func
        self.terminal_func = terminal_func
        self.value_func = value_func
        self.reward_func_kwargs = reward_func_kwargs if not None else dict()
        self.value_func_kwargs = value_func_kwargs if not None else dict()

        # We can support CEM if we want, but MPPI generally has better performance
        self.optimizer = MPPIOptimizer(
            self.horizon * self.plan_dim,
            self.planning_iters,
            self.num_rollouts,
            self.temperature,
            self.cost_function,
            polyak=self.polyak,
            filter_noise=self.filter_noise,
        )

        self.initial_plan = None
        self._observation = None
        self._steps_until_next_plan = 0
        self._advance_plan_counter = self.repeat_length - 1
        self.initialize_plan()

        self.diagnostics = OrderedDict()
        self._need_to_update_diagnostics = True
        self._debug_flag = False

        self.num_timesteps = 0

    def get_action(self, observation):
        if self._steps_until_next_plan == 0:
            self._steps_until_next_plan = self.plan_every

            # It is good to use var because CMA methods use covariance updates
            initial_plan = self.initial_plan
            initial_var = np.ones(initial_plan.shape) * (self.noise_std ** 2)

            self._observation = observation
            self.initial_plan, optim_diagnostics = self.optimizer.optimize(initial_plan, initial_var)

            if self._need_to_update_diagnostics:
                self._need_to_update_diagnostics = False
                self.diagnostics.update(optim_diagnostics)

        self._steps_until_next_plan -= 1
        self.num_timesteps += 1

        action = self.convert_plan_to_action(observation, self.initial_plan[:self.plan_dim], deterministic=True)
        self.advance_plan()

        return action, {}

    def convert_plan_to_action(self, obs, plan, deterministic=False):
        # both obs and plan are in numpy; plan is just directly the action here
        return plan

    def convert_plans_to_actions(self, obs, plans):
        # note that obs and plans are in *torch* here, used in planning
        return plans

    def advance_plan(self):
        # if we replan more than we get new actions, no need to advance plan
        if self._advance_plan_counter == 0 and self.repeat_length < self.plan_every:
            self.initial_plan = np.concatenate(
                [np.copy(self.initial_plan)[self.plan_dim:], np.zeros(self.plan_dim)]
            )
            self._advance_plan_counter = self.repeat_length
        self._advance_plan_counter -= 1

    def get_plan_values(self, obs, plans):
        n_part, batch_size = plans.shape[1], 32768

        returns = ptu.zeros(n_part)
        bi, ei = 0, batch_size
        while bi < n_part:
            returns[bi:ei] = self.get_plan_values_batch(obs[bi:ei], plans[:, bi:ei])
            bi, ei = bi + batch_size, ei + batch_size

        return returns

    def get_plan_values_batch(self, obs, plans):
        """
        Get corresponding values of the plans (higher corresponds to better plans). Classes
        that don't want to plan over actions or use trajectory sampling can reimplement
        convert_plans_to_actions (& convert_plan_to_action) and/or predict_transition.
        plans is input as as torch (horizon_length, num_particles (total), plan_dim).
        We maintain trajectory infos as torch (n_part, info_dim (ex. obs_dim)).
        """

        if self.use_gt_model:
            return self.get_plan_values_batch_gt(obs, plans)

        n_part = plans.shape[1]  # *total* number of particles, NOT num_particles

        discount = 1
        returns, dones, infos = ptu.zeros(n_part), ptu.zeros(n_part), dict()

        # The effective planning horizon is self.horizon * self.repeat_length
        for t in range(self.horizon):
            for k in range(self.repeat_length):
                cur_actions = self.convert_plans_to_actions(obs, plans[t])
                obs, cur_rewards, cur_dones = self.predict_transition(obs, cur_actions, infos)
                returns += discount * (1 - dones) * cur_rewards
                discount *= self.discount
                if self.predict_terminal:
                    dones = torch.max(dones, cur_dones.float())

        self.diagnostics.update(create_stats_ordered_dict(
            'MPC Termination',
            ptu.get_numpy(dones),
        ))

        if self.value_func is not None:
            terminal_values = self.value_func(obs, **self.value_func_kwargs).view(-1)
            returns += discount * (1-dones) * terminal_values

            self.diagnostics.update(create_stats_ordered_dict(
                'MPC Terminal Values',
                ptu.get_numpy(terminal_values),
            ))

        return returns

    def get_plan_values_batch_gt(self, obs, plans):
        returns = ptu.zeros(plans.shape[1])
        obs, plans = ptu.get_numpy(obs), ptu.get_numpy(plans)
        final_obs = np.copy(obs)
        for i in range(plans.shape[1]):
            returns[i], final_obs[i] = self._get_true_env_value(obs[i], plans[:, i])
        if self.value_func is not None:
            returns += (self.discount ** (self.horizon * self.repeat_length)) * (
                self.value_func(ptu.from_numpy(final_obs), **self.value_func_kwargs)
            )
        return returns

    def _get_true_env_value(self, obs, plan):
        env = copy.deepcopy(self.env)
        env.sim.set_state(copy.deepcopy(self.env.sim.get_state()))

        discount, plan_return, done = 1, 0, 0
        for t in range(self.horizon):
            for k in range(self.repeat_length):
                if len(plan.shape) == 1:  # plan is in vertical form
                    cur_plan = plan[t * self.plan_dim:(t + 1) * self.plan_dim]
                else:  # plan is in timestep form
                    cur_plan = plan[t]
                cur_action = self.convert_plan_to_action(obs, cur_plan)
                next_obs, r, d, _ = env.step(cur_action)
                if self.reward_func is not None:
                    r = self.reward_func(
                        obs.reshape(1, -1), cur_action.reshape(1, -1), next_obs.reshape(1, -1),
                        **self.reward_func_kwargs
                    )[0]

                plan_return += discount * r
                obs = next_obs
                discount *= self.discount

                if d:
                    done = True
                    break
            if done:
                break

        return plan_return, obs

    def _get_model_plan_value(self, obs, plan):
        obs, plan = ptu.from_numpy(obs), ptu.from_numpy(plan)
        plans = plan.view(-1, self.horizon, self.plan_dim)
        plans = plans.permute(1, 0, 2)
        obs = obs.view(1, -1)
        returns = self.get_plan_values(obs, plans)
        return ptu.get_numpy(returns).mean()

    def predict_transition(self, obs, actions, infos):
        if self.sampling_mode == 'ts':
            preds = self._predict_transition_ts(obs, actions, infos)
        elif self.sampling_mode == 'uniform':
            preds = self._predict_transition_uniform(obs, actions, infos)
        else:
            raise NotImplementedError('MPC sampling_mode not recognized')

        next_obs, rewards, dones = obs + preds[:, 2:], preds[:, 0], preds[:, 1] > 0.5
        if self.reward_func is not None:
            given_rewards = self.reward_func(obs, actions, next_obs, num_timesteps=self.num_timesteps)
            self.diagnostics.update(create_stats_ordered_dict(
                'Reward Squared Error',
                ptu.get_numpy((given_rewards - rewards) ** 2),
                always_show_all_stats=True,
            ))
            rewards = given_rewards

        return next_obs, rewards, dones

    def cost_function(self, x, it=0):
        with torch.no_grad():
            plans, obs = self.create_particles(x, self._observation)
            returns = self.get_plan_values(obs, plans).view(self.num_rollouts, -1)
            weighted_returns = self.get_weighted_returns(returns)
            costs = -ptu.get_numpy(weighted_returns)

        if self._need_to_update_diagnostics:
            self.diagnostics.update(create_stats_ordered_dict(
                'Iteration %d Returns' % it,
                ptu.get_numpy(weighted_returns),
                always_show_all_stats=True,
            ))

            self.diagnostics.update(create_stats_ordered_dict(
                'Iteration %d Particle Stds' % it,
                np.std(ptu.get_numpy(returns), axis=-1),
                always_show_all_stats=True,
            ))

            variance = weighted_returns.var()
            particle_variance = returns.var(dim=-1)
            self.diagnostics['Return Leftover Variance'] = \
                ptu.get_numpy(variance - particle_variance.mean()).mean()

        return costs

    def filter_noise(self, noise):
        noise = noise.reshape(-1, self.horizon, self.plan_dim)
        for t in range(2, self.horizon):
            noise[:, t] = self.filter_coefs[0] * noise[:, t] + \
                          self.filter_coefs[1] * noise[:, t - 1] + \
                          self.filter_coefs[2] * noise[:, t - 2]
        noise = noise.reshape(noise.shape[0], -1)
        return noise

    def create_particles(self, plans, obs, n_part=None):
        n_opt = plans.shape[0]
        if n_part is None:
            n_part = self.num_particles * self.num_models

        # (N, H*m)
        plans = ptu.from_numpy(plans)
        # (N, H, m)
        plans = plans.view(-1, self.horizon, self.plan_dim)
        # (H, N, m)
        transposed = plans.transpose(0, 1)
        # (H, N, 1, m)
        expanded = transposed[:, :, None]
        # (H, N, P, m)
        tiled = expanded.expand(-1, -1, n_part, -1)
        # (H, N*P, m)
        plans = tiled.contiguous().view(self.horizon, -1, self.plan_dim)

        # (n,)
        obs = ptu.from_numpy(self._observation)
        # (1, n)
        obs = obs[None]
        # (N*P, n)
        obs = obs.expand(n_opt * n_part, -1)

        return plans, obs

    def get_weighted_returns(self, returns):
        sorted_returns, _ = torch.sort(returns, dim=-1)
        weighted_returns = sorted_returns * risk_aversion.get_mask(
            self.risk_mode, sorted_returns.shape[1], self.risk_kwargs)
        return weighted_returns.sum(dim=1)

    """
    Some variants of MPC may want to perform trajectory sampling
    """

    def _predict_transition_ts(self, obs, actions, infos):
        # parallelized ensemble handles TS cleanly
        preds = self.dynamics_model.forward(torch.cat(
            (self._expand_to_ts_form(obs), self._expand_to_ts_form(actions)), dim=-1
        ))
        preds = self._flatten_from_ts(preds)
        return preds

    def _predict_transition_uniform(self, obs, actions, infos):
        preds = self.dynamics_model.sample(torch.cat((obs, actions), dim=-1))
        return preds

    def _predict_transition_penalty_soft(self, obs, actions, infos):
        # soft penalty for disagreement, see MOPO
        raise NotImplementedError

    def _expand_to_ts_form(self, x):
        d = x.shape[-1]
        reshaped = x.view(-1, self.num_models, self.num_particles, d)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(self.num_models, -1, d)
        return reshaped

    def _flatten_from_ts(self, y):
        d = y.shape[-1]
        reshaped = y.view(self.num_models, -1, self.num_particles, d)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(-1, d)
        return reshaped

    """
    Boilerplate functions
    """

    def initialize_plan(self):
        self.initial_plan = np.zeros(self.horizon * self.plan_dim)

    def reset(self):
        # End of episode
        self.initialize_plan()
        self._steps_until_next_plan = 0
        self._advance_plan_counter = self.repeat_length - 1

    def get_diagnostics(self):
        return self.diagnostics

    def end_epoch(self, epoch):
        self._need_to_update_diagnostics = True
