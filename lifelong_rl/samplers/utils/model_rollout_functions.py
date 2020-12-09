import numpy as np
import torch

import lifelong_rl.torch.pytorch_util as ptu

"""
Interface for generating rollouts from a dynamics model. Designed to support many
types of rollouts, and measuring different metrics. Three main types of rollouts
are (planned to be) supported, depending on GPU memory and other use cases:
    1) (standard_)rollout: keep everything in GPU memory, return all paths
    2) online_rollout: keep everything in GPU memory, but only store current
                       transitions, so only return final/cumulative info in
                       the torch format
    3) np_rollout: only use GPU as necessary, store everything in numpy, return
                   all paths (which are in numpy)
"""


def _create_full_tensors(start_states, max_path_length, obs_dim, action_dim):
    num_rollouts = start_states.shape[0]
    observations = ptu.zeros((num_rollouts, max_path_length+1, obs_dim))
    observations[:,0] = ptu.from_numpy(start_states)
    actions = ptu.zeros((num_rollouts, max_path_length, action_dim))
    rewards = ptu.zeros((num_rollouts, max_path_length, 1))
    terminals = ptu.zeros((num_rollouts, max_path_length, 1))
    return observations, actions, rewards, terminals


def _sample_from_model(dynamics_model, state_actions, t):
    return dynamics_model.sample(state_actions)


def _get_prediction(sample_from_model, dynamics_model, states, actions, t, terminal_cutoff=0.5):
    state_actions = torch.cat([states, actions], dim=-1)
    transitions = sample_from_model(dynamics_model, state_actions, t)
    if (transitions != transitions).any():
        print('WARNING: NaN TRANSITIONS IN DYNAMICS MODEL ROLLOUT')
        transitions[transitions != transitions] = 0

    rewards = transitions[:,:1]
    dones = (transitions[:,1:2] > terminal_cutoff).float()
    delta_obs = transitions[:,2:]

    return rewards, dones, delta_obs


def _create_paths(observations, actions, rewards, terminals, max_path_length):
    observations_np = ptu.get_numpy(observations)
    actions_np = ptu.get_numpy(actions)
    rewards_np = ptu.get_numpy(rewards)
    terminals_np = ptu.get_numpy(terminals)

    paths = []
    for i in range(len(observations)):
        rollout_len = 1
        while rollout_len < max_path_length and terminals[i,rollout_len-1,0] < 0.5:  # just check 0 or 1
            rollout_len += 1
        paths.append(dict(
            observations=observations_np[i, :rollout_len],
            actions=actions_np[i, :rollout_len],
            rewards=rewards_np[i, :rollout_len],
            next_observations=observations_np[i, 1:rollout_len + 1],
            terminals=terminals_np[i, :rollout_len],
            agent_infos=[[] for _ in range(rollout_len)],
            env_infos=[[] for _ in range(rollout_len)],
        ))
    return paths


"""
Methods for generating actions from states
"""


def _get_policy_actions(states, t, action_kwargs):
    policy = action_kwargs['policy']
    actions, *_ = policy.forward(states)
    return actions


def _get_policy_latent_actions(states, t, action_kwargs):
    latents = action_kwargs['latents']
    state_latents = torch.cat([states, latents], dim=-1)
    return _get_policy_actions(state_latents, t, action_kwargs)


def _get_policy_latent_prior_actions(states, t, action_kwargs):
    latent_prior = action_kwargs['latent_prior']
    latents, *_ = latent_prior(states)
    state_latents = torch.cat([states, latents], dim=-1)
    return _get_policy_actions(state_latents, t, action_kwargs)


def _get_open_loop_actions(states, t, action_kwargs):
    actions = action_kwargs['actions']
    return actions[:,t]


"""
Base classes for doing rollout work
TODO: support recurrent dynamics models?
"""


def _model_rollout(
        dynamics_model,                             # torch dynamics model: (s, a) --> (r, d, s')
        start_states,                               # numpy array of states: (num_rollouts, obs_dim)
        get_action,                                 # method for getting action
        action_kwargs=None,                         # kwargs for get_action (ex. policy or actions)
        max_path_length=1000,                       # maximum rollout length (if not terminated)
        terminal_cutoff=0.5,                        # output Done if model pred > terminal_cutoff
        create_full_tensors=_create_full_tensors,
        sample_from_model=_sample_from_model,
        get_prediction=_get_prediction,
        create_paths=_create_paths,
        *args,
        **kwargs,
):
    if action_kwargs is None:
        action_kwargs = dict()
    if terminal_cutoff is None:
        terminal_cutoff = 1e6
    if max_path_length is None:
        raise ValueError('Must specify max_path_length in rollout function')

    obs_dim = dynamics_model.obs_dim
    action_dim = dynamics_model.action_dim

    s, a, r, d = create_full_tensors(start_states, max_path_length, obs_dim, action_dim)
    for t in range(max_path_length):
        a[:,t] = get_action(s[:,t], t, action_kwargs)
        r[:,t], d[:,t], delta_t = get_prediction(
            sample_from_model,
            dynamics_model,
            s[:,t], a[:,t], t,
            terminal_cutoff=terminal_cutoff,
        )
        s[:,t+1] = s[:,t] + delta_t

    paths = create_paths(s, a, r, d, max_path_length)

    return paths


# TODO: _model_online_rollout


# TODO: _model_np_rollout


"""
Interface for rollout functions for other classes to use
"""


def policy(dynamics_model, policy, start_states, **kwargs):
    return _model_rollout(
        dynamics_model,
        start_states,
        _get_policy_actions,
        action_kwargs=dict(policy=policy),
        **kwargs,
    )


def open_loop_actions(dynamics_model, actions, start_states, **kwargs):
    return _model_rollout(
        dynamics_model,
        start_states,
        _get_open_loop_actions,
        action_kwargs=dict(actions=actions),
        **kwargs,
    )


def policy_latent(dynamics_model, policy, start_states, latents, **kwargs):
    return _model_rollout(
        dynamics_model,
        start_states,
        _get_policy_latent_actions,
        action_kwargs=dict(policy=policy, latents=latents),
        **kwargs,
    )


def policy_latent_prior(dynamics_model, policy, latent_prior, start_states, **kwargs):
    return _model_rollout(
        dynamics_model,
        start_states,
        _get_policy_latent_prior_actions,
        action_kwargs=dict(policy=policy, latent_prior=latent_prior),
        **kwargs,
    )


def _rollout_with_disagreement(base_rollout_func, *args, **kwargs):
    disagreement_type = kwargs.get('disagreement_type', 'mean')

    disagreements = []

    def sample_with_disagreement(dynamics_model, state_actions, t):
        # note that disagreement has shape (num_rollouts, 1), e.g. it is unsqueezed
        transitions, disagreement = dynamics_model.sample_with_disagreement(
            state_actions, disagreement_type=disagreement_type)
        disagreements.append(disagreement)
        return transitions

    paths = base_rollout_func(sample_from_model=sample_with_disagreement, *args, **kwargs)
    disagreements = torch.cat(disagreements, dim=-1)

    return paths, disagreements


def policy_with_disagreement(*args, **kwargs):
    return _rollout_with_disagreement(policy, *args, **kwargs)


def policy_latent_with_disagreement(*args, **kwargs):
    return _rollout_with_disagreement(policy_latent, *args, **kwargs)


def policy_latent_prior_with_disagreement(*args, **kwargs):
    return _rollout_with_disagreement(policy_latent_prior, *args, **kwargs)


def open_loop_with_disagreement(*args, **kwargs):
    return _rollout_with_disagreement(open_loop_actions, *args, **kwargs)
