from lifelong_rl.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
from lifelong_rl.policies.mpc.mpc import MPCPolicy
from lifelong_rl.trainers.mbrl.mbrl import MBRLTrainer
from lifelong_rl.trainers.mpc.mpc_trainer import MPPITrainer


def get_config(
        variant,
        expl_env,
        eval_env,
        obs_dim,
        action_dim,
        replay_buffer,
):

    """
    Model-based reinforcement learning (MBRL) dynamics models
    """

    M = variant['mbrl_kwargs']['layer_size']

    dynamics_model = ProbabilisticEnsemble(
        ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M, M],
    )
    model_trainer = MBRLTrainer(
        ensemble=dynamics_model,
        **variant['mbrl_kwargs'],
    )

    """
    Setup of MPPI policies
    """

    policy = MPCPolicy(
        env=expl_env,
        dynamics_model=dynamics_model,
        plan_dim=action_dim,
        **variant['mpc_kwargs'],
    )
    eval_policy = MPCPolicy(
        env=eval_env,
        dynamics_model=dynamics_model,
        plan_dim=action_dim,
        **variant['mpc_kwargs'],
    )
    trainer = MPPITrainer(
        policy=policy,
    )

    """
    Create config dict
    """

    config = dict()
    config.update(dict(
        trainer=trainer,
        model_trainer=model_trainer,
        exploration_policy=policy,
        evaluation_policy=eval_policy,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        replay_buffer=replay_buffer,
    ))
    config['algorithm_kwargs'] = variant['algorithm_kwargs']

    return config
