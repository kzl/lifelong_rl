import torch

from lifelong_rl.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
from lifelong_rl.policies.models.gaussian_policy import TanhGaussianPolicy
from lifelong_rl.policies.mpc.mpc import MPCPolicy
from lifelong_rl.models.networks import FlattenMlp
from lifelong_rl.trainers.mbrl.mbrl import MBRLTrainer
from lifelong_rl.trainers.mpc.mpc_trainer import MPPITrainer
from lifelong_rl.trainers.multi_trainer import MultiTrainer
from lifelong_rl.trainers.q_learning.sac import SACTrainer
import lifelong_rl.util.pythonplusplus as ppp


def value_func(obs, critic_policy=None, qf1=None, qf2=None):
    actions, *_ = critic_policy(obs)
    sa = torch.cat([obs, actions], dim=-1)
    q1, q2 = qf1(sa), qf2(sa)
    min_q = torch.min(q1, q2)
    return min_q


def get_config(
        variant,
        expl_env,
        eval_env,
        obs_dim,
        action_dim,
        replay_buffer,
):
    """
    Set up terminal value function
    """

    M = variant['policy_kwargs']['layer_size']

    critic_policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )

    qf1, qf2, target_qf1, target_qf2 = ppp.group_init(
        4,
        FlattenMlp,
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )

    critic_policy_trainer = SACTrainer(
        env=expl_env,
        policy=critic_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['policy_trainer_kwargs'],
    )

    """
    Set up dynamics model
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
    Set up MPC
    """

    policy = MPCPolicy(
        env=expl_env,
        dynamics_model=dynamics_model,
        plan_dim=action_dim,
        value_func=value_func,
        value_func_kwargs=dict(
            critic_policy=critic_policy,
            qf1=qf1,
            qf2=qf2,
        ),
        **variant['mpc_kwargs'],
    )
    trainer = MPPITrainer(
        policy=policy,
    )

    trainer = MultiTrainer(
        trainers=[trainer, critic_policy_trainer],
        trainer_steps=[1, 1],
        trainer_names=['mpc_trainer', 'sac_trainer'],
    )

    config = dict()
    config.update(dict(
        trainer=trainer,
        model_trainer=model_trainer,
        exploration_policy=policy,
        evaluation_policy=critic_policy,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        replay_buffer=replay_buffer,
        dynamics_model=dynamics_model,
    ))
    config['algorithm_kwargs'] = variant.get('algorithm_kwargs', dict())

    return config
