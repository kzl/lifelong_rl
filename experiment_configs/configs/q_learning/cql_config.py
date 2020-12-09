from lifelong_rl.models.networks import FlattenMlp
from lifelong_rl.policies.base.base import MakeDeterministic
from lifelong_rl.policies.models.gaussian_policy import TanhGaussianPolicy
from lifelong_rl.trainers.q_learning.cql import CQLTrainer
import lifelong_rl.util.pythonplusplus as ppp


def get_config(
        variant,
        expl_env,
        eval_env,
        obs_dim,
        action_dim,
        replay_buffer,
):
    """
    Policy construction
    """

    M = variant['policy_kwargs']['layer_size']

    qf1, qf2, target_qf1, target_qf2 = ppp.group_init(
        4,
        FlattenMlp,
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )

    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs'],
    )

    """
    Create config dict
    """

    config = dict()
    config.update(dict(
        trainer=trainer,
        exploration_policy=policy,
        evaluation_policy=MakeDeterministic(policy),
        exploration_env=expl_env,
        evaluation_env=eval_env,
        replay_buffer=replay_buffer,
    ))

    return config
