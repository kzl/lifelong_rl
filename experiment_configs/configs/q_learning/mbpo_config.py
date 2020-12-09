from lifelong_rl.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from lifelong_rl.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
from lifelong_rl.models.networks import FlattenMlp
from lifelong_rl.policies.base.base import MakeDeterministic
from lifelong_rl.policies.models.gaussian_policy import TanhGaussianPolicy
from lifelong_rl.trainers.mbrl.mbrl import MBRLTrainer
from lifelong_rl.trainers.q_learning.mbpo import MBPOTrainer
from lifelong_rl.trainers.q_learning.sac import SACTrainer
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
    Setup of soft actor critic (SAC), used as the policy optimization procedure of MBPO
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

    policy_trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']['policy_kwargs']
    )

    """
    Model-based reinforcement learning (MBRL) dynamics models
    """

    dynamics_model = ProbabilisticEnsemble(
        ensemble_size=variant['mbrl_kwargs']['ensemble_size'],
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=variant['mbrl_kwargs']['hidden_sizes'],
    )
    model_trainer = MBRLTrainer(
        ensemble=dynamics_model,
        **variant['mbrl_kwargs'],
    )

    """
    Setup of model-based policy optimization (MBPO)
    """

    generated_replay_buffer = EnvReplayBuffer(
        variant['trainer_kwargs']['generated_buffer_size'],
        expl_env,
    )

    rollout_len_schedule = variant['trainer_kwargs']['rollout_len_schedule']

    def rollout_len(train_steps):
        """
        rollout_len_schedule: [a, b, len_a, len_b]
        Linearly increase length from len_a -> len_b over epochs a -> b
        """
        if 'algorithm_kwargs' in variant:
            epoch = train_steps // variant['algorithm_kwargs']['num_trains_per_train_loop']
        else:
            epoch = 1
        if epoch < rollout_len_schedule[0]:
            return 1
        elif epoch >= rollout_len_schedule[1]:
            return rollout_len_schedule[3]
        else:
            return int(
                (epoch - rollout_len_schedule[0]) / \
                (rollout_len_schedule[1] - rollout_len_schedule[0]) * \
                (rollout_len_schedule[3] - rollout_len_schedule[2])
            ) + 1

    trainer = MBPOTrainer(
        policy_trainer=policy_trainer,
        dynamics_model=dynamics_model,
        replay_buffer=replay_buffer,
        generated_data_buffer=generated_replay_buffer,
        rollout_len_func=rollout_len,
        **variant['trainer_kwargs']
    )

    """
    Create config dict
    """

    config = dict()
    config.update(dict(
        trainer=trainer,
        model_trainer=model_trainer,
        exploration_policy=policy,
        evaluation_policy=MakeDeterministic(policy),
        exploration_env=expl_env,
        evaluation_env=eval_env,
        replay_buffer=replay_buffer,
    ))

    return config
