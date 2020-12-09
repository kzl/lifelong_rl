import torch

from lifelong_rl.models.dynamics_models.probabilistic_ensemble import ProbabilisticEnsemble
from lifelong_rl.policies.base.latent_prior_policy import PriorLatentPolicy
from lifelong_rl.policies.models.gaussian_policy import TanhGaussianPolicy
from lifelong_rl.models.networks import FlattenMlp
from lifelong_rl.trainers.lisp.lisp import LiSPTrainer
from lifelong_rl.trainers.dads.skill_dynamics import SkillDynamics
from lifelong_rl.trainers.mbrl.mbrl import MBRLTrainer
from lifelong_rl.trainers.q_learning.sac import SACTrainer
import lifelong_rl.torch.pytorch_util as ptu
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
    latent_dim = variant['policy_kwargs']['latent_dim']
    restrict_dim = variant['discriminator_kwargs']['restrict_input_size']

    control_policy = TanhGaussianPolicy(
        obs_dim=obs_dim + latent_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
        restrict_obs_dim=restrict_dim,
    )

    prior = torch.distributions.uniform.Uniform(
        -ptu.ones(latent_dim), ptu.ones(latent_dim),
    )

    policy = PriorLatentPolicy(
        policy=control_policy,
        prior=prior,
        unconditional=True,
    )

    qf1, qf2, target_qf1, target_qf2 = ppp.group_init(
        4,
        FlattenMlp,
        input_size=obs_dim + latent_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )

    """
    Learned skill-practice distribution
    """

    skill_practice_dist = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=latent_dim,
        hidden_sizes=[M, M],
    )

    prior_qf1, prior_qf2, prior_target_qf1, prior_target_qf2 = ppp.group_init(
        4,
        FlattenMlp,
        input_size=obs_dim + latent_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )

    skill_practice_trainer = SACTrainer(
        env=expl_env,
        policy=skill_practice_dist,
        qf1=prior_qf1,
        qf2=prior_qf2,
        target_qf1=prior_target_qf1,
        target_qf2=prior_target_qf2,
        **variant['skill_practice_trainer_kwargs'],
    )

    """
    Discriminator
    """

    discrim_kwargs = variant['discriminator_kwargs']
    discriminator = SkillDynamics(
        observation_size=obs_dim if restrict_dim == 0 else restrict_dim,
        action_size=action_dim,
        latent_size=latent_dim,
        normalize_observations=True,
        fix_variance=True,
        fc_layer_params=[discrim_kwargs['layer_size']] * discrim_kwargs['num_layers'],
        # restrict_observation=0,  # we handle this outside of skill-dynamics
        # use_latents_as_delta=variant.get('use_latents_as_delta', False),
    )

    """
    Policy trainer
    """

    policy_trainer = SACTrainer(
        env=expl_env,
        policy=control_policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['policy_trainer_kwargs'],
    )

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

    rollout_len_schedule = variant['rollout_len_schedule']

    def rollout_len(train_steps):
        """
        rollout_len_schedule: [a, b, len_a, len_b]
        linearly increase length from len_a -> len_b over epochs a -> b
        """
        epoch = train_steps // 1000
        if epoch < rollout_len_schedule[0]:
            return 1
        elif epoch >= rollout_len_schedule[1]:
            return rollout_len_schedule[3]
        else:
            return int(
                (epoch - rollout_len_schedule[0]) /
                (rollout_len_schedule[1] - rollout_len_schedule[0]) *
                (rollout_len_schedule[3] - rollout_len_schedule[2])
            ) + rollout_len_schedule[2]

    """
    Setup of intrinsic control
    """

    trainer = LiSPTrainer(
        skill_practice_dist=skill_practice_dist,
        skill_practice_trainer=skill_practice_trainer,
        dynamics_model=dynamics_model,
        rollout_len_func=rollout_len,
        control_policy=control_policy,
        discriminator=discriminator,
        replay_buffer=replay_buffer,
        replay_size=variant['generated_replay_buffer_size'],
        policy_trainer=policy_trainer,
        restrict_input_size=restrict_dim,
        **variant['trainer_kwargs'],
    )

    """
    Create config dict
    """

    config = dict()
    config.update(dict(
        trainer=trainer,
        model_trainer=model_trainer,
        exploration_policy=policy,
        evaluation_policy=policy,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        replay_buffer=replay_buffer,
        dynamics_model=dynamics_model,
        prior=prior,
        learned_prior=skill_practice_dist,
        skill_practice_trainer=skill_practice_trainer,
        control_policy=control_policy,
        latent_dim=latent_dim,
        policy_trainer=policy_trainer,
        rollout_len_func=rollout_len,
    ))
    config['algorithm_kwargs'] = variant.get('algorithm_kwargs', dict())
    config['offline_kwargs'] = variant.get('offline_kwargs', dict())

    return config
