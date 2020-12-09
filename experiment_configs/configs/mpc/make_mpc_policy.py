from lifelong_rl.policies.mpc.policy_mpc import PolicyMPCController


def make_get_config(base_get_config):

    """
    Convert an algorithm that has a skill policy into one which performs MPC
    over the space of skills.
    """

    def get_config(
            variant,
            expl_env,
            eval_env,
            obs_dim,
            action_dim,
            replay_buffer,
    ):
        config = base_get_config(
            variant,
            expl_env,
            eval_env,
            obs_dim,
            action_dim,
            replay_buffer,
        )

        policy = PolicyMPCController(
            env=expl_env,
            dynamics_model=config['dynamics_model'],
            policy=config['control_policy'],
            latent_dim=config['latent_dim'],
            **variant['mppi_kwargs'],
        )

        config['exploration_policy'] = policy

        if variant['use_as_eval_policy'] == 'mppi':
            config['evaluation_policy'] = policy

        return config

    return get_config
