from experiment_utils.launch_experiment import launch_experiment

from experiment_configs.configs.dads.dads_config import get_config
from experiment_configs.algorithms.batch import get_algorithm

ENV_NAME = 'Gridworld'
experiment_kwargs = dict(
    exp_name='dads-gridworld',
    num_seeds=1,
    instance_type='c4.4xlarge',
    use_gpu=True,
)


if __name__ == "__main__":
    variant = dict(
        algorithm='DADS',
        collector_type='batch_latent',
        replay_buffer_size=int(1e6),   # for DADS, only used to store past history
        generated_replay_buffer_size=10000,   # off-policy replay buffer helps learning
        env_name=ENV_NAME,
        env_kwargs=dict(
            grid_files=['blank'],  # specifies which file to load for gridworld
            terminates=False,
        ),
        policy_kwargs=dict(
            layer_size=256,
            latent_dim=2,
        ),
        discriminator_kwargs=dict(
            layer_size=512,
            num_layers=2,
            restrict_input_size=0,
        ),
        trainer_kwargs=dict(
            num_prior_samples=512,
            num_discrim_updates=32,
            num_policy_updates=128,
            discrim_learning_rate=3e-4,
            policy_batch_size=256,
            reward_bounds=(-30, 30),
            reward_scale=5,  # increasing reward scale helps learning signal
        ),
        policy_trainer_kwargs=dict(
            discount=0.99,
            policy_lr=3e-4,
            qf_lr=3e-4,
            soft_target_tau=5e-3,
        ),
        algorithm_kwargs=dict(
            num_epochs=100,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1,
            num_expl_steps_per_train_loop=2000,
            min_num_steps_before_training=0,
            max_path_length=100,
            save_snapshot_freq=100,
        ),
    )

    sweep_values = {
    }

    launch_experiment(
        get_config=get_config,
        get_algorithm=get_algorithm,
        variant=variant,
        sweep_values=sweep_values,
        **experiment_kwargs
    )
