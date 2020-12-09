from experiment_utils.launch_experiment import launch_experiment

from experiment_configs.configs.mpc.mpc import get_config
from experiment_configs.algorithms.mbrl import get_algorithm

ENV_NAME = 'InvertedPendulum'
experiment_kwargs = dict(
    exp_name='mpc-pendulum',
    num_seeds=1,
    instance_type='c4.4xlarge',
    use_gpu=True,
)


if __name__ == "__main__":
    variant = dict(
        algorithm='PETS',  # most closely resembles PETS but can be adjusted with hyperparameters
        collector_type='step',
        env_name=ENV_NAME,
        env_kwargs=dict(),
        replay_buffer_size=int(1e6),
        mpc_kwargs=dict(
            discount=.99,
            horizon=25,
            repeat_length=1,
            plan_every=1,
            temperature=.01,
            noise_std=.5,
            num_rollouts=400,
            num_particles=5,  # this is the num_particles PER ensemble member
            planning_iters=5,
            polyak=0.,
            sampling_mode='ts',  # note that model is written specifically for trajectory sampling
            filter_coefs=(0.2, 0.8, 0),  # smoothing of noise for planning
            predict_terminal=True,
        ),
        mbrl_kwargs=dict(
            ensemble_size=4,
            layer_size=256,
            learning_rate=1e-3,
            batch_size=256,
        ),
        trainer_kwargs=dict(),
        algorithm_kwargs=dict(
            num_epochs=500,
            num_eval_steps_per_epoch=0,
            num_trains_per_train_loop=200,
            num_expl_steps_per_train_loop=200,
            min_num_steps_before_training=200,
            num_model_trains_per_train_loop=1,
            max_path_length=200,
            batch_size=256,
            model_batch_size=256,
            save_snapshot_freq=500,
        ),
    )

    sweep_values = dict()

    launch_experiment(
        get_config=get_config,
        get_algorithm=get_algorithm,
        variant=variant,
        sweep_values=sweep_values,
        **experiment_kwargs
    )
