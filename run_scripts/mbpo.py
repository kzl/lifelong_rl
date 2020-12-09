from experiment_utils.launch_experiment import launch_experiment

from experiment_configs.configs.q_learning.mbpo_config import get_config
from experiment_configs.algorithms.mbrl import get_algorithm

ENV_NAME = 'Hopper'
experiment_kwargs = dict(
    exp_name='mbpo-hopper',
    num_seeds=1,
    instance_type='c4.4xlarge',
    use_gpu=True,
)


if __name__ == "__main__":
    variant = dict(
        algorithm='MBPO',
        collector_type='step',
        env_name=ENV_NAME,
        env_kwargs=dict(),
        replay_buffer_size=int(1e6),
        policy_kwargs=dict(
            layer_size=256,
        ),
        trainer_kwargs=dict(
            num_model_rollouts=400,
            rollout_generation_freq=250,
            rollout_len_schedule=[20, 100, 1, 15],  # same format as MBPO codebase
            generated_buffer_size=int(1e5),  # size of synthetic generated replay buffer
            num_policy_updates=20,
            real_data_pct=0.05,
            policy_kwargs=dict(  # kwargs for training the policy (note: inside trainer_kwargs)
                discount=0.99,
                soft_target_tau=5e-3,
                target_update_period=1,
                policy_lr=3e-4,
                qf_lr=3e-4,
            ),
        ),
        mbrl_kwargs=dict(
            ensemble_size=7,
            num_elites=5,
            learning_rate=1e-3,
            batch_size=256,
            hidden_sizes=[256,256,256,256],
        ),
        algorithm_kwargs=dict(
            num_epochs=100,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            num_model_trains_per_train_loop=4,
            max_path_length=1000,
            batch_size=256,
            model_batch_size=256,
            save_snapshot_freq=100,
        ),
    )

    sweep_values = {}

    launch_experiment(
        get_config=get_config,
        get_algorithm=get_algorithm,
        variant=variant,
        sweep_values=sweep_values,
        **experiment_kwargs
    )
