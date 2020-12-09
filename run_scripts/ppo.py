from experiment_utils.launch_experiment import launch_experiment

from experiment_configs.configs.pg.ppo_config import get_config
from experiment_configs.algorithms.batch import get_algorithm

ENV_NAME = 'Hopper'
experiment_kwargs = dict(
    exp_name='ppo-hopper',
    num_seeds=1,
    instance_type='c4.4xlarge',
    use_gpu=False,
)


if __name__ == "__main__":
    variant = dict(
        algorithm='PPO',
        collector_type='batch',
        env_name=ENV_NAME,
        env_kwargs=dict(),
        replay_buffer_size=int(1e6),
        policy_kwargs=dict(
            layer_size=64,
        ),
        value_kwargs=dict(
            layer_size=256,
        ),
        policy_trainer_kwargs=dict(
            discount=0.99,
            gae_lambda=0.97,
            ppo_epsilon=0.2,
            policy_lr=3e-4,
            value_lr=3e-4,
            target_kl=None,
            num_epochs=10,
            policy_batch_size=64,
            value_batch_size=64,
            normalize_advantages=True,
        ),
        algorithm_kwargs=dict(
            num_epochs=1000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1,
            num_expl_steps_per_train_loop=2048,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            save_snapshot_freq=1000,
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
