from experiment_utils.launch_experiment import launch_experiment

from experiment_configs.configs.q_learning.mbpo_config import get_config
from experiment_configs.algorithms.offline_mbrl import get_offline_algorithm

ENV_NAME = 'Hopper'
experiment_kwargs = dict(
    exp_name='morel-hopper-medexp',
    num_seeds=1,
    instance_type='c4.4xlarge',
    use_gpu=True,
)

"""
Note: implementation uses MBPO/SAC as base instead of NPG.
"""

if __name__ == "__main__":
    variant = dict(
        algorithm='MOReL',
        collector_type='rf',
        env_name=ENV_NAME,
        env_kwargs=dict(),
        do_offline_training=True,  # here we specify we want to train offline
        do_online_training=False,
        teacher_data_files=['d4rl-hopper-medium-expert'],  # download this from example script
        replay_buffer_size=int(1e6),
        policy_kwargs=dict(
            layer_size=256,
        ),
        trainer_kwargs=dict(
            num_model_rollouts=400,
            rollout_generation_freq=250,
            rollout_len_schedule=[1, 1, 5, 5],  # Using a constant rollout length of 5
            generated_buffer_size=int(1e5),
            num_policy_updates=20,
            real_data_pct=0.05,
            sampling_mode='mean_disagreement',  # Here we specify the MOReL algorithm (special case of MBPO)
            sampling_kwargs=dict(
                threshold=0.05,
                penalty=100,  # Kappa in paper
            ),
            policy_kwargs=dict(
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
            learning_rate=3e-4,
            batch_size=256,
            hidden_sizes=[256] * 4,
        ),
        offline_kwargs=dict(
            num_epochs=10000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=100,
            model_batch_size=256,
            max_path_length=1000,
            batch_size=256,
            save_snapshot_freq=10000,
        ),
    )

    sweep_values = {}

    launch_experiment(
        get_config=get_config,
        get_offline_algorithm=get_offline_algorithm,
        variant=variant,
        sweep_values=sweep_values,
        **experiment_kwargs
    )
