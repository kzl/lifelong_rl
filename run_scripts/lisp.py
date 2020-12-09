from experiment_utils.launch_experiment import launch_experiment

from experiment_configs.algorithms.mbrl import get_algorithm
from experiment_configs.algorithms.offline_mbrl import get_offline_algorithm
from experiment_configs.configs.lisp.lisp_config import get_config
from experiment_configs.configs.mpc.make_mpc_policy import make_get_config


ENV_NAME = 'LifelongHopper'
experiment_kwargs = dict(
    exp_name='lisp-lifelong-hopper',
    num_seeds=1,
    instance_type='c4.4xlarge',
    use_gpu=True,
)


if __name__ == "__main__":
    variant = dict(
        algorithm='LiSP',
        collector_type='rf',  # reset-free exploration environment
        env_name=ENV_NAME,
        env_kwargs=dict(
            terminates=False,
        ),
        do_offline_training=True,  # perform both offline and online training (offline always first)
        do_online_training=True,
        teacher_data_files=['lifelong_hopper_full'],  # see README to download
        replay_buffer_size=int(1e6),
        generated_replay_buffer_size=5000,  # off-policy buffer for policy training
        use_as_eval_policy='uniform',  # sample uniformly from skill policy for evaluation
        policy_kwargs=dict(
            layer_size=256,
            latent_dim=4,
        ),
        discriminator_kwargs=dict(
            layer_size=512,
            num_layers=2,
            restrict_input_size=0,
        ),
        rollout_len_schedule=[-1, -1, 1, 1],
        trainer_kwargs=dict(
            num_model_samples=400,
            num_prior_samples=32,
            num_discrim_updates=4,
            num_policy_updates=8,
            discrim_learning_rate=3e-4,
            policy_batch_size=256,
            reward_bounds=(-30, 30),
            empowerment_horizon=1,
            reward_scale=10,
            disagreement_threshold=.1,
            relabel_rewards=True,
            train_every=10,
            practice_batch_size=256,
            practice_train_steps=4,
            epsilon_greedy=0.2,
        ),
        policy_trainer_kwargs=dict(
            discount=0.99,
            policy_lr=3e-4,
            qf_lr=3e-4,
            soft_target_tau=5e-3,
        ),
        skill_practice_trainer_kwargs=dict(
            discount=0.99,
            policy_lr=3e-4,
            qf_lr=3e-4,
            soft_target_tau=5e-3,
            use_automatic_entropy_tuning=True,
            target_entropy=-4,
        ),
        mppi_kwargs=dict(
            discount=.99,
            horizon=60,
            repeat_length=3,
            plan_every=1,
            temperature=0.01,
            noise_std=1,
            num_rollouts=400,
            num_particles=5,
            planning_iters=10,
            polyak=0.2,
            sampling_mode='ts',
            sampling_kwargs=dict(
                reward_penalty=-20,
                disagreement_threshold=0.1,
            ),
            filter_coefs=(0.05, 0.8, 0),
        ),
        mbrl_kwargs=dict(
            ensemble_size=5,
            num_elites=5,
            layer_size=256,
            learning_rate=1e-3,
            batch_size=256,
        ),
        offline_kwargs=dict(
            num_epochs=2000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=100,
            model_batch_size=256,
            max_path_length=200,
            batch_size=256,
            save_snapshot_freq=1000,
        ),
        algorithm_kwargs=dict(
            num_epochs=10000,
            num_eval_steps_per_epoch=0,
            num_trains_per_train_loop=10,
            num_expl_steps_per_train_loop=10,
            min_num_steps_before_training=0,
            num_model_trains_per_train_loop=1,
            max_path_length=200,
            batch_size=256,
            model_batch_size=256,
            save_snapshot_freq=2500,
        ),
    )

    sweep_values = {
    }

    launch_experiment(
        get_config=make_get_config(get_config),
        get_algorithm=get_algorithm,
        get_offline_algorithm=get_offline_algorithm,
        variant=variant,
        sweep_values=sweep_values,
        **experiment_kwargs
    )
