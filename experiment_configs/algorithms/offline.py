from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchOfflineRLAlgorithm


def get_offline_algorithm(config, eval_path_collector):

    algorithm = TorchOfflineRLAlgorithm(
        trainer=config['trainer'],
        evaluation_policy=config['evaluation_policy'],
        evaluation_env=config['evaluation_env'],
        replay_buffer=config['replay_buffer'],
        evaluation_data_collector=eval_path_collector,
        **config['offline_kwargs']
    )

    return algorithm
