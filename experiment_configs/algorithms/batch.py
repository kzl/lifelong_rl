from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchBatchRLAlgorithm


def get_algorithm(config, expl_path_collector, eval_path_collector):

    algorithm = TorchBatchRLAlgorithm(
        trainer=config['trainer'],
        exploration_policy=config['exploration_policy'],
        exploration_env=config['exploration_env'],
        evaluation_env=config['evaluation_env'],
        replay_buffer=config['replay_buffer'],
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        **config['algorithm_kwargs']
    )

    return algorithm
