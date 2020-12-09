import numpy as np
import torch

import gtimer as gt
import os
import random

from experiment_utils.teacher_data import add_transitions
from lifelong_rl.core.logging.logging import logger
from lifelong_rl.core.logging.logging_setup import setup_logger
from lifelong_rl.data_management.replay_buffers.env_replay_buffer import EnvReplayBuffer
from lifelong_rl.data_management.replay_buffers.mujoco_replay_buffer import MujocoReplayBuffer
from lifelong_rl.envs.env_processor import make_env
from lifelong_rl.envs.wrappers import ContinualLifelongEnv, FollowerEnv
import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.envs.env_utils import get_dim
from lifelong_rl.samplers.data_collector.path_collector import MdpPathCollector, LatentPathCollector
from lifelong_rl.samplers.data_collector.step_collector import MdpStepCollector, RFCollector, \
    GoalConditionedReplayStepCollector


def experiment(
        experiment_config,
        exp_prefix,
        variant,
        gpu_kwargs=None,
        log_to_wandb=False,
):

    """
    Reset timers
    (Useful if running multiple seeds from same command)
    """

    gt.reset()
    gt.start()

    """
    Setup logging
    """

    seed = variant['seed']
    setup_logger(exp_prefix, variant=variant, seed=seed, log_to_wandb=log_to_wandb)
    output_csv = logger.get_tabular_output()

    """
    Set GPU mode for pytorch (+ possible other things later)
    """

    if gpu_kwargs is None:
        gpu_kwargs = {'mode': False}
    ptu.set_gpu_mode(**gpu_kwargs)

    """
    Set experiment seeds
    """

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    """
    Environment setup
    """

    envs_list = variant.get('envs_list', None)

    if envs_list is None:
        expl_env, env_infos = make_env(variant['env_name'], **variant.get('env_kwargs', {}))

    else:
        # TODO: not sure if this is tested
        if len(envs_list) == 0:
            raise AttributeError('length of envs_list is zero')
        switch_every = variant['switch_every']
        expl_envs = []
        for env_params in envs_list:
            expl_env, env_infos = make_env(**env_params)
            expl_envs.append(expl_env)
        expl_env = ContinualLifelongEnv(expl_envs[0], switch_every, expl_envs)

    obs_dim = get_dim(expl_env.observation_space)
    action_dim = get_dim(expl_env.action_space)

    if env_infos['mujoco']:
        replay_buffer = MujocoReplayBuffer(variant['replay_buffer_size'], expl_env)
    else:
        replay_buffer = EnvReplayBuffer(variant['replay_buffer_size'], expl_env)

    eval_env = FollowerEnv(expl_env)

    """
    Import any teacher data
    """

    if 'teacher_data_files' in variant:
        for data_file in variant['teacher_data_files']:
            if 'max_teacher_transitions' in variant:
                add_transitions(
                    replay_buffer, data_file, obs_dim, action_dim,
                    max_transitions=variant['max_teacher_transitions'],
                )
            else:
                add_transitions(replay_buffer, data_file, obs_dim, action_dim)

    """
    Experiment-specific configuration
    """

    config = experiment_config['get_config'](
        variant,
        expl_env=expl_env,
        eval_env=eval_env,
        obs_dim=obs_dim,
        action_dim=action_dim,
        replay_buffer=replay_buffer,
    )

    if 'load_config' in experiment_config:
        experiment_config['load_config'](config, variant, gpu_kwargs)

    if 'algorithm_kwargs' not in config:
        config['algorithm_kwargs'] = variant.get('algorithm_kwargs', dict())
    if 'offline_kwargs' not in config:
        config['offline_kwargs'] = variant.get('offline_kwargs', dict())

    """
    Path collectors for sampling from environment
    """

    collector_type = variant.get('collector_type', 'step')
    exploration_policy = config['exploration_policy']
    if collector_type == 'step':
        expl_path_collector = MdpStepCollector(expl_env, exploration_policy)
    elif collector_type == 'batch':
        expl_path_collector = MdpPathCollector(expl_env, exploration_policy)
    elif collector_type == 'batch_latent':
        expl_path_collector = LatentPathCollector(
            sample_latent_every=None,
            env=expl_env,
            policy=exploration_policy,
        )
    elif collector_type == 'rf':
        expl_path_collector = RFCollector(expl_env, exploration_policy)
    else:
        raise NotImplementedError('collector_type of experiment not recognized')

    if collector_type == 'gcr':
        eval_path_collector = GoalConditionedReplayStepCollector(
            eval_env, config['evaluation_policy'], replay_buffer, variant['resample_goal_every'],
        )
    else:
        eval_path_collector = MdpPathCollector(
            eval_env,
            config['evaluation_policy'],
        )

    """
    Finish timer
    """

    gt.stamp('initialization', unique=False)

    """
    Offline RL pretraining
    """

    if 'get_offline_algorithm' in experiment_config and variant.get('do_offline_training', False):
        logger.set_tabular_output(os.path.join(logger.log_dir, 'offline_progress.csv'))

        offline_algorithm = experiment_config['get_offline_algorithm'](
            config,
            eval_path_collector=eval_path_collector,
        )
        offline_algorithm.to(ptu.device)
        offline_algorithm.train()

        logger.set_tabular_output(output_csv)

    """
    Generate algorithm that performs training
    """

    if 'get_algorithm' in experiment_config and variant.get('do_online_training', True):
        algorithm = experiment_config['get_algorithm'](
            config,
            expl_path_collector=expl_path_collector,
            eval_path_collector=eval_path_collector,
        )
        algorithm.to(ptu.device)
        algorithm.train()
