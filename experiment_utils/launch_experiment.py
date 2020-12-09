import argparse
from datetime import datetime
import itertools
import multiprocessing
import os
import random
import sys
import time

import doodad as dd
import doodad.mount as mount
from doodad.easy_sweep.hyper_sweep import run_sweep_doodad, Sweeper
import doodad.easy_sweep.launcher as launcher

from experiment_utils import config
from experiment_utils.sweeper import generate_variants
from experiment_utils.utils import query_yes_no
from experiment_configs.base_experiment import experiment as run_experiment


def launch_experiment(
        exp_name,
        variant,
        sweep_values=None,
        num_seeds=1,
        get_confirmation=True,

        # arguments specifying where the code to run the experiment is
        experiment_class=None,
        get_config=None,
        get_algorithm=None,
        get_offline_algorithm=None,
        load_config=None,

        # misc arguments
        instance_type='c4.2xlarge',
        use_gpu=False,
        include_date=True,
):

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='local',
                        help='Mode for running the experiments - local: runs on local machine, '
                             'ec2: runs on AWS ec2 cluster (requires a proper configuration file)')

    parser.add_argument('--gpu_id', '-id', type=int, default=0,
                        help='GPU id for running experiments (if using single GPU)')

    parser.add_argument('--num_gpu', '-g', type=int, default=3,
                        help='Number of GPUs to use for running the experiments')

    parser.add_argument('--exps_per_gpu', '-e', type=int, default=1,
                        help='Number of experiments per GPU simultaneously')

    parser.add_argument('--num_cpu', '-c', type=int, default=multiprocessing.cpu_count(),
                        help='Number of threads to use for running experiments')

    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False,
                        help='Whether or not to log to Weights and Biases')

    args = parser.parse_args(sys.argv[1:])

    """
    Generating experiment from specified functions:
    
    If the user specifies experiment_class, it is assumed that if get_algorithm and/or
        get_offline_algorithm are specified, then they are located there. This is mostly
        just for backwards compatibility.
    Otherwise, get_algorithm and get_offline_algorithm should be fed into launch_experiment,
        which is generally more modular than specifying the class. get_config must be
        specified, either in experiment_class or in the method call.
    load_config is called after the initialization of the config dict, so it can modify any
        values of the dict in place as needed, and must be fed directly.
    """

    experiment_config = dict()
    if experiment_class is not None:
        experiment_config['get_config'] = experiment_class.get_config
        if hasattr(experiment_class, 'get_algorithm'):
            experiment_config['get_algorithm'] = experiment_class.get_algorithm
        if hasattr(experiment_class, 'get_offline_algorithm'):
            experiment_config['get_offline_algorithm'] = \
                experiment_class.get_offline_algorithm

    if get_config is not None:
        experiment_config['get_config'] = get_config
    if get_algorithm is not None:
        experiment_config['get_algorithm'] = get_algorithm
    if get_offline_algorithm is not None:
        experiment_config['get_offline_algorithm'] = get_offline_algorithm

    if load_config is not None:
        experiment_config['load_config'] = load_config

    if sweep_values is None:
        variants = [variant]
    else:
        variants = generate_variants(variant, sweep_values, num_seeds=num_seeds)

    """
    Setup in the form to feed into the doodad sweeper.
    """

    if include_date:
        timestamp = datetime.now().strftime('%m-%d')
        exp_name = '%s-%s' % (timestamp, exp_name)

    gpu_id = args.gpu_id
    log_to_wandb = args.log_to_wandb
    sweep_params = dict(
        experiment_config=[experiment_config],
        exp_prefix=[exp_name],
        variant=variants,
        gpu_kwargs=[{'mode': use_gpu if args.mode != 'ec2' else False,  # don't use GPU with EC2
                     'gpu_id': gpu_id}],
        log_to_wandb=[log_to_wandb],
    )

    """
    Confirmation
    """

    print('\n')
    print('=' * 50)
    print('Launching experiment: %s' % exp_name)
    print('num variants: %d, num seeds: %d' % (len(variants) // num_seeds, num_seeds))
    print('About to launch %d total experiments' % (len(variants)))
    print('=' * 50)
    for k in sweep_values:
        print('%s:' % k, sweep_values[k])
    print('=' * 50)
    print('\n')

    if get_confirmation and not query_yes_no('Confirm?'):
        return

    """
    Standard run_sweep
    """

    local_mount = mount.MountLocal(local_dir=config.BASE_DIR, pythonpath=True)

    docker_mount_point = os.path.join(config.DOCKER_MOUNT_DIR, exp_name)

    sweeper = launcher.DoodadSweeper([local_mount], docker_img=config.DOCKER_IMAGE,
                                     docker_output_dir=docker_mount_point,
                                     local_output_dir=os.path.join(config.DATA_DIR, 'local', exp_name))

    # it's annoying to have to set up s3 if we don't want to use it
    # TODO: if you want to use S3, uncomment this
    sweeper.mount_out_s3 = None  # mount.MountS3(s3_path='', mount_point=docker_mount_point, output=True)

    if args.mode == 'ec2':
        print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format(exp_name, len(
            list(itertools.product(*[value for value in sweep_params.values()])))))

        if query_yes_no("Continue?"):
            sweeper.run_sweep_ec2(run_experiment, sweep_params, bucket_name=config.S3_BUCKET_NAME,
                                  instance_type=instance_type,
                                  region='us-east-2', s3_log_name=exp_name, add_date_to_logname=False)

    elif args.mode == 'local_docker':
        mode_docker = dd.mode.LocalDocker(
            image=sweeper.image,
        )
        run_sweep_doodad(run_experiment, sweep_params, run_mode=mode_docker,
                         mounts=sweeper.mounts)

    elif args.mode == 'local':
        sweeper.run_sweep_serial(run_experiment, sweep_params)

    elif args.mode == 'local_par':
        sweeper.run_sweep_parallel(run_experiment, sweep_params)

    elif args.mode == 'multi_gpu':
        run_sweep_multi_gpu(run_experiment, sweep_params, num_gpu=args.num_gpu, exps_per_gpu=args.exps_per_gpu)

    else:
        raise NotImplementedError('experiment run mode not recognized')


def run_sweep_multi_gpu(
        run_method,
        params,
        repeat=1,
        num_cpu=multiprocessing.cpu_count(),
        num_gpu=2,
        exps_per_gpu=2
):
    sweeper = Sweeper(params, repeat, include_name=False)
    gpu_frac = 0.9 / exps_per_gpu
    num_runs = num_gpu * exps_per_gpu
    cpu_per_gpu = num_cpu / num_gpu
    exp_args = []
    for config in sweeper:
        exp_args.append((config, run_method))
    random.shuffle(exp_args)
    processes = [None] * num_runs
    run_info = [(i, (i * cpu_per_gpu, (i + 1) * cpu_per_gpu)) for i in range(num_gpu)] * exps_per_gpu
    for kwarg, run in exp_args:
        launched = False
        while not launched:
            for idx in range(num_runs):
                if processes[idx] is None or not processes[idx].is_alive():
                    kwarg['gpu_frac'] = gpu_frac
                    p = multiprocessing.Process(target=run, kwargs=kwarg)
                    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % run_info[idx][0]
                    os.system("taskset -p -c %d-%d %d" % (run_info[idx][1] + (os.getpid(),)))
                    p.start()
                    processes[idx] = p
                    launched = True
                    break
            if not launched:
                time.sleep(10)
