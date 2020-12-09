# lifelong_rl

### Overview
Pytorch implementations of RL algorithms, focusing on model-based, lifelong, reset-free, and offline algorithms.
Official codebase for [Reset-Free Lifelong Learning with Skill-Space Planning](https://sites.google.com/berkeley.edu/reset-free-lifelong-learning).
Originally dervied from [rlkit](https://github.com/vitchyr/rlkit).

#### Status

Project is released but will receive updates periodically.
Contributions, bugs, benchmarking, or other comments are welcome.

#### Algorithms in this codebase
   - Reset-Free RL
     - Lifelong Skill-Space Planning* ([Lu et al. 2020](https://arxiv.org/abs/2012.03548))
   - Model-Based Online RL*
     - Model-Based Policy Optimization ([Janner et al. 2019](https://arxiv.org/abs/1906.08253))
     - Model Predictive Control (ex. [Chua et al. 2018](https://arxiv.org/abs/1805.12114))
     - Learning Off-Policy with Online Planning ([Sikchi et al. 2020](https://arxiv.org/abs/2008.10066))
   - Online Skill Discovery/Multitask RL
     - Dynamics-Aware Discovery of Skills ([Sharma et al. 2019](https://arxiv.org/abs/1907.01657))
     - Hindsight Experience Replay ([Andrychowicz et al. 2017](https://arxiv.org/abs/1707.01495))
   - Offline RL
     - Model-Based Offline RL* ([Kidambi et al. 2020](https://arxiv.org/abs/2005.05951))
     - Model-Based Offline Policy Optimization* ([Yu et al. 2020](https://arxiv.org/abs/2005.13239))
   - Model-Free Online RL
     - Soft Actor Critic ([Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290))
     - Twin Delayed Deep Deterministic Policy Gradient ([Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477))
     - Proximal Policy Optimization ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347))
     - Deep Deterministic Policy Gradient ([Lillicrap et al. 2015](https://arxiv.org/abs/1509.02971))
     - Natural Policy Gradient
     - Vanilla Policy Gradient
   
Note: "Online" here means not offline, i.e. data is being collected in an environment.
"Batch" refers to algorithms that learn from data in batches, ex. PPO (rather than from a replay buffer), not as a synonym for offline RL.

*Reward and terminal functions are learned in this codebase for ease of flexibility, but we also support providing these by hand.

## Usage

### Installation

1. Install Anaconda environment
    ```
    $ conda env create -f environment.yml
    ```
    
    Optionally, also install MuJoCo: see instructions [here](https://github.com/openai/mujoco-py).

2. Install [doodad](https://github.com/jonasrothfuss/doodad) to run experiments (v0.2).

### Running experiments

You can run experiments with:
```
python run_scripts/<script name>.py
```

Use ```-h``` to see more options for running.
Experiments require a ```variant``` dictionary (equivalently to rlkit), which specify a base setting for
each hyperparameter. Additionally, experiments also require a ```sweep_values``` dictionary, which should
only contain the hyperparameters that will be swept over (overwriting the original value in ```variant```).

### Logging experiments

Results from experiments are saved in ```data/```, and a snapshot containing the relevant networks to
evaluate policies offline is stored in ```itr_$n``` every ```save_snapshot_every``` epochs. Data from
the offline training phase is stored in ```offline_itr_$n``` instead.
We support [Viskit](https://github.com/rll/rllab) for plotting or Weights and Biases (include
```-w True``` the call to the run script).

### Visualizing experiments

```scripts/viz_hist.py``` can be used to record a video from a MuJoCo environment using stored data from
the agent's replay buffer, which is modified to additionally store env sim states for MuJoCo environments.
There are also a variety of ways visualization can be done manually.

### Repo structure

- ```agent_data/```
  - Stores ```.pkl``` files of numpy arrays of past transitions
  - Useful for demonstrations, offline data, etc.
  - You can download some example datasets from our link [here](https://drive.google.com/drive/folders/1ctwwWporlw_P5T4-82ozk_bALm7sQY3c?usp=sharing)
- ```data/```
  - Stores logging information and experiment models
  - ```itr_$n``` is the snapshot after epoch ```$n```; similarly ```offline_itr_$n``` is for offline training
- ```experiment_configs/```
  - Experiment configuration files
  - ```get_config``` creates a dictionary consisting of networks and parameters used to initialize a run
  - ```get_offline_algorithm``` and ```get_algorithm``` create an RLAlgorithm from the config
- ```experiment_utils/```
  - Files associated with launching experiments with doodad (should not require modification)
- ```lifelong_rl/```
  - Main codebase
- ```run_scripts/```
  - Scripts to launch experiments: pick config, algorithm, hyperparameters
  - If only both an offline algorithm and algorithm are specified, the offline algorithm is run first
  - Should specify hyperparameters for runs in ```variant```
  - Optionally, perform a grid search over some hyperparameters using```sweep_params```
- ```scripts/```
  - Example utility scripts

### Acknowledgements

This codebase was originally modified from [rlkit](https://github.com/vitchyr/rlkit).
Some parts of the code are taken from [ProMP](https://github.com/jonasrothfuss/ProMP), 
[mjrl](https://github.com/aravindr93/mjrl),
[handful-of-trials-pytorch](https://github.com/quanvuong/handful-of-trials-pytorch), and
[dads](https://github.com/google-research/dads).

### Citation

This is the official codebase for Reset-Free Lifelong Learning with Skill-Space Planning.
Note that the code has been modified since the paper so results may be slightly different.

```
@misc{lu2020resetfree,
      title={Reset-Free Lifelong Learning with Skill-Space Planning}, 
      author={Kevin Lu and Aditya Grover and Pieter Abbeel and Igor Mordatch},
      year={2020},
      eprint={2012.03548},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
