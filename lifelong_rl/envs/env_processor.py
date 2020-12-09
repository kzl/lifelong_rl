import gym

from lifelong_rl.envs.wrappers import NormalizedBoxEnv, NonTerminatingEnv, SwapColorEnv


gym.logger.set_level(40)  # stop annoying Box bound precision error


def make_env(env_name, terminates=True, **kwargs):
    env = None
    base_env = None
    env_infos = dict()

    """
    Episodic reinforcement learning
    """
    if env_name == 'HalfCheetah':
        from gym.envs.mujoco import HalfCheetahEnv
        base_env = HalfCheetahEnv
        env_infos['mujoco'] = True
    elif env_name == 'Hopper':
        from gym.envs.mujoco import HopperEnv
        base_env = HopperEnv
        env_infos['mujoco'] = True
    elif env_name == 'InvertedPendulum':
        from gym.envs.mujoco import InvertedPendulumEnv
        base_env = InvertedPendulumEnv
        env_infos['mujoco'] = True
    elif env_name == 'Humanoid':
        from lifelong_rl.envs.environments.humanoid_env import HumanoidTruncatedObsEnv as HumanoidEnv
        from gym.envs.mujoco import HumanoidEnv
        base_env = HumanoidEnv
        env_infos['mujoco'] = True

    """
    Lifelong reinforcement learning
    """
    if env_name == 'LifelongHopper':
        from lifelong_rl.envs.environments.hopper_env import LifelongHopperEnv
        base_env = LifelongHopperEnv
        env_infos['mujoco'] = True
    elif env_name == 'LifelongAnt':
        from lifelong_rl.envs.environments.ant_env import LifelongAntEnv
        base_env = LifelongAntEnv
        env_infos['mujoco'] = True
    elif env_name == 'Gridworld':
        from lifelong_rl.envs.environments.continuous_gridworld.cont_gridworld import ContinuousGridworld
        base_env = ContinuousGridworld
        env_infos['mujoco'] = False
    
    if env is None and base_env is None:
        raise NameError('env_name not recognized')

    if env is None:
        env = base_env(**kwargs)

    if not isinstance(env.action_space, gym.spaces.Discrete):
        env = NormalizedBoxEnv(env)

    if not terminates:
        env = NonTerminatingEnv(env)

    return env, env_infos
