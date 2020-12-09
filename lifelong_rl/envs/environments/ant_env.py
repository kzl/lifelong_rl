import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import os


DEFAULT_VEL = 3

DEFAULT_CAMERA_CONFIG = {
    'distance': 10.0,
}


class LifelongAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    """
    Described in Lu et al. 2020.
    """

    def __init__(
            self,
            xml_file='ant.xml',
            gear_ratio=30,
            ctrl_cost_weight=0.01,
            contact_cost_weight=5e-4,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
            healthy_z_range=(0.2, 1.2),
            contact_force_range=(-1.0, 1.0),
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=True,
            target_vel=DEFAULT_VEL,
            height_cost=3,
            target_height=0.7,
            rgb_rendering_tracking=True,
            action_noise=0.,
    ):
        utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)

        self._target_vel = target_vel
        self._target_vel_reward_weight = 1

        self._height_cost = height_cost
        self._target_height = target_height

        self.action_noise = action_noise

        xml_path = 'lifelong_rl/envs/environments/assets/'
        model_path = os.path.abspath(os.path.join(xml_path, xml_file))

        mujoco_env.MujocoEnv.__init__(self, model_path, 5)

    """
    Required for compatibility with lifelong_rl lifelong environment setting
    """

    def get_env_state(self):
        return self.sim.get_state()

    def set_env_state(self, state):
        self.sim.set_state(state)

    """
    =================================================================
    """

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    def set_target_vel(self, vel):
        self._target_vel = vel

    def get_target_vel(self):
        if self._target_vel is not None:
            return self._target_vel
        else:
            return DEFAULT_VEL

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        action += np.random.randn(*action.shape) * self.action_noise
        action = action.clip(-1., 1.)

        xy_position_before = self.get_body_com('torso')[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com('torso')[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        z = self.state_vector()[2]

        rewards = self.get_target_vel()

        vel_cost = abs(x_velocity - self.get_target_vel())
        height_cost = self._height_cost * (z - self._target_height) ** 2
        action_cost = .01 * np.sum(action ** 2)
        costs = vel_cost + height_cost + action_cost

        reward = rewards - costs
        done = not self.is_healthy

        observation = self._get_obs()

        info = {
            'x velocity': x_velocity,
            'target velocity': self.get_target_vel(),
            'z': z,
            'x': self.state_vector()[0],
            'y': self.state_vector()[1],
            'height cost': height_cost,
        }

        return observation, reward, done, info

    def _get_obs(self):
        if self._exclude_current_positions_from_observation:
            return np.concatenate([
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
            ])
        else:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ])

    def get_obs(self):
        return self._get_obs()

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
