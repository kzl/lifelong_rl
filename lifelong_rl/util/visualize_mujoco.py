import cv2
import numpy as np

import copy
import os
import time


def visualize_mujoco_from_states(env, sim_states, time_delay=0.008):
    """
    Given the states of the simulator, we can visualize the past Mujoco timesteps.
        - Simulator states are obtained via env.sim.get_state()
    """
    for t in range(len(sim_states)):
        env.sim.set_state(sim_states[t])
        env.sim.forward()
        env.render()
        time.sleep(time_delay)
    env.close()


def mujoco_rgb_from_states(env, sim_states, time_delay=0.008):
    """
    Given the states of the simulator, we can visualize the past Mujoco timesteps.
        - Simulator states are obtained via env.sim.get_state()
    """
    rgb = []
    for t in range(len(sim_states)):
        env.sim.set_state(sim_states[t])
        env.sim.forward()
        rgb.append(env.render(mode='rgb_array'))
        time.sleep(time_delay)
    env.close()
    return rgb


def record_mujoco_video_from_states(env, file_name, sim_states, time_delay=0.008, video_params=None):
    rgb = mujoco_rgb_from_states(env, sim_states, time_delay=0)

    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    if video_params is None:
        video_params = dict(
            size=(rgb[0].shape[0], rgb[0].shape[1]),  # (width, height)
            fourcc=cv2.VideoWriter_fourcc(*'mp4v'),  # use 'XVID' if not mp4
            fps=int(1/time_delay),
        )

    out = cv2.VideoWriter(file_name, video_params['fourcc'], video_params['fps'], video_params['size'])
    for i in range(len(rgb)):
        img = cv2.cvtColor(rgb[i], cv2.COLOR_BGR2RGB)
        out.write(img)
    out.release()
