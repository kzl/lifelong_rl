import gym
import numpy as np

import copy
import os

import lifelong_rl.envs.environments.continuous_gridworld.tiles as tiles


# Used for converting tokens from .txt file to grid tile classes
CHAR_TO_TILE = {

    # Base set of tiles
    ' ': tiles.Space,
    '#': tiles.Wall,
    'S': tiles.Sand,
    'L': tiles.Lava,
    'G': tiles.Goal,
    'H': tiles.Hole,

    # Minecraft environment tiles
    'C': tiles.CraftingTable,
    'w': tiles.Wood,
    'r': tiles.Stone,  # "r" stands for "rock"
    'i': tiles.Iron,
    'd': tiles.Diamond,

}


class ContinuousGridworld:

    """
    Continuous 2D gridworld environment. The environment consists of a 2D grid of tiles,
    but the agent has a continuous 2D position. We represent each tile as a unique
    instance in an array, where each tile must have relevant methods implemented. This
    allows for greater flexibility and convenience of experimenting.

    The action is an intended dx, and the observation by default includes the position,
    plus any additional values that may be given by a tile.
    """

    def __init__(
            self,
            grid_files,                 # List of grid files that worlds switch between (can be 1)
            switch_grid_every=None,     # How often to switch grids (None: don't switch)
            start_pos=(0.,0.),          # Starting position of agent at beginning of episode
            dt=0.1,                     # Maximum distance of travel for one action
            num_collision_steps=10,     # How often to check for wall collision
            grid_kwargs=None,           # Additional kwargs to give to tiles
            act_noise=0.,               # Apply a clipped Gaussian noise to actions
    ):
        self.grid_files = grid_files
        self.switch_grid_every = switch_grid_every
        self.start_pos = start_pos
        self.dt = dt
        self.act_noise = act_noise
        self.num_collision_steps = num_collision_steps
        self.grid_kwargs = grid_kwargs if grid_kwargs is not None else dict()
        self._grids_dir = os.path.join(os.path.dirname(__file__), 'grids')

        self._grid = []
        self._num_steps_total = 0
        self._cur_grid_ind = 0
        self._x = np.array(start_pos)
        self._agent_infos = dict(env_infos=dict())

        obs_dim = self.parse_grid(self.grid_files[self._cur_grid_ind])
        self.observation_space = gym.spaces.Box(
            -np.ones(obs_dim), np.ones(obs_dim)
        )
        self.action_space = gym.spaces.Box(
            np.array([-1, -1]), np.array([1, 1])
        )

    def step(self, action):
        action += np.random.randn(*action.shape) * self.act_noise
        action = np.clip(action, -1, 1)

        # Perform collision detection: increment x in small steps
        ddt = self.dt / self.num_collision_steps
        for _ in range(self.num_collision_steps):
            # Process any necessary info from current state
            cur_ind = self.get_index(self._x)
            cur_tile = self._grid[cur_ind[1]][cur_ind[0]]
            state_infos = self.get_state_infos()

            # Calculate movement parameters from current tile
            delta_x = 0
            if cur_tile.can_pass_through(state_infos, self._agent_infos):
                speed = cur_tile.get_speed(state_infos, self._agent_infos)
                delta_x += ddt * action * speed
            else:
                break

            # Process any necessary info from next state
            next_x = np.clip(self._x + delta_x, -1, 1)
            next_ind = self.get_index(next_x)
            next_tile = self._grid[next_ind[1]][next_ind[0]]
            next_state_infos = self.get_state_infos(next_x, self._num_steps_total+1)

            # Check to ensure we can reach the next state
            if next_tile.can_pass_through(next_state_infos, self._agent_infos):
                self._x = next_x
            else:
                break

        # Do updates and get transition, which require state_infos
        state_infos = self.get_state_infos()

        # Update agent infos for every tile
        for row in self._grid:
            for tile in row:
                tile.update_agent_infos(state_infos, self._agent_infos)
        self._agent_infos['env_infos']['x'] = self._x[0]
        self._agent_infos['env_infos']['y'] = self._x[1]

        # Get transition
        next_obs = self.get_obs(state_infos=state_infos)
        reward = self.get_reward(state_infos=state_infos)
        done = False
        env_infos = copy.deepcopy(self._agent_infos['env_infos'])

        # Perform updates related to continual learning
        self._num_steps_total += 1
        if self.switch_grid_every is not None and \
                self._num_steps_total % self.switch_grid_every == 0:
            self.advance_grid()

        return next_obs, reward, done, env_infos

    def reset(self):
        self._x = np.array(self.start_pos)
        self._agent_infos = dict()
        for row in self._grid:
            for tile in row:
                tile.reset(self._agent_infos)
        return self.get_obs()

    def get_meta_infos(self):
        return dict(
            grid_file=self.grid_files[self._cur_grid_ind],
            num_steps=self._num_steps_total,
            cur_grid_ind=self._cur_grid_ind,
            x=self._x.copy(),
            agent_infos=copy.deepcopy(self._agent_infos),
        )

    def get_obs(self, state_infos=None):
        if state_infos is None:
            state_infos = self.get_state_infos()
        obs = [self._x]
        for row in self._grid:
            for tile in row:
                obs.append(tile.get_obs(state_infos, self._agent_infos))
        return np.concatenate(obs)

    def get_reward(self, state_infos=None):
        if state_infos is None:
            state_infos = self.get_state_infos()
        reward = 0
        for row in self._grid:
            for tile in row:
                reward += tile.get_reward(state_infos, self._agent_infos)
        reward = min(max(reward, -100), 100)
        return reward

    def get_state_infos(self, x=None, num_steps_total=None):
        """
        state_infos includes information about the current state of the environment.
        """
        if x is None:
            x = self._x
        if num_steps_total is None:
            num_steps_total = self._num_steps_total
        inds = self.get_index(x)
        state_infos = dict(
            num_steps_total=num_steps_total,
            agent_inds=inds,
            agent_position=x,
        )
        return state_infos

    def advance_grid(self):
        """
        Set the grid to the next grid in the sequence. Note that this does not
        induce a reset to the agent.
        """
        self._cur_grid_ind = (self._cur_grid_ind + 1) % len(self.grid_files)
        self.parse_grid(self.grid_files[self._cur_grid_ind])

    def get_position(self, inds, lens=None):
        """
        The position is defined at the center of the tile. We calculate this by
        considering each tile as having two halves.
        """
        if lens is None:
            lens = (len(self._grid[0]), len(self._grid))

        inds, lens = np.array(inds), np.array(lens)
        x = 2 * ((2 * inds + 1) / (2 * lens)) - 1

        return x

    def get_index(self, x, lens=None):
        """
        The the index of the cell that x resides in (e.g. the tile whose center
        is closest to the position of x).
        """
        if lens is None:
            lens = (len(self._grid[0]), len(self._grid))

        lens = np.array(lens)
        ind_vals = 0.5 * (x + 1) * lens
        inds = np.rint(ind_vals).astype(np.int)
        inds[0] = min(inds[0], lens[0]-1)
        inds[1] = min(inds[1], lens[1]-1)

        return inds[0], inds[1]

    def parse_grid(self, grid_file):
        """
        Set the self._grids list and self.observation_space from the file with name
        grid_file. The file should just consist of text characters formatted in a
        2D array corresponding to the relevant characters.
        """

        file_path = os.path.join(self._grids_dir, grid_file)
        rows = []
        with open(file_path + '.txt', 'r') as f:
            row = f.readline()
            while row:
                rows.append(row[:-1])  # don't include newline
                row = f.readline()

        lens = (len(rows[0]), len(rows))

        self._grid = []
        obs_dim = 2
        tile_classes = set()
        x, y = 0, 0
        for row in rows:
            self._grid.append([])
            x = 0
            for char in row:
                index = (x, y)
                tile_class = CHAR_TO_TILE[char]
                tile_classes.add(tile_class)
                tile = tile_class(
                    index_in_grid=index,
                    position_in_grid=self.get_position(index, lens),
                    **self.grid_kwargs,
                )
                self._grid[-1].append(tile)
                if tile_class not in tile_classes:
                    tile_classes.add(tile_class)
                    obs_dim += tile.class_info_dim
                obs_dim += tile.unique_info_dim
                x += 1
            y += 1

        return obs_dim

    def get_rgb_array(self):
        state_infos = self.get_state_infos()
        agent_infos = self._agent_infos

        rgb_array = np.zeros((len(self._grid[0]), len(self._grid), 3))
        for y, row in enumerate(self._grid):
            for x, tile in enumerate(row):
                rgb_array[x,y] = tile.get_plot_color(state_infos, agent_infos)

        return rgb_array
