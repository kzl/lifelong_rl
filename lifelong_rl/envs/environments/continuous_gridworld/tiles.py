import numpy as np

import random


class Tile:

    """
    The grid is represented as a 2D array of Tile instances. The index into the
    array of a tile object is stored as the tuple index_in_grid. The corresponding
    Cartesian coordinates in [-1, 1] are given by position_in_grid. The inclusion
    of certain Tile classes may increase the observation dimension of the agent;
    this is represented by agent_info_dim.

    For now, there are two main types of tiles: Space (which can be passed through),
    and Wall (which cannot).
    """

    def __init__(self, index_in_grid, position_in_grid, *args, **kwargs):
        self.index_in_grid = index_in_grid
        self.position_in_grid = position_in_grid
        self.unique_info_dim = 0
        self.class_info_dim = 0

    def reset(self, agent_infos):
        agent_infos['env_infos'] = dict()

    def get_obs(self, state_infos, agent_infos):
        raise NotImplementedError

    def get_reward(self, state_infos, agent_infos):
        raise NotImplementedError

    def get_speed(self, state_infos, agent_infos):
        raise NotImplementedError

    def can_pass_through(self, state_infos, agent_infos):
        raise NotImplementedError

    def update_agent_infos(self, state_infos, agent_infos):
        raise NotImplementedError

    def get_plot_color(self, state_infos, agent_infos):
        raise NotImplementedError

    def agent_is_here(self, state_infos):
        agent_inds = state_infos['agent_inds']
        return agent_inds == self.index_in_grid


"""
Basic tile types
"""


class Space(Tile):

    def reset(self, agent_infos):
        super().reset(agent_infos)

    def get_obs(self, state_infos, agent_infos):
        return []

    def get_reward(self, state_infos, agent_infos):
        return 0

    def get_speed(self, state_infos, agent_infos):
        return 1

    def can_pass_through(self, state_infos, agent_infos):
        return True

    def update_agent_infos(self, state_infos, agent_infos):
        return

    def get_plot_color(self, state_infos, agent_infos):
        return np.array([1, 1, 1])


class Wall(Tile):

    def reset(self, agent_infos):
        super().reset(agent_infos)

    def get_obs(self, state_infos, agent_infos):
        return []

    def get_reward(self, state_infos, agent_infos):
        return 0

    def get_speed(self, state_infos, agent_infos):
        return 0

    def can_pass_through(self, state_infos, agent_infos):
        return False

    def update_agent_infos(self, state_infos, agent_infos):
        return

    def get_plot_color(self, state_infos, agent_infos):
        return np.array([0, 0, 0])


"""
Types of spaces
"""


class Goal(Space):

    def __init__(self, dense_reward=True, reward_scale=3, incl_in_env_infos=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dense_reward = dense_reward
        self.reward_scale = reward_scale
        self.incl_in_env_infos = incl_in_env_infos
        self.unique_info_dim = 2

    def reset(self, agent_infos):
        super().reset(agent_infos)

    def get_obs(self, state_infos, agent_infos):
        return self.position_in_grid

    def get_reward(self, state_infos, agent_infos):
        agent_position = state_infos['agent_position']
        if self.dense_reward:
            return -np.sum((agent_position - self.position_in_grid) ** 2) * self.reward_scale
        else:
            if agent_position == self.index_in_grid:
                return self.reward_scale
            else:
                return 0

    def update_agent_infos(self, state_infos, agent_infos):
        if self.incl_in_env_infos:
            agent_infos['env_infos']['goal_x'] = self.position_in_grid[0]
            agent_infos['env_infos']['goal_y'] = self.position_in_grid[1]

    def get_plot_color(self, state_infos, agent_infos):
        return np.array([85, 168, 104]) / 255


class Sand(Space):

    def get_speed(self, state_infos, agent_infos):
        return 0.1

    def get_plot_color(self, state_infos, agent_infos):
        return np.array([204, 185, 116]) / 255


class Lava(Space):

    def get_reward(self, state_infos, agent_infos):
        if self.agent_is_here(state_infos):
            return -1
        else:
            return 0

    def get_plot_color(self, state_infos, agent_infos):
        return np.array([140, 24, 0]) / 255


class Hole(Space):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_this_hole = 0

    def get_speed(self, state_infos, agent_infos):
        return 0

    def get_reward(self, state_infos, agent_infos):
        if self.agent_is_here(state_infos):
            return -3
        else:
            return 0

    def get_plot_color(self, state_infos, agent_infos):
        return np.array([147, 120, 95]) / 255

    def update_agent_infos(self, state_infos, agent_infos):
        if 'in_hole' not in agent_infos['env_infos']:
            agent_infos['env_infos']['in_hole'] = 0
        agent_infos['env_infos']['in_hole'] -= self.in_this_hole
        if self.agent_is_here(state_infos):
            self.in_this_hole = 1
        else:
            self.in_this_hole = 0
        agent_infos['env_infos']['in_hole'] += self.in_this_hole


class TrackedSpace(Space):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited_in_episode = False

    def reset(self, agent_infos):
        super().reset(agent_infos)

        agent_infos['tracking_counter'] = 0
        # TODO: is this correct?
        agent_infos['num_tracked_squares'] = agent_infos.get('num_tracked_squares', 0) + 1

        self.visited_in_episode = False

    def update_agent_infos(self, state_infos, agent_infos):
        if self.agent_is_here(state_infos):
            if not self.visited_in_episode:
                self.visited_in_episode = True
                agent_infos['tracking_counter'] += 1

        visitation_pct = agent_infos['tracking_counter'] / agent_infos['num_tracked_squares']
        agent_infos['env_infos']['visitation_pct'] = visitation_pct


"""
Types of walls
"""


# None currently


"""
2D Minecraft tiles

  - These are mostly spaces where something happens if you walk on top of it with
    the correct items. You have up to max_items of each item. You always gain some
    reward for obtaining a new item (specified in MINECRAFT_ITEMS).
    
  - Each tile shows its own position in the observation, which allows for generalization
    when we can randomize the tiles more easily and removes partial observability issues.  
    
  - There *must* exist a single crafting table which does most of the work, like storing
    the inventory in the observation.

"""

# {ITEM_NAME: REWARD_WHEN_OBTAINED}
# Each item gives you a reward when you obtain it (either crafting or mining)
MINECRAFT_ITEMS = {
    'Wood': 1,
    'Stick': 2,
    'Wooden Pickaxe': 4,
    'Stone': 4,
    'Stone Pickaxe': 6,
    'Iron': 10,
    # 'Diamond': 30,
}

# {RESOURCE_NAME: [Color, [ITEMS_TO_BREAK_RESOURCE]]}
# The item is consumed when making the resource
RESOURCES = {
    'Wood': [(102, 84, 52), [None]],
    'Stone': [(113, 113, 113), ['Wooden Pickaxe', 'Stone Pickaxe']],
    'Iron': [(221, 187, 167), ['Stone Pickaxe']],
    'Diamond': [(161, 252, 232), ['Stone Pickaxe']],
}

# [(ITEM_1, NUM_NEEDED), ..., (ITEM_i, NUM_NEEDED), .., ITEM_CRAFTED]
# If all the necessary quantities of all items are in inventory, creates an item
RECIPES = [
    [('Wood', 1), 'Stick'],
    [('Wood', 1), ('Stick', 1), 'Wooden Pickaxe'],
    [('Stone', 1), ('Stick', 1), 'Stone Pickaxe'],
]


class CraftingTable(Space):

    def __init__(self, max_items=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_items = max_items
        self.unique_info_dim = len(MINECRAFT_ITEMS)
        self._reward_this_step = 0

    def reset(self, agent_infos):
        super().reset(agent_infos)

        agent_infos['max_items'] = self.max_items
        agent_infos['inventory'] = {item: 0 for item in MINECRAFT_ITEMS}
        self.place_inventory_in_env_infos(agent_infos)

    def place_inventory_in_env_infos(self, agent_infos):
        inventory = agent_infos['inventory']
        for item in inventory:
            agent_infos['env_infos']['%s Owned' % item] = inventory[item]

    def get_obs(self, state_infos, agent_infos):
        """
        The crafting table shows how much of each item we have
        """

        item_quantities = []
        inventory = agent_infos['inventory']
        for item in inventory:
            item_quantities.append(inventory[item] / self.max_items)
        return np.array(item_quantities)

    def update_agent_infos(self, state_infos, agent_infos):
        """
        Perform all crafting logic based on RECIPES.
        Craft all items we can possibly craft if we're here.
        Note that this is called before the reward calculation.
        """

        # Only craft if we're standing in the crafting table
        if not self.agent_is_here(state_infos):
            self.place_inventory_in_env_infos(agent_infos)
            return

        # Craft until we can no longer craft
        inventory = agent_infos['inventory']
        keep_crafting = True
        while keep_crafting:
            keep_crafting = False

            for recipe in RECIPES:
                resources_needed, item_to_craft = recipe[:-1], recipe[-1]

                # First, check that we have room in inventory to make it
                if inventory[item_to_craft] >= self.max_items:
                    continue

                # See if we have the necessary resources
                can_craft = True
                for resource, quantity in resources_needed:
                    if inventory[resource] < quantity:
                        can_craft = False
                        break

                # If we can't, just proceed to next item
                if can_craft:
                    keep_crafting = True
                else:
                    continue

                # Remove all the resources needed to make item
                for resource, quantity in resources_needed:
                    inventory[resource] -= quantity

                # Add the item to inventory and increase reward
                inventory[item_to_craft] += 1
                self._reward_this_step += MINECRAFT_ITEMS[item_to_craft]

        self.place_inventory_in_env_infos(agent_infos)

    def get_reward(self, state_infos, agent_infos):
        cur_reward = self._reward_this_step
        self._reward_this_step = 0
        return cur_reward

    def get_plot_color(self, state_infos, agent_infos):
        return np.array([172, 102, 56]) / 255


class Resource(Space):

    def __init__(self, resource_name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert resource_name in RESOURCES

        self.unique_info_dim = 2
        self._resource_name = resource_name
        self._reward_this_step = 0

    def get_obs(self, state_infos, agent_infos):
        return self.position_in_grid

    def update_agent_infos(self, state_infos, agent_infos):
        """
        If we are in the tile and have the ability to break it, do it and consume
        the item needed for breaking; we can only mine at most one per timestep
        """

        # We must be in the tile
        if not self.agent_is_here(state_infos):
            return

        inventory = agent_infos['inventory']

        # Check if we already have the max number of this item
        if inventory[self._resource_name] >= agent_infos['max_items']:
            return

        # Check if we have what is needed to break, and consume it
        can_use_to_break = RESOURCES[self._resource_name][1]
        can_break = False

        # This means we can just break with our hand
        if None in can_use_to_break:
            can_break = True

        # Otherwise, check all items
        if not can_break:
            for item in can_use_to_break:
                if inventory[item] >= 1:
                    inventory[item] -= 1
                    can_break = True
                    break

        # Add to our inventory and accumulate reward
        if can_break:
            inventory[self._resource_name] += 1
            self._reward_this_step += MINECRAFT_ITEMS[self._resource_name]

    def get_reward(self, state_infos, agent_infos):
        cur_reward = self._reward_this_step
        self._reward_this_step = 0
        return cur_reward

    def get_plot_color(self, state_infos, agent_infos):
        return np.array(RESOURCES[self._resource_name][0]) / 255


class Wood(Resource):

    def __init__(self, *args, **kwargs):
        super().__init__('Wood', *args, **kwargs)


class Stone(Resource):

    def __init__(self, *args, **kwargs):
        super().__init__('Stone', *args, **kwargs)


class ConsumedResource(Resource):

    def update_agent_infos(self, state_infos, agent_infos):
        super().update_agent_infos(state_infos, agent_infos)

        if self._reward_this_step > 0:
            return

        # Consume all our resource so we can collect more
        agent_infos['inventory'][self._resource_name] = 0


class Iron(ConsumedResource):

    def __init__(self, *args, **kwargs):
        super().__init__('Iron', *args, **kwargs)


class Diamond(ConsumedResource):

    def __init__(self, *args, **kwargs):
        super().__init__('Diamond', *args, **kwargs)
