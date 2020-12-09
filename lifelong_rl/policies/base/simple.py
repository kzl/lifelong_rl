from lifelong_rl.policies.base.base import Policy


class RandomPolicy(Policy):

    """
    Policy that outputs random actions.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        return self.action_space.sample(), {}
