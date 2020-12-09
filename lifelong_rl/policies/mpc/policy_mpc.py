import numpy as np
import torch

from lifelong_rl.policies.mpc.mpc import MPCPolicy


class PolicyMPCController(MPCPolicy):

    """
    Perform MPC planning over a policy that takes in an additional latent.
    """

    def __init__(
            self,
            policy,      # control policy to run that takes in a latent
            latent_dim,  # dimension of the latent to feed the policy
            *args,
            **kwargs
    ):
        super().__init__(plan_dim=latent_dim, *args, **kwargs)
        self.policy = policy

    def convert_plan_to_action(self, obs, plan, deterministic=False):
        action, *_ = self.policy.get_action(
            np.concatenate((obs, plan), axis=-1),
            deterministic=True,
        )
        return action

    def convert_plans_to_actions(self, obs, plans, deterministic=True):
        actions, *_ = self.policy(
            torch.cat((obs, plans), dim=-1),
            deterministic=deterministic,
        )
        return actions
