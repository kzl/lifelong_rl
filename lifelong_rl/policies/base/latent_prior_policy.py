import torch

from lifelong_rl.policies.base.base import ExplorationPolicy
import lifelong_rl.torch.pytorch_util as ptu


class PriorLatentPolicy(ExplorationPolicy):

    """
    Policy sampling according to some internal latent.
    TODO: This class needs refactoring.
    """

    def __init__(
            self,
            policy,
            prior,
            unconditional=False,
            steps_between_sampling=100,
    ):
        self.policy = policy
        self.prior = prior
        self.unconditional = unconditional
        self.steps_between_sampling = steps_between_sampling

        self.fixed_latent = False

        self._steps_since_last_sample = 0
        self._last_latent = None

    def set_latent(self, latent):
        self._last_latent = latent

    def get_current_latent(self):
        return ptu.get_numpy(self._last_latent)

    def sample_latent(self, state=None):
        if self.unconditional or state is None:  # this will probably be changed
            latent = self.prior.sample()  # n=1).squeeze(0)
        else:
            latent = self.prior.forward(ptu.from_numpy(state))
        self.set_latent(latent)
        return latent

    def get_action(self, state):
        if (self._steps_since_last_sample >= self.steps_between_sampling or
                self._last_latent is None) and not self.fixed_latent:
            latent = self.sample_latent(state)
            self._steps_since_last_sample = 0
        else:
            latent = self._last_latent
        self._steps_since_last_sample += 1

        state = ptu.from_numpy(state)
        sz = torch.cat((state, latent))
        action, *_ = self.policy.forward(sz)
        return ptu.get_numpy(action), dict()

    def eval(self):
        self.policy.eval()

    def train(self):
        self.policy.train()
