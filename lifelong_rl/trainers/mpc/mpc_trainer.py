import numpy as np
import torch

from collections import OrderedDict

from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
import lifelong_rl.torch.pytorch_util as ptu


class MPPITrainer(TorchTrainer):

    """
    Just a placeholder trainer since MPC does not require training
    """

    def __init__(
            self,
            policy,
    ):
        super().__init__()

        self.policy = policy

        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True

    def train_from_torch(self, batch):
        return

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self.policy.end_epoch(epoch)
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.policy.dynamics_model]

    def get_snapshot(self):
        return dict(
            dynamics_model=self.policy.dynamics_model,
        )
