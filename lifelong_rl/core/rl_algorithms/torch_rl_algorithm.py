from torch import nn as nn
import wandb

import abc
from collections import OrderedDict
from typing import Iterable

from lifelong_rl.core.rl_algorithms.batch.batch_rl_algorithm import BatchRLAlgorithm
from lifelong_rl.core.rl_algorithms.batch.mb_batch_rl_algorithm import MBBatchRLAlgorithm
from lifelong_rl.core.rl_algorithms.offline.offline_rl_algorithm import OfflineRLAlgorithm
from lifelong_rl.core.rl_algorithms.offline.mb_offline_rl_algorithm import OfflineMBRLAlgorithm
from lifelong_rl.core.rl_algorithms.online.online_rl_algorithm import OnlineRLAlgorithm
from lifelong_rl.core.rl_algorithms.online.mbrl_algorithm import MBRLAlgorithm
from lifelong_rl.trainers.trainer import Trainer
from lifelong_rl.torch.pytorch_util import np_to_pytorch_batch


class TorchOnlineRLAlgorithm(OnlineRLAlgorithm):
    def configure_logging(self):
        for net in set(self.trainer.networks):
            wandb.watch(net)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchBatchRLAlgorithm(BatchRLAlgorithm):
    def configure_logging(self):
        for net in set(self.trainer.networks):
            wandb.watch(net)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchMBRLAlgorithm(MBRLAlgorithm):
    def configure_logging(self):
        for net in set(self.trainer.networks + self.model_trainer.networks):
            wandb.watch(net)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
        for net in self.model_trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
        for net in self.model_trainer.networks:
            net.train(mode)


class TorchMBBatchRLAlgorithm(MBBatchRLAlgorithm):
    def configure_logging(self):
        for net in set(self.trainer.networks + self.model_trainer.networks):
            wandb.watch(net)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
        for net in self.model_trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
        for net in self.model_trainer.networks:
            net.train(mode)


class TorchOfflineRLAlgorithm(OfflineRLAlgorithm):
    def configure_logging(self):
        for net in set(self.trainer.networks):
            wandb.watch(net)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)


class TorchOfflineMBRLAlgorithm(OfflineMBRLAlgorithm):
    def configure_logging(self):
        for net in set(self.trainer.networks + self.model_trainer.networks):
            wandb.watch(net)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)
        for net in self.model_trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
        for net in self.model_trainer.networks:
            net.train(mode)


class TorchTrainer(Trainer, metaclass=abc.ABCMeta):
    def __init__(self):
        self._num_train_steps = 0

    def train(self, np_batch):
        self._num_train_steps += 1
        batch = np_to_pytorch_batch(np_batch)
        self.train_from_torch(batch)

    def get_diagnostics(self):
        return OrderedDict([
            ('num train calls', self._num_train_steps),
        ])

    def train_from_torch(self, batch):
        pass

    @property
    def networks(self) -> Iterable[nn.Module]:
        pass
