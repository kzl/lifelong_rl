from collections import OrderedDict

from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer


class MultiTrainer(TorchTrainer):

    """
    Interface for combining multiple trainers into one trainer.
    """

    def __init__(
            self,
            trainers,               # List of trainers
            trainer_steps,          # List of number of steps to call each trainer per call of MultiTrainer
            trainer_names=None,     # Optionally, specify the names (used for printing/logging)
    ):
        super().__init__()

        assert len(trainers) == len(trainer_steps), 'Must specify number of steps for each trainer'

        self.trainers = trainers
        self.trainer_steps = trainer_steps

        if trainer_names is None:
            self.trainer_names = ['trainer_%d' % i for i in range(1, len(trainers)+1)]
        else:
            self.trainer_names = trainer_names
            while len(self.trainer_names) < len(trainers):
                self.trainer_names.append('trainer_%d' % (len(self.trainer_names)+1))

        self.eval_statistics = OrderedDict()

    def train_from_torch(self, batch):
        for i in range(len(self.trainers)):
            self.trainers[i].train_from_torch(batch)
            for k, v in self.trainers[i].get_diagnostics().items():
                self.eval_statistics[self.trainer_names[i] + '/' + k] = v

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        for trainer in self.trainers:
            trainer.end_epoch(epoch)

    @property
    def networks(self):
        networks = []
        for trainer in self.trainers:
            networks.extend(trainer.networks)
        return networks

    def get_snapshot(self):
        snapshot = dict()
        for i in range(len(self.trainers)):
            for k, v in self.trainers[i].get_diagnostics().items():
                snapshot[self.trainer_names[i] + '/' + k] = v
        return snapshot
