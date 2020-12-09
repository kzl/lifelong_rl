import gtimer as gt

import abc

from lifelong_rl.core.rl_algorithms.offline.offline_rl_algorithm import OfflineRLAlgorithm


class OfflineMBRLAlgorithm(OfflineRLAlgorithm, metaclass=abc.ABCMeta):

    def __init__(
            self,
            model_trainer,
            model_batch_size,
            model_max_grad_steps=int(1e7),      # The model will train until either this number of grad steps
            model_epochs_since_last_update=10,  # or until holdout loss converged for this number of epochs
            train_at_start=True,                # Flag for debugging
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_trainer = model_trainer
        self.model_batch_size = model_batch_size
        self.model_max_grad_steps = model_max_grad_steps
        self.model_epochs_since_last_update = model_epochs_since_last_update
        self.train_at_start = train_at_start

    def _train(self):
        # Pretrain the model at the beginning of training until convergence
        # Note that convergence is measured against a holdout set of max size 8192
        if self.train_at_start:
            self.model_trainer.train_from_buffer(
                self.replay_buffer,
                max_grad_steps=self.model_max_grad_steps,
                epochs_since_last_update=self.model_epochs_since_last_update,
            )
        gt.stamp('model training', unique=False)

        for epoch in gt.timed_for(
            range(self._start_epoch, self.num_epochs),
            save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            self.training_mode(True)
            for _ in range(self.num_train_loops_per_epoch):
                for t in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                    gt.stamp('policy training', unique=False)
            self.training_mode(False)

            self._end_epoch(epoch)

    def _get_training_diagnostics_dict(self):
        training_diagnostics = super()._get_training_diagnostics_dict()
        training_diagnostics['model_trainer'] = self.model_trainer.get_diagnostics()
        return training_diagnostics

    def _get_snapshot(self):
        snapshot = super()._get_snapshot()
        for k, v in self.model_trainer.get_snapshot().items():
            snapshot['model/' + k] = v
        return snapshot

    def _end_epochs(self, epoch):
        super()._end_epochs(epoch)
        self.model_trainer.end_epoch(epoch)
