import abc

import gtimer as gt

from lifelong_rl.core.rl_algorithms.batch.batch_rl_algorithm import BatchRLAlgorithm


class MBBatchRLAlgorithm(BatchRLAlgorithm, metaclass=abc.ABCMeta):

    def __init__(
            self,
            model_trainer,
            model_batch_size,
            model_max_grad_steps=int(1e3),
            model_epochs_since_last_update=5,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.model_trainer = model_trainer
        self.model_batch_size = model_batch_size
        self.model_max_grad_steps = model_max_grad_steps
        self.model_epochs_since_last_update = model_epochs_since_last_update

    def _train(self):
        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

            self._fit_input_stats()

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
            if self.replay_buffer.num_steps_can_sample() > 0:
                self.model_trainer.train_from_buffer(
                    self.replay_buffer,
                    max_grad_steps=self.model_max_grad_steps,
                    epochs_since_last_update=self.model_epochs_since_last_update,
                )
            gt.stamp('model training', unique=False)

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    self.trainer.train_from_paths(new_expl_paths)
                gt.stamp('training', unique=False)
                self.training_mode(False)

            self._fit_input_stats()

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

