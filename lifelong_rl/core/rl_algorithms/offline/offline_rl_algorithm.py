import gtimer as gt

import abc

from lifelong_rl.core import logger
from lifelong_rl.core.rl_algorithms.rl_algorithm import _get_epoch_timings
from lifelong_rl.util import eval_util


class OfflineRLAlgorithm(object, metaclass=abc.ABCMeta):

    def __init__(
            self,
            trainer,
            evaluation_policy,
            evaluation_env,
            evaluation_data_collector,
            replay_buffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            save_snapshot_freq=10,
    ):
        self.trainer = trainer
        self.eval_policy = evaluation_policy
        self.eval_env = evaluation_env
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.save_snapshot_freq = save_snapshot_freq

        self._start_epoch = 0
        self.post_epoch_funcs = []

    def _train(self):
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
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
            self.training_mode(False)
            gt.stamp('training')

            self._end_epoch(epoch)

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        if self.save_snapshot_freq is not None and \
                (epoch + 1) % self.save_snapshot_freq == 0:
            logger.save_itr_params(epoch, snapshot, prefix='offline_itr')
        gt.stamp('saving', unique=False)

        self._log_stats(epoch)

        self._end_epochs(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _end_epochs(self, epoch):
        self.eval_data_collector.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        if hasattr(self.eval_policy, 'end_epoch'):
            self.eval_policy.end_epoch(epoch)

    def _get_trainer_diagnostics(self):
        return self.trainer.get_diagnostics()

    def _get_training_diagnostics_dict(self):
        return {'policy_trainer': self._get_trainer_diagnostics()}

    def _log_stats(self, epoch):
        logger.log("Epoch {} finished".format(epoch), with_timestamp=True)

        """
        Replay Buffer
        """
        logger.record_dict(
            self.replay_buffer.get_diagnostics(),
            prefix='replay_buffer/'
        )

        """
        Trainer
        """
        training_diagnostics = self._get_training_diagnostics_dict()
        for prefix in training_diagnostics:
            logger.record_dict(training_diagnostics[prefix], prefix=prefix + '/')

        """
        Evaluation
        """
        if self.num_eval_steps_per_epoch > 0:
            logger.record_dict(
                self.eval_data_collector.get_diagnostics(),
                prefix='evaluation/',
            )
            eval_paths = self.eval_data_collector.get_epoch_paths()
            if hasattr(self.eval_env, 'get_diagnostics'):
                logger.record_dict(
                    self.eval_env.get_diagnostics(eval_paths),
                    prefix='evaluation/',
                )
            logger.record_dict(
                eval_util.get_generic_path_information(eval_paths),
                prefix="evaluation/",
            )

        """
        Misc
        """
        logger.record_dict(_get_epoch_timings())
        logger.record_tabular('Epoch', epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False)
        gt.stamp('logging', unique=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
