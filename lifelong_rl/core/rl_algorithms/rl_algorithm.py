import gtimer as gt

import abc
from collections import OrderedDict

from lifelong_rl.core import logger
from lifelong_rl.util import eval_util


def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):

    def __init__(
            self,
            trainer,
            exploration_policy,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            save_snapshot_freq=10,
            post_epoch_funcs=None,
    ):
        self.trainer = trainer
        self.expl_policy = exploration_policy
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0
        self.save_snapshot_freq = save_snapshot_freq

        self.post_epoch_funcs = []
        if post_epoch_funcs is not None:
            self.post_epoch_funcs.extend(post_epoch_funcs)

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _fit_input_stats(self):
        if hasattr(self.trainer, 'fit_input_stats'):
            self.trainer.fit_input_stats(self.replay_buffer)

    def _end_epochs(self, epoch):
        self.expl_data_collector.end_epoch(epoch)
        self.eval_data_collector.end_epoch(epoch)
        self.replay_buffer.end_epoch(epoch)
        self.trainer.end_epoch(epoch)

        if hasattr(self.expl_policy, 'end_epoch'):
            self.expl_policy.end_epoch(epoch)

    def _end_epoch(self, epoch):
        snapshot = self._get_snapshot()
        if self.save_snapshot_freq is not None and \
           (epoch + 1) % self.save_snapshot_freq == 0:
            logger.save_itr_params(epoch, snapshot)
        gt.stamp('saving', unique=False)

        self._log_stats(epoch)

        self._end_epochs(epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(epoch)

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

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
        Exploration
        """
        logger.record_dict(
            self.expl_data_collector.get_diagnostics(),
            prefix='exploration/'
        )
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if len(expl_paths) > 0:
            if hasattr(self.expl_env, 'get_diagnostics'):
                logger.record_dict(
                    self.expl_env.get_diagnostics(expl_paths),
                    prefix='exploration/',
                )
            logger.record_dict(
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/",
            )

        if hasattr(self.expl_policy, 'get_diagnostics'):
            logger.record_dict(
                self.expl_policy.get_diagnostics(),
                prefix='policy/',
            )

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
