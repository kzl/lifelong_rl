import gtimer as gt

import abc

from lifelong_rl.core.rl_algorithms.rl_algorithm import BaseRLAlgorithm


class MBRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):

    def __init__(
            self,
            trainer,
            model_trainer,
            exploration_policy,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            batch_size,
            model_batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_model_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            initial_training_steps=0,
            save_snapshot_freq=10,
            post_epoch_funcs=None,
    ):
        super().__init__(
            trainer,
            exploration_policy,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            save_snapshot_freq=save_snapshot_freq,
            post_epoch_funcs=post_epoch_funcs,
        )

        self.model_trainer = model_trainer
        self.batch_size = batch_size
        self.model_batch_size = model_batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_model_trains_per_train_loop = num_model_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.initial_training_steps = initial_training_steps

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

    def _train(self):
        self.training_mode(False)

        if self.min_num_steps_before_training > 0:
            for _ in range(self.min_num_steps_before_training):
                s, a, r, d, ns, info = self.expl_data_collector.collect_one_step(
                    self.max_path_length,
                    discard_incomplete_paths=False,
                    initial_expl=True,
                )

                self.replay_buffer.add_sample(s, a, r, d, ns, env_info=info)
            self.expl_data_collector.end_epoch(-1)

            gt.stamp('initial exploration', unique=False)

        num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop
        if self.num_model_trains_per_train_loop == 0:
            model_train_freq = None
        else:
            model_train_freq = self.num_expl_steps_per_train_loop // self.num_model_trains_per_train_loop

        if self.replay_buffer.num_steps_can_sample() > 0 and model_train_freq is not None:
            self.model_trainer.train_from_buffer(self.replay_buffer, max_grad_steps=100000)
            gt.stamp('model training', unique=False)

        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            if self.num_eval_steps_per_epoch > 0:
                self.eval_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_eval_steps_per_epoch,
                    discard_incomplete_paths=True,
                )
            gt.stamp('evaluation sampling', unique=False)

            for _ in range(self.num_train_loops_per_epoch):
                for t in range(self.num_expl_steps_per_train_loop):
                    self.training_mode(True)
                    if model_train_freq is not None and \
                            ((t+1) % model_train_freq == 0 or \
                            (epoch == 0 and t == 0 and \
                            self.replay_buffer.num_steps_can_sample() > 0)):
                        self.model_trainer.train_from_buffer(self.replay_buffer)
                    gt.stamp('model training', unique=False)

                    if (epoch == 0 and t == 0) and self.initial_training_steps > 0:
                        for _ in range(self.initial_training_steps):
                            train_data = self.replay_buffer.random_batch(
                                self.batch_size)
                            self.trainer.train(train_data)
                        gt.stamp('initial policy training', unique=False)

                    s, a, r, d, ns, info = self.expl_data_collector.collect_one_step(
                        self.max_path_length,
                        discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)

                    self.replay_buffer.add_sample(s, a, r, d, ns, env_info=info)
                    gt.stamp('data storing', unique=False)

                    for _ in range(num_trains_per_expl_step):
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size)
                        self.trainer.train(train_data)
                    gt.stamp('policy training', unique=False)
                    self.training_mode(False)

            self._end_epoch(epoch)
