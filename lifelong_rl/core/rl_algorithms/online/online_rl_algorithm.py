import gtimer as gt

import abc

from lifelong_rl.core.rl_algorithms.rl_algorithm import BaseRLAlgorithm


class OnlineRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):

    def __init__(
            self,
            trainer,
            exploration_policy,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            save_snapshot_freq=100,
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
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
            'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'

    def _train(self):
        self.training_mode(False)
        if self.min_num_steps_before_training > 0:
            for _ in range(self.min_num_steps_before_training):
                s, a, r, d, ns, info = self.expl_data_collector.collect_one_step(
                    self.max_path_length,
                    discard_incomplete_paths=False,
                )

                self.replay_buffer.add_sample(s, a, r, d, ns, env_info=info)
                
            self.expl_data_collector.end_epoch(-1)
            gt.stamp('initial exploration', unique=False)

        num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling', unique=False)

            for _ in range(self.num_train_loops_per_epoch):
                for _ in range(self.num_expl_steps_per_train_loop):
                    s, a, r, d, ns, info = self.expl_data_collector.collect_one_step(
                        self.max_path_length,
                        discard_incomplete_paths=False,
                    )
                    gt.stamp('exploration sampling', unique=False)

                    self.replay_buffer.add_sample(s, a, r, d, ns, env_info=info)
                    gt.stamp('data storing', unique=False)

                    self.training_mode(True)
                    for _ in range(num_trains_per_expl_step):
                        train_data = self.replay_buffer.random_batch(
                            self.batch_size)
                        self.trainer.train(train_data)
                    gt.stamp('training', unique=False)
                    self.training_mode(False)

            self._end_epoch(epoch)
