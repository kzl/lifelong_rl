import numpy as np

from lifelong_rl.optimizers.random_shooting.rs_optimizer import RSOptimizer


class MPPIOptimizer(RSOptimizer):

    def __init__(
            self,
            sol_dim,
            num_iters,
            population_size,
            temperature,
            cost_function,
            upper_bound=1,
            lower_bound=-1,
            epsilon=1e-3,
            polyak=0.2,
            *args,
            **kwargs,
    ):
        super().__init__(
            sol_dim,
            num_iters,
            population_size,
            cost_function,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            epsilon=epsilon,
            polyak=polyak,
            *args,
            **kwargs,
        )

        self.temperature = temperature

    def update_sol(self, costs, samples, noise, init_mean, init_var):
        w = np.exp(-costs / self.temperature)
        w_total = np.sum(w) + 1e-6
        updated_mean = np.sum((w * samples.T).T, axis=0) / w_total
        updated_var = np.sum((w * np.square(noise).T).T, axis=0) / w_total
        return updated_mean, updated_var
