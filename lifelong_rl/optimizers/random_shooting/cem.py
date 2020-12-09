import numpy as np

from lifelong_rl.optimizers.random_shooting.rs_optimizer import RSOptimizer


class CEMOptimizer(RSOptimizer):

    def __init__(
            self,
            sol_dim,
            num_iters,
            population_size,
            elites_frac,
            cost_function,
            upper_bound=1,
            lower_bound=-1,
            epsilon=1e-3,
            polyak=0.2,
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
        )

        self.elites_frac = max(min(elites_frac, 1), .01)

    def update_sol(self, costs, samples, noise, init_mean, init_var):
        elites = samples[np.argsort(costs)][:int(self.elites_frac * self.population_size)]
        updated_mean = np.mean(elites, axis=0)
        updated_var = np.var(elites, axis=0)
        return updated_mean, updated_var
