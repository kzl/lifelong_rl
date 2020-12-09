import numpy as np

from collections import OrderedDict

from lifelong_rl.optimizers.optimizer import Optimizer


class RSOptimizer(Optimizer):

    def __init__(
            self,
            sol_dim,
            num_iters,
            population_size,
            cost_function,
            upper_bound=1,
            lower_bound=-1,
            epsilon=1e-3,
            polyak=0.2,
            min_var=0.5,
            learn_variance=False,
            filter_noise=None,
    ):
        super().__init__(sol_dim)
        self.num_iters = num_iters
        self.population_size = population_size
        self.cost_function = cost_function
        self.filter_noise = filter_noise

        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.epsilon = epsilon
        self.polyak = polyak
        self.min_var = min_var
        self.learn_variance = learn_variance

    def optimize(self, init_mean, init_var):
        mean, var = init_mean, init_var

        diagnostics = OrderedDict()
        for it in range(self.num_iters):
            noise = np.random.randn(self.population_size, self.sol_dim) * np.sqrt(var)
            if self.filter_noise is not None:
                noise = self.filter_noise(noise)

            samples = mean + noise
            samples = np.minimum(np.maximum(samples, self.lower_bound), self.upper_bound)

            costs = self.cost_function(samples, it)

            # normalization technique: puts costs in [0, 1], so softmax will be over [-1, 0]
            costs[costs != costs] = np.max(costs)
            costs = (costs - np.max(costs)) / (np.max(costs) - np.min(costs) + 1e-6) + 1

            updated_mean, updated_var = self.update_sol(costs, samples, noise, mean, var)

            mean = self.polyak * mean + (1 - self.polyak) * updated_mean
            if self.learn_variance:
                var = self.polyak * var + (1 - self.polyak) * updated_var
                var = np.maximum(var, self.min_var)

            diagnostics['Iteration %d Variance Mean' % it] = np.mean(var)
            diagnostics['Iteration %d Variance Std' % it] = np.std(var)

        return mean, diagnostics

    def update_sol(self, costs, samples, noise, init_mean, init_var):
        return samples[np.argmin(costs)], init_var
