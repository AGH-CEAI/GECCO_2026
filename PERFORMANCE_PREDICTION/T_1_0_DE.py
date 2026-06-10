import numpy as np

class Algorithm():

    def __init__(self, func, dim, bounds, max_evals, params=None):

        if params is None:
            self.params = {'pop_size': 20, 'F': 0.8, 'CR': 0.9}
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals

    def run(self):
        pop_size = self.params.get('pop_size', 10 * self.dim)
        F = self.params.get('F', 0.5)
        Cr = self.params.get('CR', 0.7)

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = np.asarray([self.func(ind) for ind in pop])
        evals = pop_size
        gen = 0

        best_index = np.argmin(fitness)
        best = fitness[best_index]

        while evals < self.max_evals:
            for i in range(pop_size):
                # Mutation: rand/1
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover: bin
                cross_points = np.random.rand(self.dim) <= Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = self.func(trial)
                evals += 1
                if trial_fitness <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness <= best:
                        best = trial_fitness

                if evals >= self.max_evals:
                    break

            gen += 1

        return best

