import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 100,  # Adjusted population size for better exploration
                'F': 0.5,  # Differential weight
                'CR': 0.9,  # Crossover probability
                'p': 0.1,  # Probability for choosing individuals for mutation based on their fitness
                'beta': 0.5  # Learning parameter for local search
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals

    def local_search(self, best, best_fitness, evals):
        candidate = best + self.params['beta'] * (np.random.rand(self.dim) - 0.5) * (self.bounds[1] - self.bounds[0])
        candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
        candidate_fitness = self.func(candidate)
        evals += 1
        if candidate_fitness < best_fitness:
            return candidate, candidate_fitness, evals
        return best, best_fitness, evals

    def run(self):
        pop_size = self.params.get('pop_size', 100)
        F = self.params.get('F', 0.5)
        Cr = self.params.get('CR', 0.9)
        p = self.params.get('p', 0.1)

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = np.asarray([self.func(ind) for ind in pop])
        evals = pop_size

        best_index = np.argmin(fitness)
        best = pop[best_index]
        best_fitness = fitness[best_index]

        while evals < self.max_evals:
            for i in range(pop_size):
                # Enhanced mutation strategy with fitness-based selection
                if np.random.rand() < p:
                    idxs = fitness.argsort()[:max(3, int(pop_size * p))]
                else:
                    idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) <= Cr
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = self.func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

                # Local search around the best solution found so far
                if evals % 100 == 0:
                    best, best_fitness, evals = self.local_search(best, best_fitness, evals)

                if evals >= self.max_evals:
                    break

        return best_fitness

