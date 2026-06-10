import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 100,  # Increased population size for better exploration
                'F': 0.5,  # Differential weight
                'CR': 0.9,  # Crossover probability
                'p': 0.1,  # Probability for choosing individuals for mutation from best solutions
                'archive_size': 100,  # Size of the archive
                'mutation_strategy': 'current-to-pbest/1',  # Mutation strategy
            }
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
        p = self.params.get('p', 0.1)
        archive_size = self.params.get('archive_size', pop_size)
        mutation_strategy = self.params.get('mutation_strategy', 'rand/1')

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = np.asarray([self.func(ind) for ind in pop])
        evals = pop_size
        archive = []

        best_index = np.argmin(fitness)
        best = fitness[best_index]

        while evals < self.max_evals:
            for i in range(pop_size):
                # Mutation
                if mutation_strategy == 'current-to-pbest/1':
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    pbest = int(pop_size * p)
                    candidates = np.argsort(fitness)[:pbest]
                    pbest_idx = np.random.choice(candidates)
                    a, b = pop[np.random.choice(idxs, 2, replace=False)]
                    mutant = np.clip(pop[i] + F * (pop[pbest_idx] - pop[i]) + F * (a - b), self.bounds[0], self.bounds[1])
                else:  # Default mutation strategy (rand/1)
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
                    archive.append(pop[i].copy())
                    if len(archive) > archive_size:
                        archive.pop(np.random.randint(0, len(archive)))
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best:
                        best = trial_fitness

                if evals >= self.max_evals:
                    break

        return best

