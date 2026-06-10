import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 150,  # Adjusted population size for broader search
                'F_base': 0.5,  # Base value for F
                'F_amp': 0.5,  # Amplitude for F variation
                'CR': 0.9,  # Crossover probability
                'p': 0.2,  # Probability for choosing best individuals
                'archive_size': 200,  # Increased size of the archive
                'mutation_strategy': 'rand-to-best/2',  # Changed mutation strategy
                'adaptive': True,  # Enable adaptive parameters
                'F_adapt': True,  # Enable adaptive F
                'CR_adapt': True,  # Enable adaptive CR
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals

    def run(self):
        pop_size = self.params.get('pop_size', 10 * self.dim)
        F_base = self.params.get('F_base', 0.5)
        F_amp = self.params.get('F_amp', 0.5)
        Cr = self.params.get('CR', 0.7)
        p = self.params.get('p', 0.1)
        archive_size = self.params.get('archive_size', pop_size)
        mutation_strategy = self.params.get('mutation_strategy', 'rand/1')
        adaptive = self.params.get('adaptive', False)
        F_adapt = self.params.get('F_adapt', False)
        CR_adapt = self.params.get('CR_adapt', False)

        # Initialize population
        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = np.asarray([self.func(ind) for ind in pop])
        evals = pop_size
        archive = []

        best_index = np.argmin(fitness)
        best = fitness[best_index]

        while evals < self.max_evals:
            for i in range(pop_size):
                if adaptive:
                    # Adaptive F and Cr
                    F = np.clip(np.random.normal(F_base, F_amp), 0, 1) if F_adapt else F_base + F_amp * np.sin(evals * np.pi / self.max_evals)
                    Cr = np.clip(np.random.normal(0.9, 0.1), 0, 1) if CR_adapt else Cr

                # Mutation
                if mutation_strategy == 'rand-to-best/2':
                    idxs = [idx for idx in range(pop_size) if idx != i]
                    best_idx = np.argmin(fitness)
                    a, b, c, d = pop[np.random.choice(idxs, 4, replace=False)]
                    mutant = np.clip(a + F * (pop[best_idx] - a) + F * (b - c) + F * (c - d), self.bounds[0], self.bounds[1])
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

