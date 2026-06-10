import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 250,
                'F': 0.8,
                'CR': 0.9,
                'p': 0.3,
                'beta': 0.5,
                'archive_size': 200,
                'epsilon': 1e-10,
                'mutation_strategy': 'rand/2/bin',
                'local_search_frequency': 20,
                'gamma': 0.1,
                'adaptive_F': True,
                'F_init': 0.5,
                'F_end': 0.8,
                'adaptive_CR': True,
                'CR_init': 0.5,
                'CR_end': 0.9
            }
        else:
            self.params = params
        self.func = func
        self.dim = dim
        self.bounds = bounds
        self.max_evals = max_evals
        self.archive = []

    def local_search(self, best, best_fitness, evals):
        noise = self.params['gamma'] * (np.random.rand(self.dim) - 0.5) * (self.bounds[1] - self.bounds[0])
        candidate = best + self.params['beta'] * noise
        candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
        candidate_fitness = self.func(candidate)
        evals += 1
        if candidate_fitness < best_fitness:
            return candidate, candidate_fitness, evals
        return best, best_fitness, evals

    def update_archive(self, individual):
        if len(self.archive) < self.params['archive_size']:
            self.archive.append(individual)
        else:
            idx = np.random.randint(0, self.params['archive_size'])
            self.archive[idx] = individual

    def adaptive_parameters(self, gen, max_gen):
        if self.params['adaptive_F']:
            self.params['F'] = self.params['F_init'] + (self.params['F_end'] - self.params['F_init']) * (gen / max_gen)
        if self.params['adaptive_CR']:
            self.params['CR'] = self.params['CR_init'] + (self.params['CR_end'] - self.params['CR_init']) * (gen / max_gen)

    def run(self):
        pop_size = self.params.get('pop_size', 250)
        max_gen = self.max_evals // pop_size

        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = np.asarray([self.func(ind) for ind in pop])
        evals = pop_size

        best_index = np.argmin(fitness)
        best = pop[best_index]
        best_fitness = fitness[best_index]

        for gen in range(max_gen):
            self.adaptive_parameters(gen, max_gen)
            F = self.params['F']
            Cr = self.params['CR']

            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                a, b, c, d, e = pop[np.random.choice(idxs, 5, replace=False)]

                if self.params['mutation_strategy'] == 'rand/2/bin':
                    mutant = np.clip(a + F * (b - c) + F * (d - e), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) <= Cr
                if not np.any(cross_points):
                    cross_points[int(np.random.randint(0, self.dim))] = True
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = self.func(trial)
                evals += 1
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    self.update_archive(trial)
                    if trial_fitness < best_fitness:
                        best = trial
                        best_fitness = trial_fitness

                if evals % self.params['local_search_frequency'] == 0:
                    best, best_fitness, evals = self.local_search(best, best_fitness, evals)

                if evals >= self.max_evals or abs(best_fitness - self.params['epsilon']) < self.params['epsilon']:
                    break

        return best_fitness

