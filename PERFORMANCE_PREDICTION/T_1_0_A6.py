import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 300,
                'F_base': 0.5,
                'F_amp': 0.3,
                'CR_base': 0.5,
                'CR_amp': 0.4,
                'p': 0.35,
                'beta': 0.5,
                'archive_size': 250,
                'epsilon': 1e-10,
                'mutation_strategy': 'best/2/bin',
                'local_search_frequency': 15,
                'gamma': 0.1,
                'adaptive_F_CR': True,
                'strategy_variation_frequency': 10
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

    def adaptive_F_CR(self, gen, max_gen):
        cycle = np.sin(2 * np.pi * gen / max_gen)
        self.params['F'] = self.params['F_base'] + self.params['F_amp'] * cycle
        self.params['CR'] = self.params['CR_base'] + self.params['CR_amp'] * cycle

    def switch_mutation_strategy(self, gen):
        if gen % self.params['strategy_variation_frequency'] == 0:
            if self.params['mutation_strategy'] == 'best/2/bin':
                self.params['mutation_strategy'] = 'rand/1/bin'
            else:
                self.params['mutation_strategy'] = 'best/2/bin'

    def run(self):
        pop_size = self.params.get('pop_size', 300)
        max_gen = self.max_evals // pop_size

        pop = np.random.rand(pop_size, self.dim) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        fitness = np.asarray([self.func(ind) for ind in pop])
        evals = pop_size

        best_index = np.argmin(fitness)
        best = pop[best_index]
        best_fitness = fitness[best_index]

        for gen in range(max_gen):
            if self.params['adaptive_F_CR']:
                self.adaptive_F_CR(gen, max_gen)
            F = self.params.get('F', 0.5)
            Cr = self.params.get('CR', 0.5)

            self.switch_mutation_strategy(gen)

            for i in range(pop_size):
                idxs = [idx for idx in range(pop_size) if idx != i]
                if self.params['mutation_strategy'] == 'best/2/bin':
                    a, b, c, d = best, *pop[np.random.choice(idxs, 3, replace=False)]
                else:  # 'rand/1/bin'
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    d = None

                if d is not None:
                    mutant = np.clip(a + F * (b - c) + F * (c - d), self.bounds[0], self.bounds[1])
                else:
                    mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

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

