import numpy as np

class Algorithm():
    def __init__(self, func, dim, bounds, max_evals, params=None):
        if params is None:
            self.params = {
                'pop_size': 400,
                'F_base': 0.5,
                'F_amp': 0.3,
                'CR_base': 0.5,
                'CR_amp': 0.4,
                'p': 0.45,
                'beta': 0.5,
                'archive_size': 350,
                'epsilon': 1e-10,
                'mutation_strategy': 'rand-to-best/2/bin',
                'local_search_frequency': 10,
                'gamma': 0.1,
                'adaptive_F_CR': True,
                'strategy_variation_frequency': 5,
                'diversification_frequency': 50,
                'diversification_strength': 0.05,
                'elite_preservation': True,
                'elite_size': 0.1
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
            strategies = ['rand/1/bin', 'best/2/bin', 'rand-to-best/2/bin']
            self.params['mutation_strategy'] = np.random.choice(strategies)

    def diversify_population(self, pop, fitness):
        idx_worst = np.argmax(fitness)
        diversification_vector = self.params['diversification_strength'] * (np.random.rand(self.dim) - 0.5) * (self.bounds[1] - self.bounds[0])
        pop[idx_worst] += diversification_vector
        pop[idx_worst] = np.clip(pop[idx_worst], self.bounds[0], self.bounds[1])
        fitness[idx_worst] = self.func(pop[idx_worst])
        return pop, fitness

    def preserve_elites(self, pop, fitness):
        elite_size = int(self.params['elite_size'] * len(pop))
        elite_indices = np.argpartition(fitness, elite_size)[:elite_size]
        return pop[elite_indices], fitness[elite_indices]

    def run(self):
        pop_size = self.params.get('pop_size', 400)
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
                a, b, c, d, e = pop[np.random.choice(idxs, 5, replace=False)]

                if self.params['mutation_strategy'] == 'rand-to-best/2/bin':
                    mutant = np.clip(a + F * (best - a) + F * (b - c) + F * (d - e), self.bounds[0], self.bounds[1])
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

            if gen % self.params['diversification_frequency'] == 0:
                pop, fitness = self.diversify_population(pop, fitness)

            if self.params['elite_preservation']:
                elite_pop, elite_fitness = self.preserve_elites(pop, fitness)
                pop = np.concatenate((pop, elite_pop))
                fitness = np.concatenate((fitness, elite_fitness))
                idxs = np.argpartition(fitness, pop_size)[:pop_size]
                pop = pop[idxs]
                fitness = fitness[idxs]

            if evals >= self.max_evals or abs(best_fitness - self.params['epsilon']) < self.params['epsilon']:
                break

        return best_fitness

