import numpy as np
from base_algorithm import BaseAlgorithm

class Algorithm(BaseAlgorithm):

    def __init__(self, gnbg):
        super().__init__(gnbg)

        self.N = 120            # Population size
        self.p = 0.1            # Top 10% used for pbest mutation target
        self.c = 0.1            # Adaptation rate for F and CR memory
        
    def run(self):
        dim = self.dim
        lb, ub = self.lb, self.ub
        N = self.N

        # 1. Opposition-Based Learning (OBL) Initialization
        pop_rand = self.random_population(N)
        pop_opp = lb + ub - pop_rand
        pop_opp = self.reflect(pop_opp)  # Handle any potential floating-point drifts

        fit_rand = self.evaluate_pop(pop_rand)
        if self.should_stop():
            best_idx = int(np.argmin(fit_rand))
            return fit_rand[best_idx], pop_rand[best_idx].copy()

        fit_opp = self.evaluate_pop(pop_opp)

        # Pool together and select the best N individuals
        all_pop = np.vstack((pop_rand, pop_opp))
        all_fit = np.concatenate((fit_rand, fit_opp))

        best_idxs = np.argsort(all_fit)[:N]
        pop = all_pop[best_idxs]
        fitness = all_fit[best_idxs]

        # Tracking global best
        best_val = fitness[0]
        best_pos = pop[0].copy()

        # Adaptive memory variables (F: Cauchy, CR: Normal)
        mu_F = 0.5
        mu_CR = 0.5

        # Historical archive to maintain diversity
        archive = np.zeros((N, dim))
        archive_len = 0

        # Pre-allocate static index array for fast vectorization
        idxs = np.arange(N)
        range_span = ub[0] - lb[0]

        while not self.should_stop():
            # Generate F and CR based on historical success distributions
            CR = np.random.normal(mu_CR, 0.1, N)
            CR = np.clip(CR, 0.0, 1.0)

            F = np.random.standard_cauchy(N) * 0.1 + mu_F
            F = np.clip(F, 0.05, 1.2)  # Bound F to prevent collapse or extreme explosions

            # --- Mutation: current-to-pbest/1 ---
            # 1. Select target 'pbest' from top p%
            p_num = max(2, int(N * self.p))
            sorted_fit_idxs = np.argsort(fitness)
            top_p_idxs = sorted_fit_idxs[:p_num]
            pbest_idxs = np.random.choice(top_p_idxs, N)

            # 2. Select r1 mutually exclusive from the current individual i
            r1_idxs = np.random.randint(0, N, N)
            r1_idxs = np.where(r1_idxs == idxs, (r1_idxs + 1) % N, r1_idxs)

            # 3. Select r2 from (Population U Archive), mutually exclusive from i and r1
            total_size = N + archive_len
            r2_idxs = np.random.randint(0, total_size, N)
            r2_idxs = np.where(r2_idxs == r1_idxs, (r2_idxs + 1) % total_size, r2_idxs)
            r2_idxs = np.where(r2_idxs == idxs, (r2_idxs + 2) % total_size, r2_idxs)

            # Build concatenated array for r2 selection
            pop_archive = np.vstack((pop, archive[:archive_len])) if archive_len > 0 else pop

            # Create Mutant Vectors (V)
            V = pop + F[:, None] * (pop[pbest_idxs] - pop) + F[:, None] * (pop[r1_idxs] - pop_archive[r2_idxs])

            # --- Crossover (Binomial) ---
            cross_mask = np.random.rand(N, dim) < CR[:, None]
            j_rand = np.random.randint(0, dim, N)
            cross_mask[idxs, j_rand] = True  # Guarantee at least 1 dimension mutates

            U = np.where(cross_mask, V, pop)

            # Boundary Enforcement
            U = self.reflect(U)

            if self.should_stop():
                break

            # Evaluate trial vectors
            fit_U = self.evaluate_pop(U)
            
            # Fast safety break in case budget exhausted mid-evaluation
            if self.should_stop() and np.all(np.isinf(fit_U)):
                break

            # Selection: Strict improvement mask
            success_mask = fit_U <= fitness
            replaced_idxs = np.where(success_mask)[0]

            # Update archive and parameters if there were improvements
            if len(replaced_idxs) > 0:
                replaced_pop = pop[replaced_idxs]
                n_replaced = len(replaced_pop)

                # Manage Archive Capacity
                if archive_len + n_replaced <= N:
                    archive[archive_len:archive_len+n_replaced] = replaced_pop
                    archive_len += n_replaced
                else:
                    rem = N - archive_len
                    if rem > 0:
                        archive[archive_len:N] = replaced_pop[:rem]
                    leftover = n_replaced - rem
                    replace_pos = np.random.choice(N, leftover, replace=False)
                    archive[replace_pos] = replaced_pop[rem:]
                    archive_len = N

                # Adapt parameters via Lehmer Mean (F) and Arithmetic Mean (CR)
                succ_F = F[replaced_idxs]
                succ_CR = CR[replaced_idxs]

                if np.sum(succ_F) > 0:
                    mu_F = (1 - self.c) * mu_F + self.c * (np.sum(succ_F**2) / np.sum(succ_F))
                mu_CR = (1 - self.c) * mu_CR + self.c * np.mean(succ_CR)

            # Apply selection to main population
            pop = np.where(success_mask[:, None], U, pop)
            fitness = np.where(success_mask, fit_U, fitness)

            # Track Global Best (Ignoring infinites from budget exhaustion)
            valid_mask = ~np.isinf(fit_U)
            if np.any(valid_mask):
                min_u_idx = int(np.argmin(fit_U))
                if fit_U[min_u_idx] < best_val:
                    best_val = fit_U[min_u_idx]
                    best_pos = U[min_u_idx].copy()

            # --- Diversity Tracking & Stagnation Restart ---
            # Measures spatial dispersion to trigger a reset before budget is wasted
            pop_diversity = np.mean(np.std(pop, axis=0))
            if pop_diversity < 1e-4 * range_span:
                restart_count = int(N * 0.7)  # Inject 70% fresh DNA
                worst_idxs = np.argsort(fitness)[-restart_count:]

                new_pop = self.random_population(restart_count)
                new_pop = self.reflect(new_pop)

                if self.should_stop():
                    break

                new_fit = self.evaluate_pop(new_pop)

                pop[worst_idxs] = new_pop
                fitness[worst_idxs] = new_fit

                # Reset adaptation parameters & flush archive
                mu_F = 0.5
                mu_CR = 0.5
                archive_len = 0

        return best_val, best_pos