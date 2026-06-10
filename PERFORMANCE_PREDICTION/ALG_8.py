import numpy as np
from base_algorithm import BaseAlgorithm


class Algorithm(BaseAlgorithm):

    def __init__(self, gnbg):
        super().__init__(gnbg)

        # ===== Fixed hyperparameters (global, no per-function tuning) =====
        self.N_base = 10
        self.N_scale = 6.0               # N_init = clip(N_base + N_scale * sqrt(dim), 18, 60)
        self.N_min_frac = 0.25           # minimum population fraction after reduction

        self.H = 6                       # success-history memory size
        self.F_init = 0.55
        self.CR_init = 0.85
        self.F_cauchy_scale = 0.10
        self.CR_normal_scale = 0.10

        self.pbest_min = 0.08            # top fraction used by current-to-pbest
        self.pbest_span = 0.20           # more exploratory early, narrower later

        self.elite_frac = 0.22           # fraction used to form elite centroid
        self.stagnation_base = 6         # stagnation trigger base
        self.reseed_cooldown = 5         # minimum generations between rejuvenations
        self.reseed_frac = 0.16          # fraction of worst members considered for rejuvenation
        self.reseed_max = 8              # hard cap on rejuvenated members per event

        self.archive_mult = 3            # external archive size multiplier

    def run(self):
        dim = self.dim
        lb, ub = self.lb, self.ub
        box = ub - lb

        # ---------- helpers ----------
        def sample_F(mu, size):
            # Cauchy-like heavy tail, resample non-positive entries
            F = mu + self.F_cauchy_scale * np.tan(np.pi * (np.random.rand(size) - 0.5))
            bad = F <= 0.0
            while np.any(bad):
                F[bad] = mu[bad] + self.F_cauchy_scale * np.tan(
                    np.pi * (np.random.rand(np.sum(bad)) - 0.5)
                )
                bad = F <= 0.0
            return np.minimum(F, 1.0)

        def binomial_crossover(target, donor, cr):
            mask = np.random.rand(dim) < cr
            if not np.any(mask):
                mask[np.random.randint(dim)] = True
            trial = target.copy()
            trial[mask] = donor[mask]
            return trial

        # ---------- population size ----------
        N_init = int(np.clip(self.N_base + self.N_scale * np.sqrt(dim), 18, 60))
        N_min = max(6, int(np.round(self.N_min_frac * N_init)))

        # ---------- initialization ----------
        pop = self.random_population(N_init)
        pop = self.reflect(pop)
        fitness = self.evaluate_pop(pop)

        if self.should_stop():
            best_idx = int(np.argmin(fitness))
            return float(fitness[best_idx]), pop[best_idx].copy()

        # Opposition-based enrichment (one-shot)
        opp = lb + ub - pop
        opp = self.reflect(opp)
        opp_fit = self.evaluate_pop(opp)

        if self.should_stop():
            all_pop = np.vstack((pop, opp))
            all_fit = np.concatenate((fitness, opp_fit))
            best_idx = int(np.argmin(all_fit))
            return float(all_fit[best_idx]), all_pop[best_idx].copy()

        all_pop = np.vstack((pop, opp))
        all_fit = np.concatenate((fitness, opp_fit))
        keep = np.argsort(all_fit)[:N_init]
        pop = all_pop[keep]
        fitness = all_fit[keep]

        best_idx = int(np.argmin(fitness))
        best_val = float(fitness[best_idx])
        best_pos = pop[best_idx].copy()

        # ---------- adaptive memories ----------
        M_F = np.full(self.H, self.F_init, dtype=float)
        M_CR = np.full(self.H, self.CR_init, dtype=float)
        mem_ptr = 0

        # ---------- archive ----------
        archive = np.empty((0, dim), dtype=float)
        archive_max = self.archive_mult * N_init

        generation = 0
        stagnation_gens = 0
        last_reseed_gen = -10**9

        while not self.should_stop():
            generation += 1
            N = pop.shape[0]
            progress = float(self.gnbg.FE) / float(self.gnbg.MaxEvals)
            progress = min(max(progress, 0.0), 1.0)

            order = np.argsort(fitness)
            pop = pop[order]
            fitness = fitness[order]

            if fitness[0] < best_val:
                best_val = float(fitness[0])
                best_pos = pop[0].copy()
                stagnation_gens = 0
            else:
                stagnation_gens += 1

            elite_count = max(2, int(np.ceil(self.elite_frac * N)))
            elites = pop[:elite_count]
            elite_weights = np.linspace(elite_count, 1, elite_count, dtype=float)
            elite_weights /= np.sum(elite_weights)
            elite_mean = np.sum(elites * elite_weights[:, None], axis=0)

            # Diversity normalized by box size
            diversity = float(np.mean(np.std(pop, axis=0) / (box + 1e-12)))

            # Adaptive p-best range: wider early, narrower late
            pbest_frac = self.pbest_min + self.pbest_span * (1.0 - progress)
            pbest_count = max(2, int(np.ceil(pbest_frac * N)))

            # Sample success-history parameters
            mem_idx = np.random.randint(0, self.H, size=N)
            F_vals = sample_F(M_F[mem_idx], N)
            CR_vals = np.clip(np.random.normal(M_CR[mem_idx], self.CR_normal_scale, size=N), 0.0, 1.0)

            # Operator probabilities: more global early, more local late
            p_global = 0.55 - 0.20 * progress + (0.10 if diversity < 0.06 else 0.0)
            p_geom = 0.28 + 0.06 * (1.0 - abs(2.0 * progress - 1.0))
            p_global = min(max(p_global, 0.20), 0.70)
            p_geom = min(max(p_geom, 0.15), 0.45)

            trials = np.empty_like(pop)

            pool = pop if archive.shape[0] == 0 else np.vstack((pop, archive))
            pool_size = pool.shape[0]

            for i in range(N):
                if self.should_stop():
                    break

                x = pop[i]
                F = F_vals[i]
                CR = CR_vals[i]
                r = np.random.rand()

                # ===== Operator 1: current-to-pbest/1 + elite drift =====
                if r < p_global:
                    pidx = np.random.randint(0, pbest_count)
                    xpbest = pop[pidx]

                    # r1 from population, distinct from i
                    r1 = np.random.randint(0, N - 1)
                    if r1 >= i:
                        r1 += 1
                    xr1 = pop[r1]

                    # r2 from pop+archive, avoid i when sampling from pop part
                    ridx = np.random.randint(0, pool_size)
                    if ridx < N and ridx == i:
                        ridx = (ridx + 1) % pool_size
                    xr2 = pool[ridx]

                    donor = (
                        x
                        + F * (xpbest - x)
                        + F * (xr1 - xr2)
                        + 0.10 * (1.0 - progress) * (elite_mean - x)
                    )

                # ===== Operator 2: elite-geometry manifold search =====
                elif r < p_global + p_geom:
                    idxs = np.random.randint(0, elite_count, size=3)
                    e1 = elites[idxs[0]]
                    e2 = elites[idxs[1]]
                    e3 = elites[idxs[2]]

                    center = 0.5 * (best_pos + elite_mean)
                    sigma = (0.16 * (1.0 - progress) ** 1.35 + 0.010) * box
                    sigma *= (1.0 + 1.8 * max(0.0, 0.05 - diversity))

                    donor = (
                        center
                        + F * (e1 - e2)
                        + 0.50 * F * (best_pos - e3)
                        + np.random.randn(dim) * sigma
                    )

                # ===== Operator 3: late-stage anisotropic local refinement =====
                else:
                    idxs = np.random.randint(0, elite_count, size=2)
                    e1 = elites[idxs[0]]
                    e2 = elites[idxs[1]]

                    anchor = best_pos if np.random.rand() < 0.70 else elites[np.random.randint(elite_count)]
                    local_sigma = (0.055 * (1.0 - progress) ** 2 + 0.0025) * box
                    donor = anchor + 0.50 * F * (e1 - e2) + np.random.randn(dim) * local_sigma

                trial = binomial_crossover(x, donor, CR)
                trials[i] = trial

            if self.should_stop():
                break

            trials = self.reflect(trials)
            trial_fit = self.evaluate_pop(trials)

            old_fitness = fitness.copy()
            improved = trial_fit < fitness

            # Best tracking
            if np.any(improved):
                imp_idx = np.where(improved)[0]
                imp_best_local = imp_idx[int(np.argmin(trial_fit[imp_idx]))]
                if trial_fit[imp_best_local] < best_val:
                    best_val = float(trial_fit[imp_best_local])
                    best_pos = trials[imp_best_local].copy()
                    stagnation_gens = 0

                # Update archive with replaced parents
                archive = np.vstack((archive, pop[improved]))
                if archive.shape[0] > archive_max:
                    sel = np.random.choice(archive.shape[0], size=archive_max, replace=False)
                    archive = archive[sel]

                # Success-history memory update
                delta = old_fitness[improved] - trial_fit[improved]
                delta_sum = np.sum(delta)
                if delta_sum > 0.0:
                    w = delta / (delta_sum + 1e-12)
                    SF = F_vals[improved]
                    SCR = CR_vals[improved]
                    denom = np.sum(w * SF) + 1e-12
                    M_F[mem_ptr] = np.sum(w * SF * SF) / denom
                    M_CR[mem_ptr] = np.sum(w * SCR)
                    mem_ptr = (mem_ptr + 1) % self.H

            pop[improved] = trials[improved]
            fitness[improved] = trial_fit[improved]

            if self.should_stop():
                break

            # ---------- budget-aware linear population reduction ----------
            progress = float(self.gnbg.FE) / float(self.gnbg.MaxEvals)
            progress = min(max(progress, 0.0), 1.0)
            target_N = int(np.round(N_min + (N_init - N_min) * (1.0 - progress)))
            target_N = max(N_min, min(N_init, target_N))

            if pop.shape[0] > target_N:
                keep = np.argsort(fitness)[:target_N]
                pop = pop[keep]
                fitness = fitness[keep]

            # ---------- diversity / stagnation focused rejuvenation ----------
            if self.should_stop():
                break

            trigger = self.stagnation_base + dim // 2
            low_div = diversity < (0.010 + 0.020 * (1.0 - progress))
            need_reseed = (
                progress < 0.92
                and (stagnation_gens >= trigger or low_div)
                and (generation - last_reseed_gen >= self.reseed_cooldown)
                and pop.shape[0] >= 8
            )

            if need_reseed:
                N = pop.shape[0]
                k = min(self.reseed_max, max(1, int(np.ceil(self.reseed_frac * N))))
                worst_idx = np.argsort(fitness)[-k:]

                seeds = self.random_population(k)
                seeds = self.reflect(seeds)

                center = 0.5 * (best_pos + elite_mean)
                beta = 0.15 + 0.55 * progress
                noise = np.random.randn(k, dim) * ((0.22 * (1.0 - progress) + 0.03) * box)

                Xnew = (
                    beta * center[None, :]
                    + (1.0 - beta) * seeds
                    + 0.30 * (center[None, :] - pop[worst_idx])
                    + noise
                )

                Xnew = self.reflect(Xnew)
                new_fit = self.evaluate_pop(Xnew)

                severe = stagnation_gens >= 2 * trigger
                replace = (new_fit < fitness[worst_idx]) | severe

                if np.any(replace):
                    ridx = worst_idx[replace]
                    pop[ridx] = Xnew[replace]
                    fitness[ridx] = new_fit[replace]

                    best_local = int(np.argmin(fitness))
                    if fitness[best_local] < best_val:
                        best_val = float(fitness[best_local])
                        best_pos = pop[best_local].copy()
                        stagnation_gens = 0

                last_reseed_gen = generation

        return best_val, best_pos