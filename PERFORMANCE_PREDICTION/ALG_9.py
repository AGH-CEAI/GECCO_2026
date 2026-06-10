import numpy as np
from base_algorithm import BaseAlgorithm


class Algorithm(BaseAlgorithm):

    def __init__(self, gnbg):
        super().__init__(gnbg)

        # Population and memories
        self.H = 10
        self.N_init_factor = 18
        self.N_init_min = 40
        self.N_init_max = 220
        self.N_min = 5
        self.archive_factor = 3

        # p-best pressure
        self.p_min = 0.05
        self.p_max = 0.22

        # Strategy adaptation
        self.strategy_alpha = 0.12
        self.strategy_floor = 0.10

        # Local search
        self.local_period = 4
        self.local_start = 0.35
        self.local_lambda_base = 4
        self.local_sigma_init = 0.18
        self.local_sigma_min = 1e-4
        self.local_sigma_max = 0.30

        # Stagnation / rejuvenation
        self.stall_local = 8
        self.stall_restart = 14
        self.restart_frac = 0.16
        self.restart_noise = 0.10

    def run(self):
        dim = self.dim
        lb, ub = self.lb, self.ub
        span = ub - lb

        def sample_cauchy_positive(loc, scale):
            val = loc + scale * np.tan(np.pi * (np.random.rand() - 0.5))
            tries = 0
            while val <= 0.0 and tries < 16:
                val = loc + scale * np.tan(np.pi * (np.random.rand() - 0.5))
                tries += 1
            if val <= 0.0:
                val = 0.05
            return min(val, 1.0)

        def sample_rank_index(cumw, N, forbid_a=-1, forbid_b=-1, forbid_c=-1):
            for _ in range(24):
                r = np.random.rand()
                idx = int(np.searchsorted(cumw, r, side="left"))
                if idx >= N:
                    idx = N - 1
                if idx != forbid_a and idx != forbid_b and idx != forbid_c:
                    return idx
            idx = int(np.random.randint(N))
            while idx == forbid_a or idx == forbid_b or idx == forbid_c:
                idx = int(np.random.randint(N))
            return idx

        def sample_union_vector(pop, archive, N, forbid_a=-1, forbid_b=-1, forbid_c=-1):
            total = N + archive.shape[0]
            while True:
                rr = int(np.random.randint(total))
                if rr < N:
                    if rr == forbid_a or rr == forbid_b or rr == forbid_c:
                        continue
                    return pop[rr]
                else:
                    return archive[rr - N]

        def add_to_archive(archive, items, max_size):
            if items.size == 0:
                return archive
            if archive.size == 0:
                archive = items.copy()
            else:
                archive = np.vstack((archive, items))
            if archive.shape[0] > max_size:
                keep = np.random.choice(archive.shape[0], size=max_size, replace=False)
                archive = archive[keep]
            return archive

        def elite_eigensystem(elite):
            m = elite.shape[0]
            if m <= 1:
                return np.eye(dim), np.maximum((0.05 * span) ** 2, 1e-18)
            centered = elite - np.mean(elite, axis=0, keepdims=True)
            cov = (centered.T @ centered) / max(1, m - 1)
            reg = 1e-12 * np.mean(span * span) + 1e-18
            cov = cov + np.eye(dim) * reg
            vals, vecs = np.linalg.eigh(cov)
            vals = np.maximum(vals, 1e-18)
            return vecs, vals

        N = int(np.clip(self.N_init_factor * dim, self.N_init_min, self.N_init_max))
        N0 = N

        # Initialize population uniformly in bounds
        pop = self.random_population(N)
        fitness = self.evaluate_pop(pop)

        best_idx = int(np.argmin(fitness))
        best_val = fitness[best_idx]
        best_pos = pop[best_idx].copy()

        # Opposition-enhanced initialization
        if not self.should_stop():
            opp = lb + ub - pop
            opp = self.reflect(opp)
            opp_fit = self.evaluate_pop(opp)

            merged = np.vstack((pop, opp))
            merged_fit = np.concatenate((fitness, opp_fit))
            keep = np.argsort(merged_fit)[:N]
            pop = merged[keep]
            fitness = merged_fit[keep]

            best_idx = int(np.argmin(fitness))
            if fitness[best_idx] < best_val:
                best_val = fitness[best_idx]
                best_pos = pop[best_idx].copy()

        archive = np.empty((0, dim))

        MF = np.full(self.H, 0.5)
        MCR = np.full(self.H, 0.9)
        mem_pos = 0

        strategy_probs = np.array([0.50, 0.30, 0.20], dtype=float)
        local_sigma = self.local_sigma_init
        stall = 0
        generation = 0
        best_shift = np.zeros(dim)

        while not self.should_stop():
            generation += 1
            prev_best = best_pos.copy()

            order = np.argsort(fitness)
            pop = pop[order]
            fitness = fitness[order]
            N = pop.shape[0]

            t = self.gnbg.FE / max(1, self.gnbg.MaxEvals)

            rank_w = 1.0 / np.sqrt(np.arange(1, N + 1, dtype=float))
            cumw = np.cumsum(rank_w)
            cumw /= cumw[-1]

            p_rate = self.p_min + (self.p_max - self.p_min) * ((1.0 - t) ** 0.7)
            p_num = max(2, int(np.ceil(p_rate * N)))

            elite_num = max(3, int(np.ceil(0.20 * N)))
            elite = pop[:elite_num]
            elite_mean = np.mean(elite, axis=0)
            Q, evals = elite_eigensystem(elite)

            eig_prob = 0.10 + 0.70 * t

            trial = np.empty_like(pop)
            F_used = np.empty(N)
            CR_used = np.empty(N)
            S_used = np.empty(N, dtype=int)

            for i in range(N):
                if self.should_stop():
                    break

                k = int(np.random.randint(self.H))
                Fi = sample_cauchy_positive(MF[k], 0.1)
                CRi = float(np.clip(np.random.normal(MCR[k], 0.1), 0.0, 1.0))
                strat = int(np.random.choice(3, p=strategy_probs))

                F_used[i] = Fi
                CR_used[i] = CRi
                S_used[i] = strat

                xi = pop[i]

                pbest_idx = int(np.random.randint(p_num))
                xpbest = pop[pbest_idx]

                r1 = sample_rank_index(cumw, N, i, pbest_idx, -1)
                xr1 = pop[r1]
                xr2 = sample_union_vector(pop, archive, N, i, pbest_idx, r1)

                if strat == 0:
                    mutant = xi + Fi * (xpbest - xi) + Fi * (xr1 - xr2)
                elif strat == 1:
                    drift_scale = np.random.uniform(0.0, 0.25 * (1.0 - t))
                    mutant = xi + Fi * (xpbest - xi) + Fi * (xr1 - xr2) + drift_scale * (elite_mean - xi)
                else:
                    p2 = int(np.random.randint(p_num))
                    guide = 0.5 * (xpbest + pop[p2])
                    mutant = xi + Fi * (guide - xi) + Fi * (xr1 - xr2)

                if np.random.rand() < eig_prob:
                    xrot = xi @ Q
                    vrot = mutant @ Q
                    mask = np.random.rand(dim) < CRi
                    mask[int(np.random.randint(dim))] = True
                    urot = xrot.copy()
                    urot[mask] = vrot[mask]
                    ui = urot @ Q.T
                else:
                    mask = np.random.rand(dim) < CRi
                    mask[int(np.random.randint(dim))] = True
                    ui = xi.copy()
                    ui[mask] = mutant[mask]

                trial[i] = ui

            if self.should_stop():
                break

            trial = self.reflect(trial)
            trial_fit = self.evaluate_pop(trial)

            success = trial_fit < fitness
            if np.any(success):
                old_fit = fitness[success].copy()
                archive = add_to_archive(
                    archive,
                    pop[success],
                    max(self.archive_factor * max(N, self.N_min), 1),
                )

                pop[success] = trial[success]
                fitness[success] = trial_fit[success]

                improv = old_fit - trial_fit[success]
                w = improv / (np.sum(improv) + 1e-32)
                SF = F_used[success]
                SCR = CR_used[success]

                denom = np.sum(w * SF)
                if denom > 0.0:
                    MF[mem_pos] = np.sum(w * SF * SF) / (denom + 1e-32)
                else:
                    MF[mem_pos] = 0.5
                MCR[mem_pos] = np.sum(w * SCR)
                mem_pos = (mem_pos + 1) % self.H

                gains = np.zeros(3)
                suc_ids = np.where(success)[0]
                for idx_local, gain in zip(suc_ids, improv):
                    gains[S_used[idx_local]] += gain
                if np.sum(gains) > 0.0:
                    target = gains / np.sum(gains)
                    strategy_probs = (1.0 - self.strategy_alpha) * strategy_probs + self.strategy_alpha * target
                    strategy_probs = np.maximum(strategy_probs, self.strategy_floor)
                    strategy_probs /= np.sum(strategy_probs)

            best_idx = int(np.argmin(fitness))
            if fitness[best_idx] < best_val:
                best_val = fitness[best_idx]
                best_pos = pop[best_idx].copy()
                delta = best_pos - prev_best
                if np.any(delta != 0.0):
                    best_shift = delta
                stall = 0
            else:
                stall += 1

            # Linear population size reduction
            target_N = int(round(N0 - t * (N0 - self.N_min)))
            target_N = int(np.clip(target_N, self.N_min, N0))
            if pop.shape[0] > target_N:
                order = np.argsort(fitness)
                keep = order[:target_N]
                pop = pop[keep]
                fitness = fitness[keep]
                N = target_N

                max_archive = self.archive_factor * N
                if archive.shape[0] > max_archive:
                    keep_a = np.random.choice(archive.shape[0], size=max_archive, replace=False)
                    archive = archive[keep_a]

            # Budget-aware local covariance sampling
            do_local = ((t >= self.local_start) and (generation % self.local_period == 0)) or (stall >= self.stall_local)
            if do_local and not self.should_stop():
                N = pop.shape[0]
                order = np.argsort(fitness)

                mu = max(3, min(N, max(dim + 1, int(np.ceil(0.22 * N)))))
                elite_ls = pop[order[:mu]]
                elite_ls_mean = np.mean(elite_ls, axis=0)
                Qls, Dls = elite_eigensystem(elite_ls)

                m = self.local_lambda_base + dim
                z = np.random.normal(size=(m, dim))
                steps = (z * np.sqrt(Dls)) @ Qls.T

                scale = local_sigma * (0.15 + 0.85 * ((1.0 - t) ** 1.6))
                local_pop = best_pos + scale * steps

                dir_count = max(2, m // 3)
                hist = best_shift
                guide = best_pos - elite_ls_mean
                anis = np.std(elite_ls, axis=0) + 1e-12
                noise = np.random.normal(size=(dir_count, dim)) * (0.12 * scale) * anis
                line_pop = []
                for j in range(dir_count):
                    a = np.random.uniform(0.4, 1.2)
                    b = np.random.uniform(-0.3, 0.8)
                    cand = best_pos + a * scale * guide + b * scale * hist + noise[j]
                    line_pop.append(cand)
                line_pop = np.asarray(line_pop)

                local_pop = np.vstack((local_pop, line_pop))
                local_pop = self.reflect(local_pop)
                local_fit = self.evaluate_pop(local_pop)

                finite = np.isfinite(local_fit)
                if np.any(finite):
                    local_pop = local_pop[finite]
                    local_fit = local_fit[finite]

                    loc_best_idx = int(np.argmin(local_fit))
                    if local_fit[loc_best_idx] < best_val:
                        new_best = local_pop[loc_best_idx].copy()
                        delta = new_best - best_pos
                        if np.any(delta != 0.0):
                            best_shift = delta
                        best_val = local_fit[loc_best_idx]
                        best_pos = new_best
                        stall = 0
                        local_sigma = min(self.local_sigma_max, local_sigma * 1.15)
                    else:
                        local_sigma = max(self.local_sigma_min, local_sigma * 0.88)

                    order_loc = np.argsort(local_fit)
                    order_worst = np.argsort(fitness)[::-1]
                    replace_count = min(len(order_loc), len(order_worst), max(1, dim // 2))
                    for j in range(replace_count):
                        if self.should_stop():
                            break
                        li = order_loc[j]
                        wi = order_worst[j]
                        if local_fit[li] < fitness[wi]:
                            pop[wi] = local_pop[li]
                            fitness[wi] = local_fit[li]

                    best_idx = int(np.argmin(fitness))
                    if fitness[best_idx] < best_val:
                        delta = pop[best_idx] - best_pos
                        if np.any(delta != 0.0):
                            best_shift = delta
                        best_val = fitness[best_idx]
                        best_pos = pop[best_idx].copy()
                        stall = 0

            # Structured rejuvenation after prolonged stagnation
            if stall >= self.stall_restart and not self.should_stop() and t < 0.95:
                N = pop.shape[0]
                order = np.argsort(fitness)
                keep_elite = max(2, int(np.ceil(0.18 * N)))
                center = 0.5 * (best_pos + np.mean(pop[order[:keep_elite]], axis=0))

                q = max(2, int(np.ceil(self.restart_frac * N)))
                worst_idx = order[-q:]
                worst = pop[worst_idx]

                opposite = lb + ub - worst
                guided = 0.5 * opposite + 0.5 * (2.0 * center - worst)
                guided += np.random.normal(size=guided.shape) * (self.restart_noise * (1.0 - 0.6 * t)) * span

                guided = self.reflect(guided)
                guided_fit = self.evaluate_pop(guided)

                for j in range(q):
                    if self.should_stop():
                        break
                    if guided_fit[j] < fitness[worst_idx[j]]:
                        pop[worst_idx[j]] = guided[j]
                        fitness[worst_idx[j]] = guided_fit[j]

                best_idx = int(np.argmin(fitness))
                if fitness[best_idx] < best_val:
                    delta = pop[best_idx] - best_pos
                    if np.any(delta != 0.0):
                        best_shift = delta
                    best_val = fitness[best_idx]
                    best_pos = pop[best_idx].copy()
                    stall = 0
                else:
                    stall = self.stall_local

            if self.should_stop():
                break

        return best_val, best_pos