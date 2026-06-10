import math
import numpy as np
from base_algorithm import BaseAlgorithm


class Algorithm(BaseAlgorithm):
    def __init__(self, gnbg):
        super().__init__(gnbg)
        self.N_init_factor = 20
        self.N_min = 4
        self.H = 5
        self.p_best_init = 0.11
        self.archive_rate = 2.6
        self.prescreening_ratio = 3
        self.surr_update_period = 5
        self.stag_limit = 100
        self.restart_frac = 0.35
        self.nm_trigger = 0.75
        self.nm_budget_frac = 0.06
        self.nm_period = 30

    def run(self):
        dim = self.dim
        lb, ub = self.lb, self.ub
        budget = self.gnbg.MaxEvals
        rng = ub - lb

        # === INITIALIZATION with opposition ===
        N_init = max(60, self.N_init_factor * dim)
        N = N_init

        pop = self.random_population(N)
        opp = lb + ub - pop
        opp = self.clip(opp)
        combined = np.vstack([pop, opp])
        combined_fit = self.evaluate_pop(combined)
        if self.should_stop():
            bi = int(np.argmin(combined_fit))
            return combined_fit[bi], combined[bi].copy()

        order = np.argsort(combined_fit)[:N]
        pop = combined[order].copy()
        fitness = combined_fit[order].copy()

        best_val = fitness[0]
        best_pos = pop[0].copy()

        # === SHADE MEMORY ===
        M_F = np.full(self.H, 0.3)
        M_CR = np.full(self.H, 0.8)
        mem_k = 0

        # === ARCHIVE ===
        archive_max = int(self.archive_rate * N_init)
        archive = np.empty((0, dim))

        # === SURROGATE STATE (diagonal quadratic: f ≈ c + b·x + d·x²) ===
        # Rolling buffer stored as pre-allocated arrays
        buf_max = max(500, 4 * N_init)
        buf_X = np.empty((buf_max, dim))
        buf_y = np.empty(buf_max)
        buf_n = 0
        surr_w = None  # weights: [c, b_1..b_d, d_1..d_d]
        surr_mean = None
        surr_std = None
        surr_valid = False

        def _buf_add(X, y):
            nonlocal buf_X, buf_y, buf_n
            valid = np.isfinite(y)
            X2, y2 = X[valid], y[valid]
            n_new = len(X2)
            if n_new == 0:
                return
            if buf_n + n_new <= buf_max:
                buf_X[buf_n:buf_n + n_new] = X2
                buf_y[buf_n:buf_n + n_new] = y2
                buf_n += n_new
            else:
                # Shift out oldest, keep most recent
                keep = buf_max - n_new
                if keep > 0 and buf_n > 0:
                    buf_X[:keep] = buf_X[buf_n - keep:buf_n]
                    buf_y[:keep] = buf_y[buf_n - keep:buf_n]
                buf_X[keep:keep + n_new] = X2
                buf_y[keep:keep + n_new] = y2
                buf_n = keep + n_new

        def _fit_surrogate():
            nonlocal surr_w, surr_mean, surr_std, surr_valid
            if buf_n < 2 * dim + 5:
                surr_valid = False
                return
            X = buf_X[:buf_n]
            y = buf_y[:buf_n]
            surr_mean = X.mean(axis=0)
            surr_std = X.std(axis=0)
            surr_std[surr_std < 1e-12] = 1e-12
            Xn = (X - surr_mean) / surr_std
            # Diagonal quadratic features: [1, x, x²]
            Phi = np.hstack([np.ones((buf_n, 1)), Xn, Xn ** 2])
            try:
                surr_w, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
                # Quick quality check: rank correlation on last quarter
                n4 = max(10, buf_n // 4)
                Xv = Xn[-n4:]
                yv = y[-n4:]
                Phi_v = np.hstack([np.ones((n4, 1)), Xv, Xv ** 2])
                pred = Phi_v @ surr_w
                # Spearman
                ry = np.argsort(np.argsort(yv)).astype(float)
                rp = np.argsort(np.argsort(pred)).astype(float)
                dd = ry - rp
                rho = 1.0 - 6.0 * np.sum(dd ** 2) / (n4 * (n4 ** 2 - 1))
                surr_valid = rho > 0.25
            except Exception:
                surr_valid = False

        def _predict(X_new):
            if not surr_valid or surr_w is None:
                return None
            Xn = (X_new - surr_mean) / surr_std
            Phi = np.hstack([np.ones((len(X_new), 1)), Xn, Xn ** 2])
            return Phi @ surr_w

        def _surr_gradient(x):
            if not surr_valid or surr_w is None:
                return None
            xn = (x - surr_mean) / surr_std
            b = surr_w[1:dim + 1]
            d = surr_w[dim + 1:2 * dim + 1]
            return (b + 2 * d * xn) / surr_std

        # Add initial evals to buffer
        _buf_add(pop, fitness)

        # === TRACKING ===
        stag_counter = 0
        gen = 0

        # === MAIN LOOP ===
        while not self.should_stop():
            gen += 1
            progress = self.gnbg.FE / budget

            # Rebuild surrogate periodically
            if gen % self.surr_update_period == 1:
                _fit_surrogate()

            # --- Linear pop reduction ---
            N_target = max(self.N_min, int(round(
                N_init - (N_init - self.N_min) * progress)))
            if N_target < N:
                keep = np.argsort(fitness[:N])[:N_target]
                pop = pop[keep].copy()
                fitness = fitness[keep].copy()
                N = N_target
                a_max = max(N, int(self.archive_rate * N))
                if len(archive) > a_max:
                    archive = archive[np.random.choice(len(archive), a_max, False)]

            # --- Adaptive pbest ---
            p_best = max(2.0 / N, self.p_best_init * (1.0 - 0.5 * progress))
            n_pbest = max(1, int(round(p_best * N)))

            # --- Generate F, CR (vectorized) ---
            ri = np.random.randint(0, self.H, size=N)
            # F: Cauchy
            F_vals = M_F[ri] + 0.1 * np.random.standard_cauchy(N)
            for attempt in range(20):
                bad = F_vals <= 0
                if not bad.any():
                    break
                F_vals[bad] = M_F[ri[bad]] + 0.1 * np.random.standard_cauchy(bad.sum())
            F_vals = np.clip(F_vals, 0.01, 1.0)
            if progress < 0.2:
                F_vals = np.maximum(F_vals, 0.2)
            elif progress < 0.4:
                F_vals = np.maximum(F_vals, 0.1)

            # CR: Gaussian
            CR_vals = np.clip(np.random.normal(M_CR[ri], 0.1), 0.0, 1.0)

            sorted_idx = np.argsort(fitness[:N])
            pool_size = N + len(archive)

            # --- Generate candidates (vectorized) ---
            K = self.prescreening_ratio if surr_valid else 1
            n_cand = K * N

            # For each candidate, pick parent = ci % N
            parent_idx = np.tile(np.arange(N), K)

            # pbest indices
            pb_idx = sorted_idx[np.random.randint(0, n_pbest, size=n_cand)]

            # r1 != parent
            r1 = np.random.randint(0, N, size=n_cand)
            same = r1 == parent_idx
            r1[same] = (r1[same] + 1 + np.random.randint(0, N - 1, size=same.sum())) % N

            # r2 from pool != parent, != r1
            r2 = np.random.randint(0, pool_size, size=n_cand)
            bad2 = (r2 == parent_idx) | (r2 == r1)
            r2[bad2] = (r2[bad2] + 1 + np.random.randint(0, pool_size - 1, size=bad2.sum())) % pool_size

            # Build r2 vectors
            if len(archive) > 0:
                full_pool = np.vstack([pop[:N], archive])
            else:
                full_pool = pop[:N]
            # Clamp r2 to valid range
            r2 = np.clip(r2, 0, len(full_pool) - 1)
            x_r2 = full_pool[r2]

            # F values tiled for K candidates per parent
            F_tiled = np.tile(F_vals, K)
            CR_tiled = np.tile(CR_vals, K)

            # Mutation: current-to-pbest/1
            parents = pop[parent_idx]
            mutants = parents + F_tiled[:, None] * (pop[pb_idx] - parents) + \
                      F_tiled[:, None] * (pop[r1] - x_r2)

            # Binomial crossover (vectorized)
            trials = parents.copy()
            j_rand = np.random.randint(0, dim, size=n_cand)
            mask = np.random.rand(n_cand, dim) < CR_tiled[:, None]
            # Ensure j_rand is always crossed
            mask[np.arange(n_cand), j_rand] = True
            trials[mask] = mutants[mask]

            # Boundary handling
            trials = self.reflect(trials)

            # --- Surrogate pre-screening ---
            if K > 1 and surr_valid:
                pred = _predict(trials)
                if pred is not None:
                    # For each parent i, pick the candidate with lowest prediction
                    selected = np.empty((N, dim))
                    for i in range(N):
                        # Candidates for parent i are at indices i, i+N, i+2N, ...
                        cidx = np.arange(i, n_cand, N)
                        best_c = cidx[np.argmin(pred[cidx])]
                        selected[i] = trials[best_c]
                else:
                    selected = trials[:N]
            else:
                selected = trials[:N]

            # --- Evaluate ---
            trial_fit = self.evaluate_pop(selected)
            if self.should_stop():
                for j in range(N):
                    if np.isfinite(trial_fit[j]) and trial_fit[j] < best_val:
                        best_val = trial_fit[j]
                        best_pos = selected[j].copy()
                break

            _buf_add(selected, trial_fit)

            # --- Selection + SHADE memory ---
            S_F, S_CR, S_delta = [], [], []
            improved = trial_fit <= fitness[:N]
            for i in range(N):
                if improved[i]:
                    delta = fitness[i] - trial_fit[i]
                    if delta > 0:
                        S_F.append(F_vals[i])
                        S_CR.append(CR_vals[i])
                        S_delta.append(delta)
                    # Archive old
                    if len(archive) < archive_max:
                        archive = np.vstack([archive, pop[i:i + 1]]) if len(archive) > 0 else pop[i:i + 1].copy()
                    elif archive_max > 0:
                        archive[np.random.randint(len(archive))] = pop[i]
                    pop[i] = selected[i]
                    fitness[i] = trial_fit[i]

            if S_F:
                sf = np.array(S_F)
                scr = np.array(S_CR)
                sd = np.array(S_delta)
                w = sd / sd.sum()
                M_F[mem_k] = np.sum(w * sf * sf) / max(np.sum(w * sf), 1e-30)
                M_CR[mem_k] = np.sum(w * scr)
                mem_k = (mem_k + 1) % self.H

            # --- Update best ---
            bi = int(np.argmin(fitness[:N]))
            if fitness[bi] < best_val:
                best_val = fitness[bi]
                best_pos = pop[bi].copy()
                stag_counter = 0
            else:
                stag_counter += 1

            # --- Surrogate gradient descent (cheap exploitation) ---
            if surr_valid and gen % 7 == 0 and not self.should_stop():
                x_g = best_pos.copy()
                step = 0.02 * np.mean(rng) * max(0.01, 1 - progress)
                for _ in range(3):
                    if self.should_stop():
                        break
                    g = _surr_gradient(x_g)
                    if g is None or not np.all(np.isfinite(g)):
                        break
                    gn = np.linalg.norm(g)
                    if gn < 1e-15:
                        break
                    x_new = x_g - step * (g / gn)
                    x_new = self.clip(x_new.reshape(1, -1)).ravel()
                    val = self.evaluate(x_new)
                    if self.should_stop():
                        if val < best_val:
                            best_val = val
                            best_pos = x_new.copy()
                        break
                    _buf_add(x_new.reshape(1, -1), np.array([val]))
                    if val < best_val:
                        best_val = val
                        best_pos = x_new.copy()
                        x_g = x_new.copy()
                        stag_counter = 0
                    else:
                        step *= 0.5

            # --- Stagnation restart ---
            if stag_counter >= self.stag_limit and N > self.N_min + 2:
                n_rst = max(1, int(self.restart_frac * N))
                worst = np.argsort(fitness[:N])[-n_rst:]
                mid = (lb + ub) / 2.0
                for idx in worst:
                    if np.random.rand() < 0.5:
                        pop[idx] = 2 * best_pos - pop[idx]
                    else:
                        radius = rng * max(0.01, 0.3 * (1 - progress))
                        pop[idx] = best_pos + radius * np.random.randn(dim)
                pop[worst] = self.reflect(pop[worst])
                fitness[worst] = self.evaluate_pop(pop[worst])
                if self.should_stop():
                    for wi in worst:
                        if fitness[wi] < best_val:
                            best_val = fitness[wi]
                            best_pos = pop[wi].copy()
                    break
                for wi in worst:
                    if fitness[wi] < best_val:
                        best_val = fitness[wi]
                        best_pos = pop[wi].copy()
                stag_counter = 0

            # --- Nelder-Mead in late stages ---
            if (progress > self.nm_trigger and gen % self.nm_period == 0
                    and not self.should_stop()):
                remaining = budget - self.gnbg.FE
                nm_bud = max(dim + 2, int(self.nm_budget_frac * remaining))
                nm_val, nm_pos = self._nelder_mead(best_pos.copy(), nm_bud)
                if nm_val < best_val:
                    best_val = nm_val
                    best_pos = nm_pos.copy()
                    worst_i = int(np.argmax(fitness[:N]))
                    pop[worst_i] = nm_pos.copy()
                    fitness[worst_i] = nm_val
                    stag_counter = 0
                if self.should_stop():
                    break

        return best_val, best_pos

    def _nelder_mead(self, start, max_fe):
        dim = self.dim
        lb, ub = self.lb, self.ub
        rng = ub - lb
        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        n = dim
        simplex = np.empty((n + 1, n))
        simplex[0] = start.copy()
        step = 0.02 * rng
        for i in range(n):
            simplex[i + 1] = start.copy()
            if start[i] + step[i] <= ub[i]:
                simplex[i + 1, i] += step[i]
            else:
                simplex[i + 1, i] -= step[i]

        simplex = self.clip(simplex)
        f_simplex = self.evaluate_pop(simplex)
        fe = n + 1

        if self.should_stop() or fe >= max_fe:
            bi = int(np.argmin(f_simplex))
            return f_simplex[bi], simplex[bi].copy()

        while fe < max_fe and not self.should_stop():
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]
            centroid = simplex[:-1].mean(axis=0)

            # Reflect
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = self.clip(xr.reshape(1, -1)).ravel()
            fr = self.evaluate(xr)
            fe += 1
            if self.should_stop() or fe >= max_fe:
                return (fr, xr) if fr < f_simplex[0] else (f_simplex[0], simplex[0].copy())

            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr; f_simplex[-1] = fr; continue

            if fr < f_simplex[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = self.clip(xe.reshape(1, -1)).ravel()
                fev = self.evaluate(xe)
                fe += 1
                if fev < fr:
                    simplex[-1] = xe; f_simplex[-1] = fev
                else:
                    simplex[-1] = xr; f_simplex[-1] = fr
                if self.should_stop():
                    bi = int(np.argmin(f_simplex))
                    return f_simplex[bi], simplex[bi].copy()
                continue

            if fr < f_simplex[-1]:
                xc = centroid + rho * (xr - centroid)
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
            xc = self.clip(xc.reshape(1, -1)).ravel()
            fc = self.evaluate(xc)
            fe += 1
            if self.should_stop():
                return (fc, xc) if fc < f_simplex[0] else (f_simplex[0], simplex[0].copy())

            if fc < max(fr, f_simplex[-1]):
                simplex[-1] = xc; f_simplex[-1] = fc; continue

            for i in range(1, n + 1):
                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            simplex = self.clip(simplex)
            f_simplex[1:] = self.evaluate_pop(simplex[1:])
            fe += n
            if self.should_stop():
                bi = int(np.argmin(f_simplex))
                return f_simplex[bi], simplex[bi].copy()

        bi = int(np.argmin(f_simplex))
        return f_simplex[bi], simplex[bi].copy()
