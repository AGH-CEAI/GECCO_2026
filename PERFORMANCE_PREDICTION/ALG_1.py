import numpy as np
from base_algorithm import BaseAlgorithm


class Algorithm(BaseAlgorithm):
    def __init__(self, gnbg):
        super().__init__(gnbg)

        # Population sizing
        self.pop_factor = 8         # N_init = max(25, min(factor*dim, 200))
        self.pop_min = 4

        # SHADE memory: 7 adaptive slots + 1 terminal exploitation slot
        self.H = 8

        # p-best fraction range (adapts with progress)
        self.p_min = 0.05
        self.p_max = 0.22

        # External archive size multiplier
        self.arc_rate = 2.6

        # Strategy adaptation
        self.n_strat = 3
        self.strat_period = 25      # adapt every N generations
        self.strat_lr = 0.7         # exponential smoothing rate

        # Eigenvector crossover
        self.eigen_period = 40      # recompute every N generations
        self.eigen_frac = 0.30      # fraction of pop for covariance
        self.eigen_prob = 0.35      # probability of eigen crossover per individual

        # Stagnation handling
        self.stag_base = 40         # base stagnation threshold (+ dim)
        self.restart_frac = 0.25    # fraction of pop to restart

        # Periodic diversity injection
        self.inject_period = 80     # inject random solutions every N gens
        self.inject_frac = 0.10     # fraction of pop to replace

    def run(self):
        dim = self.dim
        lb, ub = self.lb, self.ub
        budget = self.gnbg.MaxEvals
        rng = ub - lb

        # ================================================================
        # Population initialization with opposition-based learning
        # ================================================================
        N_init = max(25, min(self.pop_factor * dim, 200))
        N_min = max(self.pop_min, 4)

        pop = self.random_population(N_init)
        opp = lb + ub - pop
        opp = self.clip(opp)
        combined = np.vstack([pop, opp])
        cfit = self.evaluate_pop(combined)
        if self.should_stop():
            idx = int(np.argmin(cfit))
            return float(cfit[idx]), combined[idx].copy()

        order = np.argsort(cfit)[:N_init]
        pop = combined[order].copy()
        fitness = cfit[order].copy()

        best_val = float(fitness[0])
        best_pos = pop[0].copy()
        N = N_init

        # ================================================================
        # SHADE memory initialization
        # ================================================================
        M_F = np.full(self.H, 0.5)
        M_CR = np.full(self.H, 0.5)
        # Terminal entry: high F for large steps, low CR for exploitation
        M_F[-1] = 0.9
        M_CR[-1] = 0.1
        mem_ptr = 0

        # ================================================================
        # Pre-allocated archive
        # ================================================================
        arc_cap = int(self.arc_rate * N_init)
        archive = np.empty((arc_cap, dim))
        arc_len = 0

        # ================================================================
        # Strategy adaptation state
        # ================================================================
        strat_probs = np.array([0.50, 0.30, 0.20])
        strat_succ = np.zeros(self.n_strat)
        strat_trials = np.zeros(self.n_strat)

        # ================================================================
        # Eigenvector state
        # ================================================================
        eigvecs = np.eye(dim)

        # ================================================================
        # Main loop state
        # ================================================================
        stag = 0
        gen = 0

        while not self.should_stop():
            prog = min(self.gnbg.FE / budget, 1.0)

            # ---- Sort population by fitness ----
            si = np.argsort(fitness[:N])
            pop[:N] = pop[si]
            fitness[:N] = fitness[si]

            # ---- Periodic eigenvector update ----
            if gen % self.eigen_period == 0 and N >= dim + 2 and dim <= 20:
                nt = max(dim + 1, int(self.eigen_frac * N))
                top = pop[:nt]
                centered = top - top.mean(axis=0)
                try:
                    cov = np.cov(centered, rowvar=False)
                    if cov.ndim == 2:
                        _, eigvecs = np.linalg.eigh(cov)
                except (np.linalg.LinAlgError, ValueError):
                    eigvecs = np.eye(dim)

            # ---- Adaptive p-best ----
            p = self.p_max - (self.p_max - self.p_min) * prog
            npb = max(2, int(p * N))

            # ---- Generate F and CR from SHADE memory ----
            ri = np.random.randint(0, self.H, N)

            F = M_F[ri] + 0.1 * np.random.standard_cauchy(N)
            bad = F <= 0
            while np.any(bad):
                F[bad] = M_F[ri[bad]] + 0.1 * np.random.standard_cauchy(int(bad.sum()))
                bad = F <= 0
            F = np.minimum(F, 1.0)

            CR = np.clip(M_CR[ri] + 0.1 * np.random.randn(N), 0.0, 1.0)

            # Late-stage CR reduction for tighter exploitation
            if prog > 0.75:
                CR *= max(0.05, 1.0 - 0.9 * (prog - 0.75) / 0.25)

            # ---- Assign strategies ----
            strats = np.random.choice(self.n_strat, N, p=strat_probs)

            # ---- Build mutation pool (pop + archive) ----
            if arc_len > 0:
                pool = np.vstack([pop[:N], archive[:arc_len]])
            else:
                pool = pop[:N]
            P = len(pool)

            # ================================================================
            # Vectorized mutation by strategy type
            # ================================================================
            mutants = np.empty((N, dim))

            for s_id in range(self.n_strat):
                idx_s = np.where(strats == s_id)[0]
                ns = len(idx_s)
                if ns == 0:
                    continue

                if s_id == 0:
                    # ---- Strategy 0: current-to-pbest/1 with archive ----
                    if N < 3 or P < 3:
                        sigma = rng * 0.1 * max(0.01, 1 - prog)
                        mutants[idx_s] = pop[idx_s] + sigma * np.random.randn(ns, dim)
                        continue
                    p_idx = np.random.randint(0, npb, ns)
                    r1 = np.random.randint(0, N, ns)
                    col = r1 == idx_s
                    r1[col] = (r1[col] + 1) % N
                    r2 = np.random.randint(0, P, ns)
                    col = (r2 < N) & ((r2 == idx_s) | (r2 == r1))
                    iters = 0
                    while np.any(col) and iters < 25:
                        r2[col] = np.random.randint(0, P, int(col.sum()))
                        col = (r2 < N) & ((r2 == idx_s) | (r2 == r1))
                        iters += 1
                    mutants[idx_s] = (pop[idx_s]
                                      + F[idx_s, None] * (pop[p_idx] - pop[idx_s])
                                      + F[idx_s, None] * (pop[r1] - pool[r2]))

                elif s_id == 1:
                    # ---- Strategy 1: current-to-best/1 with archive ----
                    if N < 2 or P < 2:
                        sigma = rng * 0.1 * max(0.01, 1 - prog)
                        mutants[idx_s] = pop[idx_s] + sigma * np.random.randn(ns, dim)
                        continue
                    r1 = np.random.randint(0, N, ns)
                    col = r1 == idx_s
                    r1[col] = (r1[col] + 1) % N
                    r2 = np.random.randint(0, P, ns)
                    col = (r2 < N) & ((r2 == idx_s) | (r2 == r1))
                    iters = 0
                    while np.any(col) and iters < 25:
                        r2[col] = np.random.randint(0, P, int(col.sum()))
                        col = (r2 < N) & ((r2 == idx_s) | (r2 == r1))
                        iters += 1
                    mutants[idx_s] = (pop[idx_s]
                                      + F[idx_s, None] * (best_pos - pop[idx_s])
                                      + F[idx_s, None] * (pop[r1] - pool[r2]))

                else:
                    # ---- Strategy 2: rand/1 (exploration) ----
                    if N < 4:
                        sigma = rng * 0.15 * max(0.01, 1 - prog)
                        mutants[idx_s] = pop[idx_s] + sigma * np.random.randn(ns, dim)
                        continue
                    r0 = np.random.randint(0, N, ns)
                    col = r0 == idx_s
                    r0[col] = (r0[col] + 1) % N

                    r1 = np.random.randint(0, N, ns)
                    col = (r1 == idx_s) | (r1 == r0)
                    iters = 0
                    while np.any(col) and iters < 25:
                        r1[col] = np.random.randint(0, N, int(col.sum()))
                        col = (r1 == idx_s) | (r1 == r0)
                        iters += 1

                    r2 = np.random.randint(0, N, ns)
                    col = (r2 == idx_s) | (r2 == r0) | (r2 == r1)
                    iters = 0
                    while np.any(col) and iters < 25:
                        r2[col] = np.random.randint(0, N, int(col.sum()))
                        col = (r2 == idx_s) | (r2 == r0) | (r2 == r1)
                        iters += 1

                    mutants[idx_s] = pop[r0] + F[idx_s, None] * (pop[r1] - pop[r2])

            if self.should_stop():
                break

            # ================================================================
            # Crossover: binomial with optional eigenvector rotation
            # ================================================================
            trials = pop[:N].copy()
            j_rand = np.random.randint(0, dim, N)

            use_eig = np.random.rand(N) < self.eigen_prob

            # Standard binomial crossover
            std_idx = np.where(~use_eig)[0]
            if len(std_idx) > 0:
                mask = np.random.rand(len(std_idx), dim) < CR[std_idx, None]
                mask[np.arange(len(std_idx)), j_rand[std_idx]] = True
                trials[std_idx] = np.where(mask, mutants[std_idx], pop[std_idx])

            # Eigenvector-guided crossover (rotate to eigen space, crossover, rotate back)
            eig_idx = np.where(use_eig)[0]
            if len(eig_idx) > 0:
                pop_rot = (eigvecs.T @ pop[eig_idx].T).T
                mut_rot = (eigvecs.T @ mutants[eig_idx].T).T
                mask = np.random.rand(len(eig_idx), dim) < CR[eig_idx, None]
                mask[np.arange(len(eig_idx)), j_rand[eig_idx]] = True
                child_rot = np.where(mask, mut_rot, pop_rot)
                trials[eig_idx] = (eigvecs @ child_rot.T).T

            # ================================================================
            # Boundary handling and evaluation
            # ================================================================
            trials = self.reflect(trials)
            tfit = self.evaluate_pop(trials)

            if self.should_stop():
                idx = int(np.argmin(tfit))
                if tfit[idx] < best_val:
                    best_val = float(tfit[idx])
                    best_pos = trials[idx].copy()
                return best_val, best_pos

            # ================================================================
            # Selection and adaptation
            # ================================================================
            improved = tfit < fitness[:N]

            if np.any(improved):
                imp = np.where(improved)[0]
                delta = fitness[imp] - tfit[imp]
                dsum = delta.sum()

                # Update SHADE memory (weighted Lehmer mean)
                if dsum > 0:
                    w = delta / dsum
                    sf = F[imp]
                    scr = CR[imp]
                    # Write to adaptive slots only (preserve terminal)
                    wp = mem_ptr % (self.H - 1)
                    M_F[wp] = float((w * sf ** 2).sum() / max((w * sf).sum(), 1e-30))
                    M_CR[wp] = float((w * scr).sum())
                    mem_ptr += 1

                # Strategy credit (normalized by fitness range for robustness)
                fit_range = fitness[N - 1] - fitness[0] if N > 1 else 1.0
                fit_range = max(fit_range, 1e-30)
                for j in range(len(imp)):
                    strat_succ[strats[imp[j]]] += delta[j] / fit_range

                # Archive replaced parents
                for idx in imp:
                    if arc_len < arc_cap:
                        archive[arc_len] = pop[idx]
                        arc_len += 1
                    else:
                        archive[np.random.randint(arc_cap)] = pop[idx]

            # Count all strategy trials
            for s_id in range(self.n_strat):
                strat_trials[s_id] += float(np.sum(strats == s_id))

            # Apply selection (use integer indexing since pop may be larger than N)
            imp_idx = np.where(improved)[0]
            pop[imp_idx] = trials[imp_idx]
            fitness[imp_idx] = tfit[imp_idx]

            # ---- Best tracking ----
            bi = int(np.argmin(fitness[:N]))
            if fitness[bi] < best_val:
                best_val = float(fitness[bi])
                best_pos = pop[bi].copy()
                stag = 0
            else:
                stag += 1

            # ================================================================
            # Strategy probability adaptation
            # ================================================================
            if gen % self.strat_period == 0 and gen > 0:
                rates = np.zeros(self.n_strat)
                for s_id in range(self.n_strat):
                    if strat_trials[s_id] > 0:
                        rates[s_id] = strat_succ[s_id] / strat_trials[s_id]
                rsum = rates.sum()
                if rsum > 0:
                    new_p = rates / rsum
                    strat_probs = self.strat_lr * new_p + (1 - self.strat_lr) * strat_probs
                strat_probs = np.maximum(strat_probs, 0.05)
                strat_probs /= strat_probs.sum()
                strat_succ[:] = 0
                strat_trials[:] = 0

            # ================================================================
            # Non-linear population size reduction (exponent > 1 for longer exploration)
            # ================================================================
            N_target = max(N_min, int(round(
                N_init - (N_init - N_min) * min(prog, 1.0) ** 1.3
            )))
            if N > N_target:
                keep = np.argsort(fitness[:N])[:N_target]
                pop[:N_target] = pop[keep]
                fitness[:N_target] = fitness[keep]
                N = N_target

                # Shrink effective archive
                arc_eff = max(N, int(self.arc_rate * N))
                if arc_len > arc_eff:
                    sel = np.random.choice(arc_len, arc_eff, replace=False)
                    archive[:arc_eff] = archive[sel]
                    arc_len = arc_eff

            if self.should_stop():
                break

            # ================================================================
            # Periodic diversity injection (discover new basins)
            # ================================================================
            if (gen % self.inject_period == 0 and gen > 0
                    and 0.25 < prog < 0.80 and N >= 8):
                n_inject = max(1, int(self.inject_frac * N))
                worst = np.argsort(fitness[:N])[-n_inject:]
                pop[worst] = self.random_population(n_inject)
                pop[worst] = self.clip(pop[worst])
                fit_inj = self.evaluate_pop(pop[worst])
                fitness[worst] = fit_inj
                for idx in worst:
                    if fitness[idx] < best_val:
                        best_val = float(fitness[idx])
                        best_pos = pop[idx].copy()
                if self.should_stop():
                    break

            # ================================================================
            # Two-phase stagnation restart
            # ================================================================
            stag_limit = self.stag_base + dim
            if stag >= stag_limit and prog < 0.92 and N >= 6:
                nr = max(1, int(self.restart_frac * N))
                worst = np.argsort(fitness[:N])[-nr:]

                if stag >= 2 * stag_limit:
                    # Severe: opposition-based jump with noise
                    pop[worst] = lb + ub - pop[worst] + 0.05 * rng * np.random.randn(nr, dim)
                else:
                    # Moderate: Gaussian restart around elite centroid
                    ne = max(2, N // 5)
                    centroid = pop[:ne].mean(axis=0)
                    sigma = rng * max(0.03, 0.25 * (1 - prog))
                    pop[worst] = centroid + sigma * np.random.randn(nr, dim)

                pop[worst] = self.reflect(pop[worst])
                fit_restart = self.evaluate_pop(pop[worst])
                fitness[worst] = fit_restart

                for idx in worst:
                    if fitness[idx] < best_val:
                        best_val = float(fitness[idx])
                        best_pos = pop[idx].copy()

                stag = 0
                if self.should_stop():
                    break

            # ================================================================
            # Multi-trial Gaussian local search (final 25% of budget)
            # ================================================================
            if prog > 0.75 and gen % 5 == 0 and N >= 2:
                # Adaptive sigma from elite spread
                ne = min(5, N)
                elite_std = np.std(pop[:ne], axis=0)
                sigma = np.maximum(elite_std * 0.3, rng * 0.0001)
                sigma *= max(0.01, (1.0 - prog) / 0.25)

                # Multi-trial: evaluate k perturbations per elite
                k = min(dim, 8)
                n_elites = min(2, N)
                for li in range(n_elites):
                    if self.should_stop():
                        break
                    candidates = pop[li] + sigma * np.random.randn(k, dim)
                    candidates = self.clip(candidates)
                    cvals = self.evaluate_pop(candidates)
                    if self.should_stop():
                        bc = int(np.argmin(cvals))
                        if cvals[bc] < best_val:
                            best_val = float(cvals[bc])
                            best_pos = candidates[bc].copy()
                        break
                    bc = int(np.argmin(cvals))
                    if cvals[bc] < fitness[li]:
                        pop[li] = candidates[bc]
                        fitness[li] = cvals[bc]
                        if cvals[bc] < best_val:
                            best_val = float(cvals[bc])
                            best_pos = candidates[bc].copy()

            gen += 1

        return best_val, best_pos
