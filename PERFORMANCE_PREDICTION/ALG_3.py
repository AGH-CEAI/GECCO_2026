
import numpy as np
from base_algorithm import BaseAlgorithm     # sibling file in algorithms/


class Algorithm(BaseAlgorithm):

    def __init__(self, gnbg, pop_size: int = 50, F: float = 0.8,
                 CR: float = 0.9, border: str = "clip"):
        super().__init__(gnbg)
        self.pop_size = pop_size
        self.F        = F
        self.CR       = CR
        self.border   = border          # 'clip' | 'reflect' | 'reinit'

    # ------------------------------------------------------------------ #
    def run(self):

        dim      = self.dim
        lb, ub   = self.lb, self.ub

        # ── 1. Initialise population uniformly inside bounds ─────────── #
        pop     = self.random_population(self.pop_size)  # (N, dim)
        fitness = self.evaluate_pop(pop)                 # (N,) — see base class

        # Track the global best (GNBG only stores the best VALUE, not position)
        best_idx  = int(np.argmin(fitness))
        best_val  = fitness[best_idx]
        best_pos  = pop[best_idx].copy()

        # ── 2. Main generational loop ─────────────────────────────────── #
        while not self.should_stop():           # checks FE budget + threshold

            for i in range(self.pop_size):

                # ── Early-exit check inside the inner loop ────────────── #
                if self.should_stop():
                    break

                # ── Mutation: DE/rand/1 ───────────────────────────────── #
                # Pick 3 distinct indices ≠ i
                candidates = [j for j in range(self.pop_size) if j != i]
                a, b, c = pop[np.random.choice(candidates, 3, replace=False)]

                mutant = a + self.F * (b - c)


                if self.border == "reflect":
                    mutant = self.reflect(mutant)
                elif self.border == "reinit":
                    mutant = self.random_reinit(mutant)
                else:                                   # default: clip
                    mutant = self.clip(mutant)

                # ── Crossover: binomial ───────────────────────────────── #
                cross_mask = np.random.rand(dim) < self.CR
                # Guarantee at least one component comes from mutant
                if not cross_mask.any():
                    cross_mask[np.random.randint(dim)] = True

                trial = np.where(cross_mask, mutant, pop[i])

                # ── Evaluate trial solution ───────────────────────────── #
                trial_fit = self.evaluate(trial)   # single call → 1 FE used

                # ── Greedy selection ──────────────────────────────────── #
                if trial_fit < fitness[i]:
                    pop[i]     = trial
                    fitness[i] = trial_fit

                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_pos = trial.copy()


        return best_val, best_pos
