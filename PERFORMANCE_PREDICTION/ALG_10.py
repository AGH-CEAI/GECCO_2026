

import numpy as np
from base_algorithm import BaseAlgorithm     # sibling file in algorithms/


class Algorithm(BaseAlgorithm):


    def __init__(self, gnbg, n_particles: int = 40, w: float = 0.7,
                 c1: float = 1.5, c2: float = 1.5,
                 vmax_frac: float = 0.2, border: str = "reflect"):
        super().__init__(gnbg)
        self.n_particles = n_particles
        self.w           = w
        self.c1          = c1
        self.c2          = c2
        self.vmax_frac   = vmax_frac
        self.border      = border

    # ------------------------------------------------------------------ #
    def run(self):

        n   = self.n_particles
        dim = self.dim
        lb, ub = self.lb, self.ub

        # ── 1. Velocity clamping limits ───────────────────────────────── #
        v_max = self.vmax_frac * (ub - lb)     # shape (dim,)

        # ── 2. Initialise positions and velocities ────────────────────── #
        pos = self.random_population(n)                       # (n, dim)
        vel = np.random.uniform(-v_max, v_max, (n, dim))     # (n, dim)

        # ── 3. Evaluate initial swarm ─────────────────────────────────── #
        fit = self.evaluate_pop(pos)

        # Personal bests
        pbest_pos = pos.copy()
        pbest_fit = fit.copy()

        # Global best
        gbest_idx = int(np.argmin(pbest_fit))
        gbest_fit = pbest_fit[gbest_idx]
        gbest_pos = pbest_pos[gbest_idx].copy()

        # ── 4. Main swarm loop ────────────────────────────────────────── #
        while not self.should_stop():

            # ── 4a. Random coefficients (unique per particle per dim) ── #
            r1 = np.random.rand(n, dim)
            r2 = np.random.rand(n, dim)

            # ── 4b. Velocity update ───────────────────────────────────── #
            vel = (self.w * vel
                   + self.c1 * r1 * (pbest_pos - pos)
                   + self.c2 * r2 * (gbest_pos - pos))

            # ── 4c. Velocity clamping ─────────────────────────────────── #
            #  Essential: prevents particles from shooting past the search space.
            vel = np.clip(vel, -v_max, v_max)

            # ── 4d. Position update ───────────────────────────────────── #
            pos = pos + vel

            # ── 4e. Boundary handling for POSITIONS ───────────────────── #
            pos, vel = self._apply_border(pos, vel, v_max)

            # ── 4f. Evaluate new positions ────────────────────────────── #
            if self.should_stop():
                break

            fit = self.evaluate_pop(pos)

            # ── 4g. Update personal bests ──────────────────────────────── #
            improved = fit < pbest_fit
            pbest_pos[improved] = pos[improved].copy()
            pbest_fit[improved] = fit[improved]

            # ── 4h. Update global best ─────────────────────────────────── #
            best_idx = int(np.argmin(pbest_fit))
            if pbest_fit[best_idx] < gbest_fit:
                gbest_fit = pbest_fit[best_idx]
                gbest_pos = pbest_pos[best_idx].copy()

        return gbest_fit, gbest_pos

    # ------------------------------------------------------------------ #
    # PSO-specific boundary handler (handles velocity too)
    # ------------------------------------------------------------------ #
    def _apply_border(self, pos: np.ndarray, vel: np.ndarray,
                      v_max: np.ndarray):

        lb, ub = self.lb, self.ub
        pos    = pos.copy()
        vel    = vel.copy()

        if self.border == "clip":
            # Positions: clamp
            # Velocities: zero the component that hit the wall
            below = pos < lb
            above = pos > ub
            vel[below | above] = 0.0        # kill velocity at the wall
            pos = np.clip(pos, lb, ub)

        elif self.border == "reinit":
            # Positions: random redraw per violated component
            for d in range(self.dim):
                mask = (pos[:, d] < lb[d]) | (pos[:, d] > ub[d])
                pos[mask, d] = np.random.uniform(lb[d], ub[d], mask.sum())
                vel[mask, d] = 0.0          # reset velocity for that dimension

        else:                               # default: 'reflect'
            # ── Reflection with velocity damping (damp = 0.5) ─────────── #
            damp = 0.5
            for _ in range(10):             # cap at 10 bounces per step
                below = pos < lb
                above = pos > ub
                if not (below.any() or above.any()):
                    break

                # Reflect positions
                pos[below] = 2.0 * lb[np.where(below)[1]] - pos[below]
                pos[above] = 2.0 * ub[np.where(above)[1]] - pos[above]

                # Reverse & damp velocity
                vel[below | above] *= -damp

            # Safety clamp (handles edge cases with repeated bounces)
            pos = np.clip(pos, lb, ub)
            vel = np.clip(vel, -v_max, v_max)

        return pos, vel
