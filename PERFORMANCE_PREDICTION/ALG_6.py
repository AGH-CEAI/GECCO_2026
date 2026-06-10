import numpy as np
from base_algorithm import BaseAlgorithm

class Algorithm(BaseAlgorithm):

    def __init__(self, gnbg):
        super().__init__(gnbg)
        
    def run(self):
        dim = self.dim
        lb, ub = self.lb, self.ub
        range_span = ub[0] - lb[0]
        
        best_val = np.inf
        best_pos = None
        
        lambda_default = 4 + int(3 * np.log(dim))
        
        n_restarts = 0
        fe_large = 0
        fe_small = 0

        chiN = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))

        while not self.should_stop():
            start_fe = self.gnbg.FE

            # --- INTELLIGENT BIPOP REGIME ROUTING ---
            if fe_large <= fe_small or n_restarts == 0:
                # --- LARGE REGIME: Global Search with Oppositional Jumpstart ---
                mult = min(n_restarts, 9) 
                lambda_ = lambda_default * (2 ** mult)
                lambda_ = int(min(lambda_, 2048))
                sigma = 0.3 * range_span  
                n_restarts += 1
                is_large = True
                
                # Oppositional Jumpstart: Evaluate 5 random points and their exact opposites
                jump_pop = self.random_population(5)
                jump_opp = self.reflect(lb + ub - jump_pop)
                pool = np.vstack((jump_pop, jump_opp))
                
                pool_fit = self.evaluate_pop(pool)
                if self.should_stop() and np.all(np.isinf(pool_fit)):
                    break
                    
                # Track global best from jumpstart
                valid_mask = ~np.isinf(pool_fit)
                if np.any(valid_mask):
                    min_idx = int(np.argmin(pool_fit))
                    if pool_fit[min_idx] < best_val:
                        best_val = pool_fit[min_idx]
                        best_pos = pool[min_idx].copy()
                        
                # Start distribution at the absolute best jumpstart point
                best_jump_idx = int(np.argmin(pool_fit))
                xmean = pool[best_jump_idx].copy()

            else:
                # --- SMALL REGIME: Localized Search ---
                fraction = np.random.rand() ** 2
                max_small = 0.5 * lambda_default * (2 ** min(n_restarts - 1, 9))
                lambda_ = int(lambda_default * (max_small / lambda_default) ** fraction)
                lambda_ = max(lambda_default, lambda_)  
                lambda_ = int(min(lambda_, 2048)) 
                is_large = False
                
                # Elite Exploitation vs. Random Scouting
                if np.random.rand() < 0.5 and best_pos is not None:
                    # Laser-focus on the global best to squeeze out the final decimals
                    xmean = best_pos.copy()
                    sigma = 1e-4 * range_span * np.random.rand()
                else:
                    # Random scouting for hidden funnels
                    xmean = self.random_position()
                    sigma = 0.05 * range_span * np.random.rand()

            # Ensure lambda is even for mirrored sampling
            if lambda_ % 2 != 0:
                lambda_ += 1

            # --- INITIALIZE CMA-ES STATE ---
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = np.sum(weights)**2 / np.sum(weights**2)
            
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)
            
            history_fit = []
            gen = 0

            # --- INNER LOOP: EVOLUTIONARY GENERATION ---
            while not self.should_stop():
                gen += 1
                
                # --- Mirrored Sampling ---
                half_lam = lambda_ // 2
                z_half = np.random.randn(half_lam, dim)
                z = np.vstack((z_half, -z_half))
                
                # Fast Broadcasting
                y = (z * D) @ B.T
                X = xmean + sigma * y
                
                if np.any(np.isnan(X)): 
                    break 
                    
                X_repaired = self.clip(X)
                fit = self.evaluate_pop(X_repaired)
                
                if self.should_stop() and np.all(np.isinf(fit)):
                    break

                valid_mask = ~np.isinf(fit)
                if np.any(valid_mask):
                    min_idx = int(np.argmin(fit))
                    if fit[min_idx] < best_val:
                        best_val = fit[min_idx]
                        best_pos = X_repaired[min_idx].copy()

                if self.should_stop():
                    break

                sort_idxs = np.argsort(fit)
                X_sorted = X_repaired[sort_idxs]
                fit_sorted = fit[sort_idxs]
                
                xmean_old = xmean.copy()
                y_selected = (X_sorted[:mu] - xmean_old) / sigma
                xmean = xmean_old + sigma * np.sum(weights[:, None] * y_selected, axis=0)
                
                y_mean = (xmean - xmean_old) / sigma
                z_mean = (y_mean @ B) * (1.0 / D)
                
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ z_mean)
                
                hsig_val = np.linalg.norm(ps) / np.sqrt(max(1e-10, 1 - (1 - cs)**(2 * gen)))
                hsig = 1.0 if hsig_val < (1.4 + 2 / (dim + 1)) else 0.0
                
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_mean
                
                # Vectorized Rank-mu update
                artmp = y_selected
                rank_mu_update = (weights[:, None] * artmp).T @ artmp
                
                C = (1 - c1 - cmu) * C \
                    + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
                    + cmu * rank_mu_update
                
                C = (C + C.T) / 2.0
                
                exp_arg = (cs / damps) * (np.linalg.norm(ps) / chiN - 1)
                exp_arg = max(min(exp_arg, 20.0), -20.0) 
                
                sigma *= np.exp(exp_arg)
                sigma = max(min(sigma, 1e5 * range_span), 1e-15)
                
                try:
                    D2, B = np.linalg.eigh(C + np.eye(dim) * 1e-12)
                    if np.any(np.isnan(D2)) or np.max(D2) > 1e15:
                        break 
                    
                    D2 = np.maximum(D2, 1e-18) 
                    D = np.sqrt(D2)
                except np.linalg.LinAlgError:
                    break
                    
                # --- CHECK RESTART CRITERIA ---
                history_fit.append(fit_sorted[0])
                history_len = 10 + int(30 * dim / lambda_)
                if len(history_fit) > history_len:
                    history_fit.pop(0)
                    
                if len(history_fit) >= history_len and (max(history_fit) - min(history_fit)) < 1e-12:
                    break
                if sigma * np.max(D) < 1e-12 or np.max(D) / np.min(D) > 1e7:
                    break
                if np.all(sigma * D < 1e-12):
                    break

            # Update Budget Tracking
            fe_consumed = self.gnbg.FE - start_fe
            if is_large:
                fe_large += fe_consumed
            else:
                fe_small += fe_consumed

        return best_val, best_pos