import numpy as np
from base_algorithm import BaseAlgorithm

class Algorithm(BaseAlgorithm):

    def __init__(self, gnbg):
        super().__init__(gnbg)
        # IPOP hyperparameter
        self.pop_multiplier = 2.0
        
    def run(self):
        dim = self.dim
        lb, ub = self.lb, self.ub
        range_span = ub[0] - lb[0]  # Assuming bounds are symmetrical per instructions
        
        # Track global best explicitly
        best_val = np.inf
        best_pos = None
        
        # IPOP dynamic parameter: starting population size based on dimension
        lambda_ = 4 + int(3 * np.log(dim))
        
        # Expected norm of a N(0, I) vector
        chiN = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))

        while not self.should_stop():

            # 1. Strategy parameters
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = np.sum(weights)**2 / np.sum(weights**2)
            
            # 2. Adaptation learning rates
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3)**2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            
            # 3. Dynamic state variables
            xmean = self.random_position()
            sigma = 0.25 * range_span  # Initial step size covering ~25% of the space
            
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)
            
            # Tracking for stagnation restarts
            history_fit = []
            gen = 0
            

            while not self.should_stop():
                gen += 1
                
                # Sample lambda offspring from N(xmean, sigma^2 * C)
                z = np.random.randn(lambda_, dim)
                y = z @ np.diag(D) @ B.T
                X = xmean + sigma * y
                
                # Apply boundary handling BEFORE evaluating
                X_repaired = self.reflect(X)
                
                # Evaluate offspring
                fit = self.evaluate_pop(X_repaired)
                
                if self.should_stop() and np.all(np.isinf(fit)):
                    break
                    
                # Track global best
                valid_mask = ~np.isinf(fit)
                if np.any(valid_mask):
                    min_idx = int(np.argmin(fit))
                    if fit[min_idx] < best_val:
                        best_val = fit[min_idx]
                        best_pos = X_repaired[min_idx].copy()

                # Sort offspring by fitness
                sort_idxs = np.argsort(fit)
                X_sorted = X_repaired[sort_idxs]
                fit_sorted = fit[sort_idxs]
                
                # Selection & Recombination
                xmean_old = xmean.copy()
                
                # Calculate the actual steps taken *after* repair to inform the covariance 
                # matrix about the walls, preventing the distribution from flattening out.
                y_selected = (X_sorted[:mu] - xmean_old) / sigma
                xmean = xmean_old + sigma * np.sum(weights[:, None] * y_selected, axis=0)
                
                # --- UPDATE EVOLUTION PATHS ---
                y_mean = (xmean - xmean_old) / sigma
                z_mean = y_mean @ B @ np.diag(1/D)
                
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ z_mean)
                
                # hsig: stalling mechanism to prevent too fast pc update if ps is large
                hsig_val = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * gen))
                hsig = 1.0 if hsig_val < (1.4 + 2 / (dim + 1)) else 0.0
                
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_mean
                
                # --- UPDATE COVARIANCE MATRIX ---
                artmp = y_selected
                C = (1 - c1 - cmu) * C \
                    + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
                    + cmu * (artmp.T @ np.diag(weights) @ artmp)
                
                # Enforce strict symmetry mathematically
                C = np.triu(C) + np.triu(C, 1).T
                
                # Update step-size (sigma)
                sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
                
                # Eigendecomposition to prepare for the next generation's sampling
                try:
                    D2, B = np.linalg.eigh(C)
                    # Force strict positive definiteness
                    D2 = np.maximum(D2, 1e-18)
                    D = np.sqrt(D2)
                except np.linalg.LinAlgError:
                    # Failsafe for numerical matrix collapse
                    break
                    
                # ==========================================
                # --- CHECK RESTART CRITERIA (STAGNATION) --
                # ==========================================
                history_fit.append(fit_sorted[0])
                history_len = 10 + int(30 * dim / lambda_)
                if len(history_fit) > history_len:
                    history_fit.pop(0)
                    
                # Trigger 1: Stagnation - The difference in recent best fitnesses is microscopic
                if len(history_fit) >= history_len and (max(history_fit) - min(history_fit)) < 1e-12:
                    break
                    
                # Trigger 2: Extreme step size or extreme landscape conditioning
                if sigma * np.max(D) < 1e-12 or np.max(D) / np.min(D) > 1e7:
                    break
                    
                # Trigger 3: "No effect axis" - Sigma is so small that xmean cannot physically move
                if np.all(sigma * D < 1e-12):
                    break

            # End of inner loop -> Restart triggered. 
            # Double the population size to shift from local to global search.
            lambda_ = int(lambda_ * self.pop_multiplier)

        return best_val, best_pos