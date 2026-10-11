"""Niveau 2 : negociation iterative (eq. 7-9)."""
from dataclasses import dataclass
import numpy as np
from .kendall import kendall_w

@dataclass
class NegotiationResult:
    M_history: list      # M^(0..T), chacune (n, L)
    w_history: list      # W_K^(0..T)
    converged: bool
    iterations: int
    @property
    def M_final(self): return self.M_history[-1]
    @property
    def w_star(self):  return self.M_final.mean(axis=1)   # eq. 10

def negotiate(M0: np.ndarray, alpha: float = 0.5, eps: float = 1e-4, t_max: int = 100,
              tie_correction: bool = False) -> NegotiationResult:
    M = np.asarray(M0, float).copy()
    Ms, Ws = [M.copy()], [kendall_w(M, tie_correction)]
    converged = False
    for _ in range(t_max):
        M = alpha * M + (1 - alpha) * M.mean(axis=1, keepdims=True)   # eq. 7-8
        Ms.append(M.copy()); Ws.append(kendall_w(M, tie_correction))
        if abs(Ws[-1] - Ws[-2]) < eps:                                  # eq. 9
            converged = True; break
    return NegotiationResult(Ms, Ws, converged, len(Ms) - 1)
