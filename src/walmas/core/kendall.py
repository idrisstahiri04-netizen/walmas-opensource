"""Coefficient de concordance de Kendall (eq. 6)."""
import numpy as np
from scipy.stats import rankdata

def ranks(M: np.ndarray) -> np.ndarray:
    """Rang 1 = poids le plus fort, rangs moyens en cas d'egalite, par colonne."""
    return np.apply_along_axis(lambda c: rankdata(-c, method="average"), 0, np.asarray(M, float))

def kendall_w(M: np.ndarray, tie_correction: bool = False) -> float:
    """M : (n criteres, L profils). Formule standard 12S / (L^2 (n^3 - n)).
    tie_correction=True ajoute la correction d'egalites (Siegel & Castellan)."""
    n, L = M.shape
    r = ranks(M)
    R = r.sum(axis=1)
    S = ((R - L * (n + 1) / 2) ** 2).sum()
    if not tie_correction:
        return float(12 * S / (L**2 * (n**3 - n)))
    T = 0.0
    for j in range(L):
        _, cnt = np.unique(r[:, j], return_counts=True)
        T += (cnt**3 - cnt).sum()
    return float(12 * S / (L**2 * (n**3 - n) - L * T))

def kendall_w_legacy(M: np.ndarray) -> float:
    """Variante du script original (1 - 12S/..., S = variance intra-critere, egalites par ordre de ligne).
    Conservee UNIQUEMENT pour reproduire/expliquer l'ecart de la note 23 du rapport."""
    n, L = M.shape
    rk = np.argsort(np.argsort(-M, axis=0), axis=0) + 1
    S = np.sum((rk - rk.mean(axis=1)[:, None]) ** 2)
    return float(1 - 12 * S / (L**2 * (n**3 - n)))
