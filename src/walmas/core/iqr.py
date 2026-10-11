"""Niveau 1 : filtrage de Tukey (eq. 3-4)."""
import numpy as np

def iqr_filter_mean(w: np.ndarray, k: float = 1.5, return_mask: bool = False):
    """w : (n, m) = n criteres x m agents. Retourne la moyenne par critere
    apres rejet des valeurs hors [Q1-k*IQR, Q3+k*IQR] (quantiles lineaires)."""
    w = np.asarray(w, dtype=float)
    q1, q3 = np.percentile(w, [25, 75], axis=1, method="linear")
    iqr = q3 - q1
    lo, hi = (q1 - k * iqr)[:, None], (q3 + k * iqr)[:, None]
    mask = (w >= lo) & (w <= hi)          # jamais vide : la mediane est toujours conservee
    means = (w * mask).sum(axis=1) / mask.sum(axis=1)
    return (means, mask) if return_mask else means
