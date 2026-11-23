"""Chargement des donnees brutes + enchainement niveaux 1-2."""
from pathlib import Path
import numpy as np, pandas as pd
from .iqr import iqr_filter_mean
from .negotiation import negotiate

CRITERIA = ["Aquifers and Surface Water", "Climate and Weather Conditions", "Ecological Sensitivity",
            "Geology", "Regulatory Compliance", "Settlements and Urban Centers",
            "Slope", "Topography", "Urban Development and Infrastructure Location"]
PROFILES = ["ENR", "GGIS", "CEUP", "ES"]          # fichiers *_G1..G4.csv
MODELS_ORIG = ["ChatGPT", "Gemini", "Copilot"]

def load_original(d, models=None) -> dict:
    """{(modele, profil): array (n=9, m)} depuis un dossier de fichiers {Modele}_G{1..4}.csv
    (donnees de Vahidnia ou collecte open source). models=None : detectes via *_G1.csv."""
    d = Path(d)
    models = models or sorted(p.name[:-len("_G1.csv")] for p in d.glob("*_G1.csv"))
    return {(a, e): pd.read_csv(d / f"{a}_G{i}.csv", header=None).values.astype(float)
            for a in models for i, e in enumerate(PROFILES, 1)}

def aggregate_levels(raw: dict, use_iqr: bool = True, profiles=None, k: float = 1.5) -> tuple:
    """raw {(a,e): (n,m)} -> (W_bar (n, L), profils). Eq. 4-5."""
    models = sorted({a for a, _ in raw}); profiles = profiles or sorted({e for _, e in raw})
    cols = []
    for e in profiles:
        per_model = [iqr_filter_mean(raw[(a, e)], k) if use_iqr else raw[(a, e)].mean(axis=1) for a in models]
        cols.append(np.mean(per_model, axis=0))
    return np.stack(cols, axis=1), list(profiles)

def run_consensus(raw: dict, alpha=0.5, eps=1e-4, t_max=100, use_iqr=True, profiles=None, k=1.5):
    W_bar, profs = aggregate_levels(raw, use_iqr, profiles, k)
    res = negotiate(W_bar, alpha, eps, t_max)
    w_star = res.w_star
    return {"W_bar": W_bar, "profiles": profs, "result": res,
            "w_star": w_star, "w_hat": w_star / w_star.sum()}
