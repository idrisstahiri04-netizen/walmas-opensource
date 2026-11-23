"""Indicateurs d'evaluation (protocole du rapport, points 0-7). Fonctions pures -> DataFrames / nombres."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from .core.pipeline import CRITERIA, PROFILES, aggregate_levels, run_consensus
from .core.iqr import iqr_filter_mean
from .core.kendall import kendall_w, kendall_w_legacy, ranks
from .core.negotiation import negotiate


def models_of(raw): return sorted({a for a, _ in raw})

def sub(raw, model): return {k: v for k, v in raw.items() if k[0] == model}

def rank_vec(w): return ranks(np.asarray(w, float)[:, None])[:, 0]


# ---- 0. Filtre IQR ----
def iqr_table(raw) -> pd.DataFrame:
    """Une ligne par (modele, profil, critere) : nb de poids, nb rejetes, IQR nul (Q1 == Q3)."""
    rows = []
    for (a, e), w in raw.items():
        _, mask = iqr_filter_mean(w, return_mask=True)
        q1, q3 = np.percentile(w, [25, 75], axis=1, method="linear")   # memes quantiles que le filtre
        for i, c in enumerate(CRITERIA):
            rows.append(dict(model=a, profile=e, criterion=c, n=w.shape[1],
                             rejected=int((~mask[i]).sum()), zero_iqr=bool(q1[i] == q3[i])))
    return pd.DataFrame(rows)

def iqr_effect(raw, alpha=0.5) -> float:
    """max |w* avec IQR - w* sans IQR|."""
    return float(np.abs(run_consensus(raw, alpha)["w_star"] - run_consensus(raw, alpha, use_iqr=False)["w_star"]).max())


# ---- 1. Distribution ----
def distribution(raw) -> pd.DataFrame:
    """Table longue : moyenne / ecart-type (ddof=1) des poids bruts et moyenne filtree IQR."""
    rows = []
    for (a, e), w in raw.items():
        for i, c in enumerate(CRITERIA):
            rows.append(dict(model=a, profile=e, criterion=c, mean=w[i].mean(),
                             std=w[i].std(ddof=1), mean_iqr=iqr_filter_mean(w[i:i + 1])[0]))
    return pd.DataFrame(rows)

def mean_std(raw) -> pd.DataFrame:
    """Ecart-type entre les m agents, moyenne sur criteres x profils, par modele (4 definitions)."""
    out = {}
    for a in models_of(raw):
        ws = [raw[(a, e)] for e in PROFILES]
        kept = [np.where(iqr_filter_mean(w, return_mask=True)[1], w, np.nan) for w in ws]
        out[a] = {f"ddof{d}_{t}": np.mean([np.nanstd(x, axis=1, ddof=d).mean() for x in xs])
                  for t, xs in [("avant_iqr", ws), ("apres_iqr", kept)] for d in (0, 1)}
    return pd.DataFrame(out).T.rename_axis("model")


# ---- 2. Similarite entre modeles ----
def model_vectors(raw, alpha=0.5) -> dict:
    """Vecteur de criteres par modele : w* (= moyenne sur profils des moyennes IQR) et moyenne brute."""
    return {a: {"w_star": run_consensus(sub(raw, a), alpha)["w_star"],
                "moy_brute": aggregate_levels(sub(raw, a), use_iqr=False)[0].mean(axis=1)}
            for a in models_of(raw)}

def model_similarity(raw, alpha=0.5) -> pd.DataFrame:
    v, ms = model_vectors(raw, alpha), models_of(raw)
    rows = [dict(model_a=a, model_b=b, **{f"rho_{k}": spearmanr(v[a][k], v[b][k])[0] for k in v[a]})
            for i, a in enumerate(ms) for b in ms[i + 1:]]
    return pd.DataFrame(rows)


# ---- 3. Comparaison a une reference ----
def compare(w, w_ref) -> tuple:
    """-> ({rho, tau, mae}, table par critere)."""
    w, w_ref = np.asarray(w, float), np.asarray(w_ref, float)
    stats = dict(spearman_rho=float(spearmanr(w, w_ref)[0]), kendall_tau=float(kendalltau(w, w_ref)[0]),
                 mae=float(np.abs(w - w_ref).mean()))
    table = pd.DataFrame(dict(criterion=CRITERIA, w_star=w, w_star_ref=w_ref, rank=rank_vec(w),
                              rank_ref=rank_vec(w_ref), diff=w - w_ref))
    return stats, table


# ---- 4. Convergence ----
def convergence(W_bar, alphas, eps=1e-4, t_max=100) -> tuple:
    """-> (table par alpha, {alpha: NegotiationResult})."""
    res = {a: negotiate(W_bar, a, eps, t_max) for a in alphas}
    df = pd.DataFrame([dict(alpha=a, iterations=r.iterations, W_initial=r.w_history[0],
                            W_final=r.w_history[-1], converged=r.converged) for a, r in res.items()])
    return df, res

def kendall_initial(W_bar) -> dict:
    return dict(standard=kendall_w(W_bar), legacy=kendall_w_legacy(W_bar))


# ---- 5. Sensibilite aux profils ----
def profile_sensitivity(raw, alpha=0.5) -> pd.DataFrame:
    """Colonnes : w* complet puis w* sans chaque profil (ecarts via max_abs_delta)."""
    profs = sorted({e for _, e in raw}, key=lambda e: PROFILES.index(e) if e in PROFILES else len(PROFILES))
    df = pd.DataFrame({"complet": run_consensus(raw, alpha, profiles=profs)["w_star"]}, index=CRITERIA)
    for e in profs:
        df[f"sans_{e}"] = run_consensus(raw, alpha, profiles=[p for p in profs if p != e])["w_star"]
    return df.rename_axis("criterion")

def max_abs_delta(sens: pd.DataFrame) -> pd.Series:
    return sens.drop(columns="complet").sub(sens["complet"], axis=0).abs().max()


# ---- 6. Stabilite (proxy : bootstrap sur les agents, une seule serie de graines) ----
def bootstrap(raw, B=1000, seed=0, alpha=0.5) -> pd.DataFrame:
    """Reechantillonne les m colonnes avec remise dans chaque (modele, profil), relance tout le pipeline."""
    rng = np.random.default_rng(seed)
    w0 = run_consensus(raw, alpha)["w_star"]; r0 = rank_vec(w0)
    reps = np.array([run_consensus({k: w[:, rng.integers(0, w.shape[1], w.shape[1])] for k, w in raw.items()},
                                   alpha)["w_star"] for _ in range(B)])
    keep = np.array([rank_vec(w) for w in reps]) == r0
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    return pd.DataFrame(dict(criterion=CRITERIA, w_star=w0, rank=r0, ic95_bas=lo, ic95_haut=hi,
                             part_rang_conserve=keep.mean(axis=0)))


# ---- 7. Reference sans negociation ----
def no_negotiation_gap(out) -> float:
    """max |w* - moyenne simple des colonnes de W_bar| (attendu ~1e-12)."""
    return float(np.abs(out["w_star"] - out["W_bar"].mean(axis=1)).max())


def final_weights(out) -> pd.DataFrame:
    df = pd.DataFrame(dict(criterion=CRITERIA, w_star=out["w_star"], w_hat=out["w_hat"]))
    df.insert(0, "rank", rank_vec(out["w_star"]))
    return df.sort_values("rank").reset_index(drop=True)


# ---- 8. Biais de position (necessite raw/responses.jsonl de la collecte niveau 0) ----
def position_bias(jsonl_path) -> pd.DataFrame:
    """Table longue (modele, profil, agent, position 1..9 dans le prompt, poids) + rho de Spearman
    par reponse entre poids et anteriorite (rho > 0 : les premiers criteres recoivent plus)."""
    import json
    from .llm.agents import parse_weights
    rows = []
    for line in open(jsonl_path):
        r = json.loads(line)
        if not r["valid"]: continue
        order = r.get("order") or CRITERIA            # collecte sans shuffle : ordre CRITERIA
        w = parse_weights(r["response"], order)
        rho = spearmanr(w, -np.arange(len(w)))[0] if len(set(w)) > 1 else np.nan
        rows += [dict(model=r["model"], profile=r["profile"], agent=r["agent"], position=p + 1,
                      criterion=c, weight=v, rho=rho) for p, (c, v) in enumerate(zip(order, w))]
    return pd.DataFrame(rows)

def position_summary(pb: pd.DataFrame) -> pd.DataFrame:
    per = pb.drop_duplicates(["model", "profile", "agent"])
    return per.groupby("model").agg(rho_moyen=("rho", "mean"), part_rho_positif=("rho", lambda x: (x > 0).mean()),
                                    n_reponses=("rho", "size"))

def criterion_position_r2(pb: pd.DataFrame) -> pd.DataFrame:
    """R^2 de la regression du poids sur le critere, la position, et les deux (indicatrices), par modele.
    Interpretable seulement si l'ordre varie entre agents (collecte --shuffle) : sinon les deux sont confondus."""
    def r2(d, cols):
        X = np.column_stack([np.ones(len(d))] + [pd.get_dummies(d[c], drop_first=True).values.astype(float) for c in cols])
        y = d.weight.values.astype(float); b = np.linalg.lstsq(X, y, rcond=None)[0]
        return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    groups = list(pb.groupby("model")) + [("ensemble", pb)]
    return pd.DataFrame([dict(model=m, R2_critere=r2(d, ["criterion"]), R2_position=r2(d, ["position"]),
                              R2_les_deux=r2(d, ["criterion", "position"])) for m, d in groups]).set_index("model")
