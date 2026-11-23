"""Evaluation WALMAS : tables CSV, figures PNG et summary.md dans results/NAME/.

Usage : python scripts/evaluate.py --data data/original --name original [--reference DIR] [--out results] [--boot 1000]
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from walmas import analysis as an
from walmas.core.pipeline import CRITERIA, PROFILES, load_original, run_consensus

LABELS = dict(zip(CRITERIA, ["Eaux (aquifères)", "Climat", "Sensibilité écologique", "Géologie",
                             "Conformité réglementaire", "Agglomérations", "Pente", "Topographie",
                             "Urbanisation / infra."]))
PROFILE_NAMES = {"ENR": "ENR (environnement)", "GGIS": "GGIS (géographie/SIG)",
                 "CEUP": "CEUP (génie civil/urbanisme)", "ES": "ES (économie/société)"}
PCOL = dict(zip(PROFILES, ["#2a9d8f", "#e9c46a", "#264653", "#e76f51"]))
ALPHAS = [0.3, 0.5, 0.7]
plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "figure.dpi": 120, "savefig.bbox": "tight"})


def mcol(models): return {a: plt.cm.tab10(i) for i, a in enumerate(models)}

def save(fig, path): fig.savefig(path); plt.close(fig)

def fig_heatmaps(dist, models, path):
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 6), sharey=True, squeeze=False)
    for ax, a in zip(axes[0], models):
        M = dist[dist.model == a].pivot(index="criterion", columns="profile", values="mean").loc[CRITERIA, PROFILES]
        im = ax.imshow(M.values, cmap="YlGnBu", vmin=1, vmax=9)
        for (i, j), v in np.ndenumerate(M.values):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white" if v > 5.5 else "black")
        ax.set_xticks(range(len(PROFILES)), PROFILES); ax.set_yticks(range(9), [LABELS[c] for c in CRITERIA])
        ax.set_title(a)
    fig.colorbar(im, ax=axes[0], shrink=0.8, label="Poids moyen (1-9)")
    fig.suptitle("Poids moyens bruts par critère et profil")
    save(fig, path)

def fig_final(boot, path):
    b = boot.sort_values("w_star")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([LABELS[c] for c in b.criterion], b.w_star, color="#457b9d",
            xerr=[b.w_star - b.ic95_bas, b.ic95_haut - b.w_star], capsize=4)
    for y, v in enumerate(b.w_star): ax.text(0.1, y, f"{v:.2f}", va="center", color="white", fontweight="bold")
    ax.set_xlabel("w* (échelle 1-9)"); ax.set_xlim(0, 9)
    ax.set_title("Poids finaux w* (α=0.5)\nbarres : IC 95 % — proxy : bootstrap sur les agents, une seule série de graines",
                 fontsize=13)
    save(fig, path)

def fig_compare(w, w_ref, per_model, models, path):
    series = {"Données": (w, "#333333"), "Référence": (w_ref, "#adb5bd"),
              **{a: (v, c) for (a, v), c in zip(per_model.items(), mcol(models).values())}}
    x, k = np.arange(9), len(series); width = 0.8 / k
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (name, (v, c)) in enumerate(series.items()):
        ax.bar(x + (i - (k - 1) / 2) * width, v, width, label=name, color=c,
               hatch="//" if name in models else None, edgecolor="white")
    ax.set_xticks(x, [LABELS[c] for c in CRITERIA], rotation=30, ha="right")
    ax.set_ylabel("w* (échelle 1-9)"); ax.set_title("Comparaison des poids w* avec la référence")
    ax.legend(ncol=k, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    save(fig, path)

def fig_wk(res, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for a, r in res.items():
        ax.plot(range(len(r.w_history)), r.w_history, "o-", lw=2, label=f"α = {a} ({r.iterations} itér.)")
    ax.set_xlabel("Itération t"); ax.set_ylabel("W de Kendall"); ax.set_ylim(0, 1.05)
    ax.set_title("Évolution de la concordance W_K(t)"); ax.legend(); ax.grid(alpha=0.3)
    save(fig, path)

def fig_iter(grid, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(grid.alpha, grid.iterations, "o-", lw=2, color="#264653")
    ax.set_xlabel("α"); ax.set_ylabel("Nombre d'itérations")
    ax.set_title("Itérations jusqu'à convergence selon α"); ax.grid(alpha=0.3)
    save(fig, path)

def fig_slope(r, profs, path, crit="Slope"):
    i = CRITERIA.index(crit); traj = np.array([M[i] for M in r.M_history])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, e in enumerate(profs):
        ax.plot(traj[:, j], "o-", lw=2, color=PCOL.get(e), label=PROFILE_NAMES.get(e, e))
    ax.axhline(traj[0].mean(), ls="--", color="black", label="Moyenne initiale")
    ax.set_xlabel("Itération t"); ax.set_ylabel("Poids (1-9)")
    ax.set_title(f"Trajectoires des profils — critère « {LABELS[crit]} » (α=0.5)"); ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    save(fig, path)

def fig_sens(sens, path):
    d = sens.drop(columns="complet").sub(sens["complet"], axis=0)
    x, k = np.arange(9), d.shape[1]; width = 0.8 / k
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, col in enumerate(d):
        e = col.removeprefix("sans_")
        ax.bar(x + (i - (k - 1) / 2) * width, d[col], width, color=PCOL.get(e), label=f"sans {e}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [LABELS[c] for c in CRITERIA], rotation=30, ha="right")
    ax.set_ylabel("Δw* (sans profil − complet)"); ax.set_title("Sensibilité de w* au retrait d'un profil")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    save(fig, path)


def fig_position(pb, path):
    m = pb.groupby(["model", "position"]).weight.mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for a, c in zip(m, mcol(list(m)).values()): ax.plot(m.index, m[a], "o-", lw=2, color=c, label=a)
    ax.set_xlabel("Position du critère dans le prompt"); ax.set_ylabel("Poids moyen (1-9)"); ax.set_ylim(1, 9)
    ax.set_xticks(range(1, 10)); ax.set_title("Poids moyen selon la position dans le prompt"); ax.legend(); ax.grid(alpha=0.3)
    save(fig, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True); p.add_argument("--name", required=True)
    p.add_argument("--reference"); p.add_argument("--out", default="results")
    p.add_argument("--boot", type=int, default=1000)
    args = p.parse_args()
    out = Path(args.out) / args.name; out.mkdir(parents=True, exist_ok=True)
    raw = load_original(args.data); models = an.models_of(raw)
    orig = Path(args.data).resolve().name == "original"   # notes sur l'article : donnees de Vahidnia seulement
    if not raw: raise SystemExit(f"Aucun fichier *_G1.csv dans {args.data}")
    profs = [e for e in PROFILES if any(k[1] == e for k in raw)]
    cons = run_consensus(raw, 0.5, profiles=profs)
    md = [f"# Évaluation WALMAS — {args.name}", "", f"Données : `{args.data}` ; modèles : {', '.join(models)} ; "
          f"m = {next(iter(raw.values())).shape[1]} agents ; α = 0.5, ε = 1e-4.", ""]

    # 0. IQR
    iq = an.iqr_table(raw); iq.to_csv(out / "iqr_detail.csv", index=False)
    agg = lambda by: iq.groupby(by).agg(rejetes=("rejected", "sum"), total=("n", "sum"), iqr_nul=("zero_iqr", "sum"))
    for by in ("model", "profile"):
        t = agg(by); t["taux"] = t.rejetes / t.total; t.to_csv(out / f"iqr_par_{by}.csv")
    tot, n = iq.rejected.sum(), iq.n.sum(); eff = an.iqr_effect(raw)
    md += ["## 0. Filtre IQR", f"- Rejetés : **{tot} / {n}** ({tot / n:.2%}).",
           "- Par modèle (rejetés / IQR nul) : " + ", ".join(f"{a} {r.rejetes} / {r.iqr_nul}" for a, r in agg("model").iterrows()),
           "- Par profil (rejetés) : " + ", ".join(f"{e} {r.rejetes}" for e, r in agg("profile").iterrows()),
           f"- max |w* avec IQR − w* sans IQR| = {eff:.4f}", ""]

    # 1. Distribution
    dist = an.distribution(raw); dist.to_csv(out / "distribution.csv", index=False)
    ms = an.mean_std(raw); ms.to_csv(out / "ecart_type_moyen.csv")
    fig_heatmaps(dist, models, out / "heatmap_poids_moyens.png")
    md += ["## 1. Distribution", "Écart-type entre agents, moyenné sur critères × profils (retenu : ddof=1 avant IQR) :", "",
           ms.round(4).to_string(), "",
           *(["Valeurs de l'article : GPT-4o 0.3345, Copilot 0.3903, Gemini 0.4398. Sur data/original, "
           "aucune de ces définitions ne les reproduit (même ordre ; ddof=0 avant IQR le plus proche, ratio 0.92-0.96).", ""] if orig else [])]

    # 2. Similarite
    sim = an.model_similarity(raw); sim.to_csv(out / "similarite_modeles.csv", index=False)
    md += ["## 2. Similarité entre modèles (Spearman ρ)",
           "Vecteur par modèle = w* du modèle seul (= moyenne sur profils des moyennes IQR) ou moyenne brute.", "",
           sim.round(3).to_string(index=False), "",
           *(["Valeurs de l'article : Gemini/GPT-4o 0.33, Copilot/GPT-4o 0.43, Gemini/Copilot 0.53. "
           "Sur data/original, non reproduites (ρ obtenus 0.79-0.92 ; par profil 0.53-0.96 ; moyenne par paire d'agents 0.74-0.88).", ""] if orig else [])]

    # 3. Reference
    if args.reference:
        ref = run_consensus(load_original(args.reference), 0.5)["w_star"]
        stats, ct = an.compare(cons["w_star"], ref); ct.to_csv(out / "comparaison_reference.csv", index=False)
        per_model = {a: v["w_star"] for a, v in an.model_vectors(raw).items()}
        pm = pd.DataFrame([dict(model=a, **an.compare(v, ref)[0]) for a, v in per_model.items()])
        pm.to_csv(out / "comparaison_reference_par_modele.csv", index=False)
        fig_compare(cons["w_star"], ref, per_model, models, out / "comparaison_reference.png")
        md += ["## 3. Comparaison avec la référence", f"Référence : `{args.reference}`.",
               f"- ρ = {stats['spearman_rho']:.3f}, τ = {stats['kendall_tau']:.3f}, MAE = {stats['mae']:.3f} (échelle 1-9)",
               "", pm.round(3).to_string(index=False), ""]

    # 4. Convergence
    conv, res = an.convergence(cons["W_bar"], ALPHAS); conv.to_csv(out / "convergence.csv", index=False)
    grid, _ = an.convergence(cons["W_bar"], np.round(np.arange(0.05, 0.96, 0.05), 2))
    grid.to_csv(out / "convergence_grille_alpha.csv", index=False)
    ki = an.kendall_initial(cons["W_bar"])
    fig_wk(res, out / "kendall_W_t.png"); fig_iter(grid, out / "iterations_vs_alpha.png")
    fig_slope(res[0.5], profs, out / "trajectoires_pente.png")
    md += ["## 4. Convergence", conv.round(4).to_string(index=False), "",
           f"W_K initial : standard {ki['standard']:.3f}, legacy {ki['legacy']:.3f}.", ""]

    # 5. Sensibilite
    sens = an.profile_sensitivity(raw); sens.to_csv(out / "sensibilite_profils.csv")
    fig_sens(sens, out / "sensibilite_profils.png")
    md += ["## 5. Sensibilité aux profils (max |Δw*| par profil retiré)",
           ", ".join(f"{k.removeprefix('sans_')} {v:.2f}" for k, v in an.max_abs_delta(sens).items()), ""]

    # 6. Bootstrap
    boot = an.bootstrap(raw, args.boot); boot.to_csv(out / "bootstrap.csv", index=False)
    fig_final(boot, out / "poids_finaux.png")
    md += [f"## 6. Stabilité — proxy : bootstrap sur les agents, une seule série de graines (B = {args.boot}, graine 0)",
           boot.round(3).to_string(index=False), ""]

    # 7. Sans negociation + poids finaux
    fw = an.final_weights(cons); fw.to_csv(out / "poids_finaux.csv", index=False)
    md += ["## 7. Référence sans négociation",
           f"max |w* − moyenne simple des colonnes de W̄| = {an.no_negotiation_gap(cons):.2e}", "",
           "## Poids finaux (α = 0.5)", fw.round(4).to_string(index=False), ""]
    # 8. Biais de position (si les reponses brutes du niveau 0 sont disponibles)
    jl = Path(args.data) / "raw" / "responses.jsonl"
    if jl.exists():
        pb = an.position_bias(jl); pb.to_csv(out / "biais_position.csv", index=False)
        ps = an.position_summary(pb); ps.to_csv(out / "biais_position_resume.csv")
        fig_position(pb, out / "biais_position.png")
        md += ["## 8. Biais de position (ρ de Spearman par réponse entre poids et antériorité dans le prompt)",
               ps.round(3).to_string(), ""]
        r2 = an.criterion_position_r2(pb); r2.to_csv(out / "decomposition_critere_position.csv")
        md += ["Part de variance des poids (R²) expliquée par le critère / la position (significatif si ordre aléatoire) :",
               r2.round(3).to_string(), ""]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Resultats ecrits dans {out}")


if __name__ == "__main__":
    main()
