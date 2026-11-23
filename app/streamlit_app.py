"""Demonstration WALMAS (Vahidnia 2025) : niveaux 0-2, analyse critique, agent Ollama en direct.
Lancement : .venv/bin/python -m streamlit run app/streamlit_app.py"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, altair as alt, streamlit as st
from walmas import analysis as an
from walmas.core.pipeline import CRITERIA, PROFILES, load_original, run_consensus
from walmas.core.iqr import iqr_filter_mean
from walmas.core.kendall import kendall_w_legacy
from walmas.core.negotiation import negotiate
from walmas.llm.agents import OllamaClient, PROFILE_NAMES as PROFILE_EN, build_prompt, parse_weights, weights_schema

DATA = Path(__file__).resolve().parents[1] / "data"
# copies de scripts/evaluate.py (non importable : charge matplotlib)
LABELS = dict(zip(CRITERIA, ["Eaux (aquifères)", "Climat", "Sensibilité écologique", "Géologie",
                             "Conformité réglementaire", "Agglomérations", "Pente", "Topographie",
                             "Urbanisation / infra."]))
PROFILE_NAMES = {"ENR": "ENR (environnement)", "GGIS": "GGIS (géographie/SIG)",
                 "CEUP": "CEUP (génie civil/urbanisme)", "ES": "ES (économie/société)"}
PCOL = dict(zip(PROFILES, ["#2a9d8f", "#e9c46a", "#264653", "#e76f51"]))
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]   # plt.cm.tab10 (couleurs modeles)
ORDER = [LABELS[c] for c in CRITERIA]
PSCALE = alt.Scale(domain=PROFILES, range=[PCOL[e] for e in PROFILES])
LIVE_DEFAULT = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b"]


# ---- donnees ----
def datasets():
    """Sous-dossiers de data/ avec les 4 fichiers *_G1..G4 pour chaque modele (collecte incomplete ignoree)."""
    out = []
    for d in DATA.iterdir():
        ms = [p.name[:-len("_G1.csv")] for p in d.glob("*_G1.csv")] if d.is_dir() else []
        if ms and all((d / f"{a}_G{i}.csv").exists() for a in ms for i in (2, 3, 4)): out.append(d.name)
    return sorted(out, key=lambda n: (n != "original", n))

@st.cache_data
def load(name): return load_original(DATA / name)

@st.cache_data
def responses(path, mtime): return pd.read_json(path, lines=True)      # mtime : collecte en cours => cache invalide

@st.cache_data
def pos_bias(path, mtime): return an.position_bias(path)

def tag_names(name):
    """tag Ollama -> nom court (run_config.json de la collecte)."""
    p = DATA / name / "run_config.json"
    return {v["tag"]: k for k, v in json.loads(p.read_text())["models"].items()} if p.exists() else {}

def consensus(raw, alpha, eps, t_max, use_iqr, k):
    return run_consensus(raw, alpha, eps, t_max, use_iqr, PROFILES, k)   # colonnes dans l'ordre PROFILES


# ---- graphiques ----
def show(ch): st.altair_chart(ch, width="stretch")

def heat(M, cols, xname, fmt=".1f"):
    df = pd.DataFrame(M, index=ORDER, columns=cols).rename_axis("critère").reset_index().melt("critère", var_name=xname, value_name="poids")
    base = alt.Chart(df).encode(alt.X(f"{xname}:O", sort=list(cols), axis=alt.Axis(labelAngle=0)), alt.Y("critère:N", sort=ORDER, title=None, axis=alt.Axis(labelLimit=220)))
    return (base.mark_rect().encode(alt.Color("poids:Q", scale=alt.Scale(scheme="yellowgreenblue", domain=[1, 9])))
            + base.mark_text(fontSize=11).encode(text=alt.Text("poids:Q", format=fmt),
                                                 color=alt.condition("datum.poids > 5.5", alt.value("white"), alt.value("black")))
            ).properties(height=330)

def wbars(w, title="w*"):
    df = pd.DataFrame({"critère": ORDER, title: w})
    return alt.Chart(df).mark_bar(color="#457b9d").encode(
        alt.X(f"{title}:Q", scale=alt.Scale(domain=[0, 9])), alt.Y("critère:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        tooltip=["critère", alt.Tooltip(f"{title}:Q", format=".3f")]).properties(height=280)

def wk_chart(hist):
    df = pd.DataFrame({"t": range(len(hist)), "W_K": hist})
    return alt.Chart(df).mark_line(point=True, color="#264653").encode(
        alt.X("t:Q", title="itération t", axis=alt.Axis(tickMinStep=1)),
        alt.Y("W_K:Q", scale=alt.Scale(domain=[0, 1.05]), title="W de Kendall")).properties(height=260)


# ---- barre laterale ----
st.set_page_config(page_title="WALMAS", layout="wide")
names = datasets()
if not names: st.error(f"Aucun jeu de données complet dans {DATA}."); st.stop()
with st.sidebar:
    st.header("Paramètres")
    name = st.selectbox("Jeu de données", names, key="dataset")
    alpha = st.slider("α (poids de sa propre position)", 0.05, 0.95, 0.5, 0.05)
    eps = st.select_slider("ε (arrêt si |ΔW_K| < ε)", [1e-2, 1e-3, 1e-4, 1e-6, 1e-8], 1e-4, format_func=lambda v: f"{v:.0e}")
    t_max = int(st.number_input("t_max", 1, 1000, 100))
    use_iqr = st.checkbox("Filtre IQR (niveau 1)", True)
    k = st.slider("k de Tukey", 0.5, 3.0, 1.5, 0.25, disabled=not use_iqr)
raw = load(name); models = an.models_of(raw); m = next(iter(raw.values())).shape[1]
MSCALE = alt.Scale(domain=models, range=TAB10[:len(models)])
out = consensus(raw, alpha, eps, t_max, use_iqr, k); res = out["result"]
jl = DATA / name / "raw" / "responses.jsonl"

st.title("WALMAS : pondération de critères par agents LLM")
tabs = st.tabs(["Méthode", "Données (niveau 0)", "Niveau 1 — IQR", "Niveau 2 — Négociation",
                "Résultats & analyse critique", "Agent en direct (Ollama)"])

# ---- 1. Methode ----
with tabs[0]:
    st.markdown("""
**Problème.** Choisir un site d'enfouissement de déchets en pondérant 9 critères (eaux, géologie, pente, agglomérations…).
WALMAS (Vahidnia 2025) remplace le panel d'experts par des agents LLM jouant 4 profils : **ENR** (environnement),
**GGIS** (géographie/SIG), **CEUP** (génie civil/urbanisme), **ES** (économie/société).

- **Niveau 0** — K modèles × 4 profils × m agents donnent chacun un poids entier de 1 à 9 à chaque critère.
- **Niveau 1** — pour chaque (modèle, profil, critère), filtre de Tukey : rejet hors [Q1 − k·IQR, Q3 + k·IQR], puis moyenne ;
  moyenne sur les modèles → matrice W̄ (9 critères × 4 profils).
- **Niveau 2** — négociation : chaque profil fait W_e ← αW_e + (1−α)·moyenne des profils, jusqu'à |ΔW_K| < ε
  (W_K : concordance de Kendall entre profils). Poids finaux w\\* = moyenne des profils, ŵ = w\\*/Σw\\*.

**Question de recherche.** Que mesurent ces poids, et que fait vraiment la négociation ? Résultat démontré :
la moyenne des profils est invariante par la mise à jour, donc **w\\* = moyenne simple de W̄** quel que soit α,
et **W_K → 1 quelles que soient les entrées** : le consensus affiché est un artefact de la procédure.
Les poids viennent donc entièrement des niveaux 0-1 (choix des modèles, prompt, ordre des critères).
""")
    st.graphviz_chart("""digraph { rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica, fontsize=11];
      A [label="Niveau 0\\nK modèles × 4 profils × m agents\\npoids entiers 1..9"];
      B [label="Niveau 1\\nfiltre IQR de Tukey\\npar (modèle, profil, critère)\\npuis moyennes"];
      C [label="W̄\\n9 critères × 4 profils"];
      D [label="Niveau 2\\nW_e ← αW_e + (1−α)·moyenne\\njusqu'à |ΔW_K| < ε"];
      E [label="w* = moyenne des profils\\nŵ = w* / Σw*"];
      A -> B -> C -> D -> E; D -> D [label=" t+1"]; }""", width="stretch")
    c = st.columns(5)
    c[0].metric("Modèles", len(models), help=", ".join(models)); c[1].metric("Agents par (modèle, profil)", m)
    c[2].metric("Poids collectés", 9 * 4 * len(models) * m); c[3].metric("W_K initial", f"{res.w_history[0]:.3f}")
    c[4].metric("Itérations", res.iterations)
    st.caption(f"Jeu « {name} » : {', '.join(models)}.")

# ---- 2. Donnees ----
with tabs[1]:
    c = st.columns(3)
    a = c[0].selectbox("Modèle", models, key="d_model")
    e = c[1].selectbox("Profil", PROFILES, format_func=PROFILE_NAMES.get, key="d_prof")
    w = raw[(a, e)]
    st.subheader(f"Poids bruts — {a}, {PROFILE_NAMES[e]} (9 critères × {m} agents)")
    show(heat(w, list(range(1, m + 1)), "agent", ".0f"))
    st.dataframe(pd.DataFrame({"moyenne": w.mean(1), "écart-type": w.std(1, ddof=1), "min": w.min(1), "max": w.max(1)},
                              index=ORDER).round(2))
    if jl.exists():
        st.subheader("Exemple de prompt et de réponse brute")
        j = c[2].number_input("Agent (exemple de réponse)", 1, m, 1)
        df = responses(str(jl), jl.stat().st_mtime)
        tag = {v: t for t, v in tag_names(name).items()}.get(a, a)
        r = df[(df.model == tag) & (df.profile == e) & (df.agent == j - 1) & df.valid]
        if r.empty: st.info("Réponse introuvable dans raw/responses.jsonl.")
        else:
            r = r.iloc[-1]
            st.caption(f"{tag}, graine {r.seed}, température {r.temperature}, tentative {r.attempt + 1}")
            st.code(r.prompt, language=None, wrap_lines=True); st.code(r.response, language="json")
    else:
        st.caption(f"Pas de réponses brutes pour « {name} » (CSV fournis seulement).")

# ---- 3. Niveau 1 ----
with tabs[2]:
    c = st.columns(3)
    a = c[0].selectbox("Modèle", models, key="i_model")
    e = c[1].selectbox("Profil", PROFILES, format_func=PROFILE_NAMES.get, key="i_prof")
    crit = c[2].selectbox("Critère", CRITERIA, format_func=LABELS.get, key="i_crit")
    v = raw[(a, e)][CRITERIA.index(crit)]
    q1, q3 = np.percentile(v, [25, 75], method="linear"); lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    keep = iqr_filter_mean(v[None], k, return_mask=True)[1][0]
    dots = pd.DataFrame({"poids": v, "agent": range(1, m + 1), "statut": np.where(keep, "conservé", "rejeté")})
    dots["pile"] = dots.groupby("poids").cumcount() + 1
    x = alt.Scale(domain=[0.5, 9.5])
    band = alt.Chart(pd.DataFrame({"Q1": [q1], "Q3": [q3]})).mark_rect(color="#adb5bd", opacity=0.35).encode(alt.X("Q1:Q", scale=x), x2="Q3")
    rules = alt.Chart(pd.DataFrame({"borne": [lo, hi]})).mark_rule(color="black", strokeDash=[5, 4]).encode(alt.X("borne:Q", scale=x))
    pts = alt.Chart(dots).mark_circle(size=160, opacity=0.9).encode(
        alt.X("poids:Q", scale=x, title="poids (1-9)"), alt.Y("pile:Q", title="nombre d'agents", axis=alt.Axis(tickMinStep=1)),
        alt.Color("statut:N", scale=alt.Scale(domain=["conservé", "rejeté"], range=["#457b9d", "#d62728"])),
        tooltip=["agent", "poids", "statut"])
    show((band + rules + pts).properties(height=260))
    st.caption(f"Bande grise : [Q1, Q3] = [{q1:g}, {q3:g}] ; tirets : bornes de Tukey [{lo:g}, {hi:g}] (k = {k:g}). "
               f"Rejetés : {(~keep).sum()} / {m} ; moyenne filtrée {v[keep].mean():.3f} vs brute {v.mean():.3f}."
               + ("" if use_iqr else " Filtre désactivé dans la barre latérale : bornes affichées pour information."))
    st.subheader("Poids rejetés par modèle et profil")
    rj = pd.DataFrame([dict(modèle=a_, profil=e_, rejetés=int((~iqr_filter_mean(w_, k, return_mask=True)[1]).sum()), total=w_.size)
                       for (a_, e_), w_ in raw.items()])
    t = rj.pivot_table(index="modèle", columns="profil", values="rejetés", aggfunc="sum")[PROFILES]
    t["total"] = t.sum(axis=1); t["taux"] = t.total / rj.groupby("modèle").total.sum()
    st.dataframe(t.style.format({"taux": "{:.1%}"}))
    st.caption(f"Total : {rj.rejetés.sum()} / {rj.total.sum()} poids rejetés ({rj.rejetés.sum() / rj.total.sum():.2%}) avec k = {k:g}.")
    st.subheader("Effet du filtre sur W̄ (avec IQR − sans IQR)")
    on, off = consensus(raw, alpha, eps, t_max, True, k), consensus(raw, alpha, eps, t_max, False, k)
    eff = pd.DataFrame(on["W_bar"] - off["W_bar"], index=ORDER, columns=[f"Δ {e_}" for e_ in PROFILES])
    eff["w* avec IQR"], eff["w* sans IQR"] = on["w_star"], off["w_star"]
    st.dataframe(eff.style.format("{:+.3f}", subset=eff.columns[:4]).format("{:.3f}", subset=eff.columns[4:]))
    st.caption(f"max |Δw*| = {np.abs(on['w_star'] - off['w_star']).max():.4f}")

# ---- 4. Niveau 2 ----
with tabs[3]:
    c = st.columns(4)
    c[0].metric("W_K initial (standard)", f"{res.w_history[0]:.3f}")
    c[1].metric("W_K initial (legacy)", f"{kendall_w_legacy(out['W_bar']):.3f}",
                help="Variante du script original (1 − 12S/…), conservée pour expliquer l'écart avec l'article.")
    c[2].metric("Itérations", res.iterations); c[3].metric("Convergé", "oui" if res.converged else "non (t_max atteint)")
    l, r_ = st.columns(2)
    with l:
        crit = st.selectbox("Critère", CRITERIA, CRITERIA.index("Slope"), format_func=LABELS.get, key="n_crit")
        i = CRITERIA.index(crit); traj = np.array([M[i] for M in res.M_history])
        df = pd.DataFrame(traj, columns=PROFILES).rename_axis("t").reset_index().melt("t", var_name="profil", value_name="poids")
        show((alt.Chart(df).mark_line(point=True).encode(alt.X("t:Q", title="itération t", axis=alt.Axis(tickMinStep=1)),
                                                         alt.Y("poids:Q", scale=alt.Scale(zero=False), title="poids (1-9)"), alt.Color("profil:N", scale=PSCALE))
              + alt.Chart(pd.DataFrame({"moyenne initiale": [traj[0].mean()]})).mark_rule(strokeDash=[6, 4], color="black")
              .encode(alt.Y("moyenne initiale:Q", title="poids (1-9)"))).properties(height=300, title=f"Trajectoires — {LABELS[crit]}"))
        st.caption("Tirets : moyenne initiale des profils, invariante au cours de la négociation.")
    with r_:
        st.markdown("**Concordance W_K(t)**"); show(wk_chart(res.w_history))
    t = st.slider("Itération t", 0, res.iterations, 0)
    st.markdown(f"**Matrice M({t})** (critères × profils)")
    show(heat(res.M_history[t], PROFILES, "profil", ".2f"))

# ---- 5. Resultats ----
with tabs[4]:
    st.subheader("Poids finaux")
    l, r_ = st.columns([3, 2])
    with l: show(wbars(out["w_star"]))
    fw = an.final_weights(out).assign(criterion=lambda d: d.criterion.map(LABELS))
    r_.dataframe(fw.rename(columns={"rank": "rang", "criterion": "critère", "w_star": "w*", "w_hat": "ŵ"})
                 .style.format({"rang": "{:.0f}", "w*": "{:.3f}", "ŵ": "{:.3f}"}), hide_index=True)
    st.metric("max |w* − moyenne simple des colonnes de W̄|", f"{an.no_negotiation_gap(out):.1e}",
              help="≈ 0 à la précision machine : la négociation ne modifie pas les poids.")

    st.subheader("Comparaison des jeux de données")
    W = {n: consensus(load(n), alpha, eps, t_max, use_iqr, k)["w_star"] for n in names}
    df = pd.DataFrame(W, index=ORDER).rename_axis("critère").reset_index().melt("critère", var_name="jeu", value_name="w*")
    show(alt.Chart(df).mark_bar().encode(alt.X("critère:N", sort=ORDER, title=None, axis=alt.Axis(labelAngle=-30, labelOverlap=False, labelLimit=200)),
                                         alt.XOffset("jeu:N"), alt.Y("w*:Q", scale=alt.Scale(domain=[0, 9])),
                                         alt.Color("jeu:N", scale=alt.Scale(domain=names, range=["#333333", "#457b9d", "#e76f51", "#adb5bd"])),
                                         tooltip=["jeu", "critère", alt.Tooltip("w*:Q", format=".3f")]).properties(height=300))
    if "original" in W and len(W) > 1:
        st.dataframe(pd.DataFrame([dict(jeu=n, **an.compare(W[n], W["original"])[0]) for n in W if n != "original"])
                     .rename(columns={"spearman_rho": "ρ Spearman", "kendall_tau": "τ Kendall", "mae": "MAE (1-9)"}).round(3),
                     hide_index=True)
        st.caption("Comparaison à la référence data/original (Vahidnia : ChatGPT, Copilot, Gemini).")

    st.subheader("Sensibilité au retrait d'un profil")
    sens = an.profile_sensitivity(raw, alpha)
    d = sens.drop(columns="complet").sub(sens["complet"], axis=0).rename(index=LABELS, columns=lambda s: s.removeprefix("sans_"))
    df = d.rename_axis("critère").reset_index().melt("critère", var_name="profil retiré", value_name="Δw*")
    show(alt.Chart(df).mark_bar().encode(alt.X("critère:N", sort=ORDER, title=None, axis=alt.Axis(labelAngle=-30, labelOverlap=False, labelLimit=200)),
                                         alt.XOffset("profil retiré:N", sort=PROFILES), alt.Y("Δw*:Q", title="Δw* (sans profil − complet)"),
                                         alt.Color("profil retiré:N", scale=PSCALE)).properties(height=280))
    st.caption("max |Δw*| : " + ", ".join(f"sans {s.removeprefix('sans_')} {v:.2f}" for s, v in an.max_abs_delta(sens).items())
               + " (filtre IQR standard k = 1.5).")

    st.subheader("Contrôle : entrées aléatoires")
    st.markdown("W̄ tiré uniformément dans [1, 9] (9 critères × 4 profils, aucune information) : la négociation affiche quand même un consensus.")
    if st.button("Entrées aléatoires"):
        R = np.random.default_rng().uniform(1, 9, (9, 4)); rr = negotiate(R, alpha, eps, t_max)
        l, r_ = st.columns(2)
        with l: show(wk_chart(rr.w_history))
        r_.metric("W_K initial → final", f"{rr.w_history[0]:.3f} → {rr.w_history[-1]:.3f}", f"{rr.iterations} itérations")
        r_.dataframe(pd.DataFrame(R, index=ORDER, columns=PROFILES).round(2))

    srcs = [p for p in sorted(DATA.iterdir()) if (p / "raw" / "responses.jsonl").exists()]
    if srcs:
        st.subheader("Biais de position dans le prompt")
        st.markdown("Poids moyen selon le rang du critère dans le prompt ; ρ : Spearman par réponse entre poids et antériorité "
                    "(ρ > 0 : les premiers critères reçoivent plus). Ordre fixe : position et critère sont confondus ; "
                    "ordre aléatoire par agent : une pente persistante signale un vrai biais de position.")
        for col, p in zip(st.columns(len(srcs)), srcs):
            f = p / "raw" / "responses.jsonl"; pb = pos_bias(str(f), f.stat().st_mtime)
            col.markdown(f"**{p.name}**" + ("" if p.name in names else " (collecte incomplète)"))
            if pb.empty: col.info("Aucune réponse valide."); continue
            pb["model"] = pb.model.map(tag_names(p.name)).fillna(pb.model)
            g = pb.groupby(["model", "position"], as_index=False).weight.mean()
            ms = sorted(g.model.unique())
            with col:
                show(alt.Chart(g).mark_line(point=True).encode(
                    alt.X("position:O", title="position dans le prompt", axis=alt.Axis(labelAngle=0)),
                    alt.Y("weight:Q", scale=alt.Scale(domain=[1, 9]), title="poids moyen"),
                    alt.Color("model:N", scale=alt.Scale(domain=ms, range=TAB10[:len(ms)]), title="modèle")).properties(height=260))
            col.dataframe(an.position_summary(pb).round(3).rename(columns={"rho_moyen": "ρ moyen", "part_rho_positif": "part ρ > 0",
                                                                          "n_reponses": "réponses"}))

# ---- 6. Agent en direct ----
@st.cache_data(ttl=60)
def ollama_tags():
    names = [t["name"] for t in OllamaClient(timeout=3)._call("/api/tags")["models"]]
    return sorted(names, key=lambda t: (t not in LIVE_DEFAULT, t))       # modeles du projet en premier

def ask(client, tag, profile, temp, seed, shuffle):
    """Un agent : meme ordre melange et meme graine que llm.agents.collect_profile."""
    order = [str(c) for c in np.random.default_rng(seed).permutation(CRITERIA)] if shuffle else list(CRITERIA)
    r = dict(prompt=build_prompt(PROFILE_EN[profile], order), text=None, w=None, err=None); t0 = time.time()
    try: r["text"] = client.chat(tag, r["prompt"], weights_schema(order), temp, seed); r["w"] = parse_weights(r["text"])
    except Exception as ex: r["err"] = f"{type(ex).__name__}: {ex}"
    r["dt"] = time.time() - t0
    return r

with tabs[5]:
    st.warning("Chaque appel prend environ 5 à 10 s sur un Mac M1 (modèles 7-8B en local). Ollama doit tourner sur localhost:11434.")
    try: tags = ollama_tags() or LIVE_DEFAULT
    except Exception as ex: tags = LIVE_DEFAULT; st.info(f"Ollama injoignable ({type(ex).__name__}) : liste de modèles par défaut.")
    client = OllamaClient(timeout=120)
    c = st.columns(3)
    tag = c[0].selectbox("Modèle", tags, key="l_model")
    prof = c[1].selectbox("Profil", PROFILES, format_func=PROFILE_NAMES.get, key="l_prof")
    temp = c[2].slider("Température", 0.0, 1.5, 0.7, 0.1)
    c = st.columns(3)
    seed = int(c[0].number_input("Graine", 0, 10**6, 0))
    n_ag = c[1].slider("Nombre d'agents", 1, 5, 1)
    shuffle = c[2].checkbox("Ordre des critères aléatoire")
    if st.button("Interroger les agents", type="primary"):
        bar = st.progress(0.0)
        st.session_state.live = []
        for s in range(n_ag):
            st.session_state.live.append(ask(client, tag, prof, temp, seed + s, shuffle)); bar.progress((s + 1) / n_ag)
    for s, r in enumerate(st.session_state.get("live", [])):
        with st.expander(f"Agent {s + 1} — {r['dt']:.1f} s", expanded=True):
            st.code(r["prompt"], language=None, wrap_lines=True)
            if r["text"] is not None: st.code(r["text"], language="json")
            if r["err"]: st.error(f"Réponse inutilisable : {r['err']}")
            else: show(wbars(r["w"], "poids"))

    st.subheader("Mini-panel : 1 modèle × 4 profils × m agents, puis consensus")
    mp = st.slider("m agents par profil", 1, 3, 2)
    st.caption(f"{4 * mp} appels, soit environ {4 * mp * 5}-{4 * mp * 10} s. Mêmes modèle, température, graine et ordre que ci-dessus.")
    if st.button("Lancer le mini-panel"):
        bar, raw_p, err = st.progress(0.0), {}, None
        for i, e in enumerate(PROFILES):
            ws = []
            for s in range(mp):
                r = ask(client, tag, e, temp, seed + s, shuffle)
                if r["err"]: err = f"{e}, agent {s + 1} : {r['err']}"; break
                ws.append(r["w"]); bar.progress((i * mp + s + 1) / (4 * mp), f"{e} : agent {s + 1}/{mp}")
            if err: break
            raw_p[(tag, e)] = np.array(ws, float).T
        st.session_state.panel = (raw_p, err)
    if "panel" in st.session_state:
        raw_p, err = st.session_state.panel
        if err: st.error(f"Mini-panel interrompu ({err}).")
        else:
            op = consensus(raw_p, alpha, eps, t_max, use_iqr, k)
            l, r_ = st.columns(2)
            with l: st.markdown("**w* du mini-panel**"); show(wbars(op["w_star"]))
            with r_: st.markdown(f"**W_K(t)** — {op['result'].iterations} itérations"); show(wk_chart(op["result"].w_history))
            st.dataframe(pd.DataFrame(op["W_bar"], index=ORDER, columns=PROFILES).round(2))
