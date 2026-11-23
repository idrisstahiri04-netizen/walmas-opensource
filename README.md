# walmas — WALMAS avec des LLM open source

Ponderation collaborative des criteres en decision multicritere spatiale par un systeme
multi-agents de grands modeles de langage (methode WALMAS, Vahidnia 2025), mise en oeuvre
exclusivement avec des modeles open source executes en local (Ollama).
Cas d'etude : selection d'un site de decharge, 9 criteres, 4 profils d'experts.

Rapport complet : [`rapport/main.pdf`](rapport/main.pdf) (sources LaTeX dans `rapport/`).

Principaux resultats :
- sur les poids publies de l'etude originale, le code retrouve les poids finaux publies ;
- la negociation ne modifie pas les poids (w* = moyenne simple des profils) et le W de Kendall
  tend vers 1 par construction : il suit la convergence sans valider le consensus ;
- Llama 3.1 8B, Qwen 2.5 7B et Mistral 7B (Q4, Ollama) : 240/240 reponses valides, profils
  d'experts quasi indiscernables (W_K initial 0,983), classement eloigne de l'etude originale
  (rho = -0,10) et fort biais de position (criteres cites en premier mieux notes) ;
- controle avec ordre aleatoire des criteres : le biais de position persiste (15 % de la variance
  des poids) mais s'ajoute a une preference propre des modeles pour les criteres environnementaux.

## Installation

    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
    .venv/bin/python -m pytest -q

## Structure

| Niveau | Module | Role |
|---|---|---|
| 0 | `src/walmas/llm/agents.py` | prompts, appels Ollama, validation des reponses, collecte |
| 1 | `src/walmas/core/iqr.py` | filtre de Tukey (IQR) puis moyenne par (modele, profil) |
| 2 | `src/walmas/core/negotiation.py`, `kendall.py` | negociation iterative, W de Kendall |
| — | `src/walmas/core/pipeline.py` | chargement des CSV et enchainement des niveaux 1-2 |
| — | `src/walmas/analysis.py`, `scripts/evaluate.py` | indicateurs d'evaluation et figures |

Donnees : fichiers `{Modele}_G{1..4}.csv` (profils ENR, GGIS, CEUP, ES ; 9 criteres en ordre
alphabetique x m agents, entiers 1-9).
- `data/original/` : poids publies par Vahidnia (GPT-4o, Gemini, Copilot), depot MAS-LLMs.
- `data/opensource/` : notre collecte (Llama 3.1 8B, Qwen 2.5 7B, Mistral 7B), avec
  `raw/responses.jsonl` (toutes les reponses brutes) et `run_config.json` (versions, digests,
  quantification, temperature, graines, prompt).
- `data/opensource_shuffle/` : experience de controle, ordre des criteres tire au hasard par agent
  (champ `order` dans `raw/responses.jsonl`).
- `results/` : tableaux, figures et `summary.md` produits par `scripts/evaluate.py`.

## Reproduire

1. Modeles (Ollama >= 0.5) :

       ollama pull llama3.1:8b && ollama pull qwen2.5:7b && ollama pull mistral:7b

2. Collecte niveau 0 (3 modeles x 4 profils x 20 agents = 240 appels ; reprise automatique) :

       .venv/bin/python scripts/collect.py --out data/opensource
       .venv/bin/python scripts/collect.py --shuffle --out data/opensource_shuffle   # controle

3. Niveaux 1-2 et evaluation :

       .venv/bin/python scripts/evaluate.py --data data/original --name original
       .venv/bin/python scripts/evaluate.py --data data/opensource --name opensource --reference data/original
       .venv/bin/python scripts/evaluate.py --data data/opensource_shuffle --name opensource_shuffle --reference data/original
       .venv/bin/python scripts/evaluate.py --data data/opensource_shuffle --name shuffle_vs_fixe --reference data/opensource

   Tableaux, figures et `summary.md` dans `results/<name>/`.

4. Application de demonstration (Streamlit) :

       .venv/bin/pip install -e ".[app]"
       .venv/bin/python -m streamlit run app/streamlit_app.py

   Onglets : methode, donnees (niveau 0), niveau 1 IQR, niveau 2 negociation, resultats et
   analyse critique, agent en direct (necessite Ollama et les modeles ci-dessus).

Parametres par defaut : temperature 0,7, graine de l'agent s = s (+1000 par nouvel essai),
3 essais max, alpha = 0,5, eps = 1e-4, t_max = 100.

## Auteurs

Hamza Ahrim, Mouafiq Saad, Med. Idriss Tahiri — Ecole Mohammadia d'Ingenieurs, Rabat.
Phase 1 (coeur algorithmique) initialement publiee dans `hamzaahrim0/walmas`.
Methode et donnees de reference : M. H. Vahidnia, *Multi-agent systems of large language models as
weight assigners*, Geomatica 77 (2025) 100071 ; depot `mhvahidnia/MAS-LLMs`.
