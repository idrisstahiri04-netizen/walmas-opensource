# Évaluation WALMAS — original

Données : `data/original` ; modèles : ChatGPT, Copilot, Gemini ; m = 20 agents ; α = 0.5, ε = 1e-4.

## 0. Filtre IQR
- Rejetés : **80 / 2160** (3.70%).
- Par modèle (rejetés / IQR nul) : ChatGPT 56 / 17, Copilot 24 / 10, Gemini 0 / 1
- Par profil (rejetés) : CEUP 17, ENR 25, ES 28, GGIS 10
- max |w* avec IQR − w* sans IQR| = 0.0542

## 1. Distribution
Écart-type entre agents, moyenné sur critères × profils (retenu : ddof=1 avant IQR) :

         ddof0_avant_iqr  ddof1_avant_iqr  ddof0_apres_iqr  ddof1_apres_iqr
model                                                                      
ChatGPT           0.3623           0.3717           0.1955           0.2005
Copilot           0.4057           0.4163           0.3394           0.3482
Gemini            0.4763           0.4887           0.4763           0.4887

Valeurs de l'article : GPT-4o 0.3345, Copilot 0.3903, Gemini 0.4398. Sur data/original, aucune de ces définitions ne les reproduit (même ordre ; ddof=0 avant IQR le plus proche, ratio 0.92-0.96).

## 2. Similarité entre modèles (Spearman ρ)
Vecteur par modèle = w* du modèle seul (= moyenne sur profils des moyennes IQR) ou moyenne brute.

model_a model_b  rho_w_star  rho_moy_brute
ChatGPT Copilot       0.900          0.900
ChatGPT  Gemini       0.828          0.795
Copilot  Gemini       0.921          0.921

Valeurs de l'article : Gemini/GPT-4o 0.33, Copilot/GPT-4o 0.43, Gemini/Copilot 0.53. Sur data/original, non reproduites (ρ obtenus 0.79-0.92 ; par profil 0.53-0.96 ; moyenne par paire d'agents 0.74-0.88).

## 4. Convergence
 alpha  iterations  W_initial  W_final  converged
   0.3           4     0.4333      1.0       True
   0.5           5     0.4333      1.0       True
   0.7           9     0.4333      1.0       True

W_K initial : standard 0.433, legacy 0.858.

## 5. Sensibilité aux profils (max |Δw*| par profil retiré)
ENR 1.04, GGIS 0.67, CEUP 0.52, ES 0.77

## 6. Stabilité — proxy : bootstrap sur les agents, une seule série de graines (B = 1000, graine 0)
                                    criterion  w_star  rank  ic95_bas  ic95_haut  part_rang_conserve
                   Aquifers and Surface Water   4.896   7.0     4.800      4.979               0.999
               Climate and Weather Conditions   5.088   6.0     4.975      5.167               0.999
                       Ecological Sensitivity   6.283   4.0     6.188      6.379               1.000
                                      Geology   4.404   8.0     4.304      4.471               1.000
                        Regulatory Compliance   7.558   1.0     7.442      7.596               1.000
                Settlements and Urban Centers   7.108   2.0     7.017      7.188               1.000
                                        Slope   5.875   5.0     5.792      5.921               1.000
                                   Topography   4.025   9.0     3.946      4.104               1.000
Urban Development and Infrastructure Location   6.804   3.0     6.675      6.858               1.000

## 7. Référence sans négociation
max |w* − moyenne simple des colonnes de W̄| = 1.78e-15

## Poids finaux (α = 0.5)
 rank                                     criterion  w_star  w_hat
  1.0                         Regulatory Compliance  7.5583 0.1452
  2.0                 Settlements and Urban Centers  7.1083 0.1366
  3.0 Urban Development and Infrastructure Location  6.8042 0.1307
  4.0                        Ecological Sensitivity  6.2833 0.1207
  5.0                                         Slope  5.8750 0.1129
  6.0                Climate and Weather Conditions  5.0875 0.0978
  7.0                    Aquifers and Surface Water  4.8958 0.0941
  8.0                                       Geology  4.4042 0.0846
  9.0                                    Topography  4.0250 0.0773
