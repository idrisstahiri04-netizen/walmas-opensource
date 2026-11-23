# Évaluation WALMAS — opensource

Données : `data/opensource` ; modèles : Llama31, Mistral, Qwen25 ; m = 20 agents ; α = 0.5, ε = 1e-4.

## 0. Filtre IQR
- Rejetés : **139 / 2160** (6.44%).
- Par modèle (rejetés / IQR nul) : Llama31 77 / 5, Mistral 43 / 18, Qwen25 19 / 3
- Par profil (rejetés) : CEUP 24, ENR 38, ES 24, GGIS 53
- max |w* avec IQR − w* sans IQR| = 0.2608

## 1. Distribution
Écart-type entre agents, moyenné sur critères × profils (retenu : ddof=1 avant IQR) :

         ddof0_avant_iqr  ddof1_avant_iqr  ddof0_apres_iqr  ddof1_apres_iqr
model                                                                      
Llama31           1.0413           1.0683           0.6514           0.6690
Mistral           0.4399           0.4513           0.2952           0.3029
Qwen25            0.9876           1.0133           0.9178           0.9416

## 2. Similarité entre modèles (Spearman ρ)
Vecteur par modèle = w* du modèle seul (= moyenne sur profils des moyennes IQR) ou moyenne brute.

model_a model_b  rho_w_star  rho_moy_brute
Llama31 Mistral       0.750          0.750
Llama31  Qwen25       0.983          0.983
Mistral  Qwen25       0.800          0.800

## 3. Comparaison avec la référence
Référence : `data/original`.
- ρ = -0.100, τ = -0.056, MAE = 2.058 (échelle 1-9)

  model  spearman_rho  kendall_tau   mae
Llama31        -0.050        0.000 2.560
Mistral        -0.233       -0.222 1.815
 Qwen25        -0.100       -0.056 2.358

## 4. Convergence
 alpha  iterations  W_initial  W_final  converged
   0.3           2     0.9828      1.0       True
   0.5           3     0.9828      1.0       True
   0.7           4     0.9828      1.0       True

W_K initial : standard 0.983, legacy 0.997.

## 5. Sensibilité aux profils (max |Δw*| par profil retiré)
ENR 0.06, GGIS 0.13, CEUP 0.14, ES 0.15

## 6. Stabilité — proxy : bootstrap sur les agents, une seule série de graines (B = 1000, graine 0)
                                    criterion  w_star  rank  ic95_bas  ic95_haut  part_rang_conserve
                   Aquifers and Surface Water   8.471   1.0     8.400      8.517               1.000
               Climate and Weather Conditions   6.043   5.0     5.904      6.154               1.000
                       Ecological Sensitivity   8.208   2.0     8.071      8.308               1.000
                                      Geology   7.668   3.0     7.569      7.835               1.000
                        Regulatory Compliance   6.364   4.0     6.310      6.618               1.000
                Settlements and Urban Centers   5.013   6.0     4.779      5.144               1.000
                                        Slope   4.139   7.0     3.935      4.281               1.000
                                   Topography   3.693   8.0     3.409      3.854               0.972
Urban Development and Infrastructure Location   3.361   9.0     3.086      3.598               0.972

## 7. Référence sans négociation
max |w* − moyenne simple des colonnes de W̄| = 1.78e-15

## Poids finaux (α = 0.5)
 rank                                     criterion  w_star  w_hat
  1.0                    Aquifers and Surface Water  8.4708 0.1599
  2.0                        Ecological Sensitivity  8.2083 0.1550
  3.0                                       Geology  7.6681 0.1448
  4.0                         Regulatory Compliance  6.3642 0.1202
  5.0                Climate and Weather Conditions  6.0428 0.1141
  6.0                 Settlements and Urban Centers  5.0132 0.0947
  7.0                                         Slope  4.1389 0.0782
  8.0                                    Topography  3.6931 0.0697
  9.0 Urban Development and Infrastructure Location  3.3612 0.0635

## 8. Biais de position (ρ de Spearman par réponse entre poids et antériorité dans le prompt)
             rho_moyen  part_rho_positif  n_reponses
model                                               
llama3.1:8b      0.709               1.0          80
mistral:7b       0.847               1.0          80
qwen2.5:7b       0.852               1.0          80

Part de variance des poids (R²) expliquée par le critère / la position (significatif si ordre aléatoire) :
             R2_critere  R2_position  R2_les_deux
model                                            
llama3.1:8b       0.762        0.762        0.762
mistral:7b        0.830        0.830        0.830
qwen2.5:7b        0.808        0.808        0.808
ensemble          0.604        0.604        0.604
