# Évaluation WALMAS — shuffle_vs_fixe

Données : `data/opensource_shuffle` ; modèles : Llama31, Mistral, Qwen25 ; m = 20 agents ; α = 0.5, ε = 1e-4.

## 0. Filtre IQR
- Rejetés : **67 / 2160** (3.10%).
- Par modèle (rejetés / IQR nul) : Llama31 10 / 0, Mistral 42 / 3, Qwen25 15 / 0
- Par profil (rejetés) : CEUP 8, ENR 22, ES 14, GGIS 23
- max |w* avec IQR − w* sans IQR| = 0.1708

## 1. Distribution
Écart-type entre agents, moyenné sur critères × profils (retenu : ddof=1 avant IQR) :

         ddof0_avant_iqr  ddof1_avant_iqr  ddof0_apres_iqr  ddof1_apres_iqr
model                                                                      
Llama31           1.5304           1.5702           1.4778           1.5166
Mistral           0.7813           0.8016           0.6521           0.6694
Qwen25            1.9952           2.0470           1.8364           1.8849

## 2. Similarité entre modèles (Spearman ρ)
Vecteur par modèle = w* du modèle seul (= moyenne sur profils des moyennes IQR) ou moyenne brute.

model_a model_b  rho_w_star  rho_moy_brute
Llama31 Mistral       0.600          0.600
Llama31  Qwen25       0.917          0.933
Mistral  Qwen25       0.533          0.567

## 3. Comparaison avec la référence
Référence : `data/opensource`.
- ρ = 0.817, τ = 0.722, MAE = 0.885 (échelle 1-9)

  model  spearman_rho  kendall_tau   mae
Llama31         0.800        0.667 0.821
Mistral         0.717        0.444 1.353
 Qwen25         0.800        0.667 0.806

## 4. Convergence
 alpha  iterations  W_initial  W_final  converged
   0.3           4     0.9438   1.0000       True
   0.5           6     0.9438   1.0000       True
   0.7           3     0.9438   0.9562       True

W_K initial : standard 0.944, legacy 0.986.

## 5. Sensibilité aux profils (max |Δw*| par profil retiré)
ENR 0.14, GGIS 0.09, CEUP 0.16, ES 0.13

## 6. Stabilité — proxy : bootstrap sur les agents, une seule série de graines (B = 1000, graine 0)
                                    criterion  w_star  rank  ic95_bas  ic95_haut  part_rang_conserve
                   Aquifers and Surface Water   8.127   1.0     7.867      8.260               0.904
               Climate and Weather Conditions   4.926   9.0     4.717      5.131               0.597
                       Ecological Sensitivity   7.954   2.0     7.718      8.082               0.903
                                      Geology   7.205   4.0     6.946      7.463               0.851
                        Regulatory Compliance   7.367   3.0     7.179      7.692               0.850
                Settlements and Urban Centers   5.350   5.0     5.142      5.675               0.555
                                        Slope   5.335   6.0     5.139      5.531               0.453
                                   Topography   5.314   7.0     5.108      5.528               0.448
Urban Development and Infrastructure Location   4.996   8.0     4.698      5.211               0.573

## 7. Référence sans négociation
max |w* − moyenne simple des colonnes de W̄| = 1.78e-15

## Poids finaux (α = 0.5)
 rank                                     criterion  w_star  w_hat
  1.0                    Aquifers and Surface Water  8.1266 0.1436
  2.0                        Ecological Sensitivity  7.9542 0.1406
  3.0                         Regulatory Compliance  7.3667 0.1302
  4.0                                       Geology  7.2047 0.1274
  5.0                 Settlements and Urban Centers  5.3496 0.0946
  6.0                                         Slope  5.3346 0.0943
  7.0                                    Topography  5.3143 0.0939
  8.0 Urban Development and Infrastructure Location  4.9958 0.0883
  9.0                Climate and Weather Conditions  4.9263 0.0871

## 8. Biais de position (ρ de Spearman par réponse entre poids et antériorité dans le prompt)
             rho_moyen  part_rho_positif  n_reponses
model                                               
llama3.1:8b      0.343             0.812          80
mistral:7b       0.295             0.750          80
qwen2.5:7b       0.543             0.988          80

Part de variance des poids (R²) expliquée par le critère / la position (significatif si ordre aléatoire) :
             R2_critere  R2_position  R2_les_deux
model                                            
llama3.1:8b       0.439        0.141        0.542
mistral:7b        0.579        0.110        0.697
qwen2.5:7b        0.283        0.315        0.552
ensemble          0.316        0.154        0.443
