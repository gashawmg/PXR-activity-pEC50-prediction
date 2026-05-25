# Predicting PXR Induction Potency — OpenADMET Blind Challenge

> **Phase 1 result:** MAE **0.4468** · RAE **0.5606** · R² **0.5459** · Spearman **0.8463** · Kendall's τ **0.6567** · Rank **35**  
> 3-model ensemble · Chemprop D-MPNN + UniMol 3D transformer · No proprietary data

---

## Table of Contents

1. [Why PXR?](#why-pxr)  
2. [The Challenge](#the-challenge)  
3. [Data: From Screen to Model-Ready Set](#data-from-screen-to-model-ready-set)  
4. [Chemical Space Analysis](#chemical-space-analysis)  
5. [Models](#models)  
6. [What We Tried (and What Failed)](#what-we-tried-and-what-failed)  
7. [Best Ensemble](#best-ensemble)  
8. [What Top Teams Are Likely Doing Differently](#what-top-teams-are-likely-doing-differently)  
9. [Repo Contents](#repo-contents)  

---

## Why PXR?

The **Pregnane X Receptor (PXR / NR1I2)** is a nuclear hormone receptor that functions as the body's master xenobiotic sensor. When a drug activates PXR, it triggers a cascade that upregulates **CYP3A4** — the enzyme responsible for metabolising roughly half of all marketed drugs. The consequences for a drug candidate are severe:

| Consequence | Mechanism |
|---|---|
| **Drug-Drug Interactions (DDIs)** | Accelerated CYP3A4 expression depletes co-administered drugs to sub-therapeutic levels |
| **Hepatotoxicity** | Increased production of reactive, toxic metabolites |
| **Chemoresistance** | Enhanced clearance of oncology agents in PXR-expressing tumours |

What makes PXR especially hard to model is its **exceptionally large and flexible ligand-binding pocket** — it accommodates structurally diverse compounds ranging from macrolide antibiotics to steroids to small synthetic fragments. Published ChEMBL data prior to this challenge contained only ~800 high-quality pEC50 measurements from nearly 150 papers, spread across heterogeneous assay formats.

---

## The Challenge

The **<a href="https://huggingface.co/spaces/openadmet/pxr-challenge" target="_blank" rel="noopener noreferrer">OpenADMET Blind Challenge</a>** (Octant Bio + UCSF) released the largest uniform-assay PXR dataset ever made public: over 11,000 compounds screened in the same cell-based luciferase reporter system.

**Assay design:** A chimeric PXR ligand-binding domain fused to a heterologous DNA-binding domain drives a luciferase reporter. A parallel **counter-screen** with a nonsense-mutated chimera filters out false positives (general transcriptional activators, HDAC inhibitors) from true PXR agonists.

**The test set is not a random sample.** It comprises 513 Enamine on-demand analogs of the 63 most potent and selective hits (ECFP4 Tanimoto > 0.4 to any of the 63). This **analog expansion design** deliberately creates activity cliffs and SAR challenges — exactly what real lead optimisation looks like.

| Phase | Dates | Description |
|---|---|---|
| **Phase 1** | April 1 – May 25, 2026 | Blind prediction for all 513 test compounds; live leaderboard |
| **Phase 2** | May 26 – July 1, 2026 | Analog Set 1 labels unblinded; refine predictions for Set 2 |

**Metric:** Mean Absolute Error (MAE) on pEC50 (primary); RAE, R², Spearman ρ also reported.

---

## Data: From Screen to Model-Ready Set

### The generation funnel

```
Primary screen (11,362 compounds @ 10 µM / 30 µM)
    ↓  Hit rate ~17% @ 10 µM
Dose-response curves (4,325 compounds, 8-concentration)
    ↓  Fitted EC50 ≤ 1 µM
~211 active compounds (pEC50 ≥ 6)
    ↓  Counter-screen: selectivity_delta ≥ 1.5 log-units
63 selective hits → Enamine analog expansion → 513 test compounds
```

The competition released **4,140 compounds with pEC50 labels** as training data. Each has both a PXR pEC50 and a counter-screen pEC50, giving a **selectivity delta**:

```
selectivity_delta = pEC50(PXR) − pEC50(counter-screen)
```

### Finding the right training set

This was the most consequential decision in the project. We tested every threshold:

| Dataset | Compounds | Filter | Leaderboard MAE | Verdict |
|---|---|---|---|---|
| `clean_train.csv` | 2,948 | delta > 1.5 | not submitted | Starting point |
| **`clean_train2.csv`** | **3,743** | **delta > 0** | **0.4622 (Chemprop)** | ✅ **FINAL** |
| `clean_train1.csv` | 2,912 | Emax ≥ 0.75 | 0.4674 | ❌ Filter too aggressive |
| Relaxed (delta ≥ −0.6) | 4,054 | — | 0.4809 | ❌ Non-selective compounds contaminate |
| Unfiltered | 4,139 | None | worse | ❌ Rejected |

**The biological logic:** The 513 test compounds are analogs of hits with selectivity delta ≥ 1.5. The training set that best represents this population is one filtered by *positive* selectivity — delta > 0 captures compounds that show some PXR preference over the counter-screen, without forcing the strict 1.5-unit threshold that would discard useful potency information from moderate hits.

> **Rule established:** `clean_train2.csv` (3,743 compounds, selectivity_delta > 0) is the hard ceiling. Adding *any* compounds beyond this — by relaxing the filter, hand-picking borderline entries, or adding external ChEMBL data — degraded leaderboard performance in every single experiment. The boundary is not arbitrary: it reflects where the training distribution starts to diverge from the analog-expansion test set.

### Why external ChEMBL data hurt

We curated 37 ChEMBL PXR compounds with ECFP4 Tanimoto ≥ 0.4 to blind test compounds and added them to training. Leaderboard MAE jumped from 0.4622 to **0.5051** (rank 36 → 80). The reason: ChEMBL PXR measurements span at least 150 different assay protocols, cell lines, and reporter constructs accumulated over two decades. The OpenADMET assay is a single, tightly controlled protocol. Adding cross-assay data introduces systematic offsets that the model learns as real signal — but they are assay artefacts.

---

## Chemical Space Analysis

### Structural coverage of the test set

Before building any model, we mapped the relationship between training and test chemical space using Morgan fingerprint Tanimoto similarity against `clean_train2.csv` (3,743 compounds). This is consistent with the challenge design: the 513 test compounds are Enamine analog expansions of the top 63 training hits, so structural coverage is expected to be high. The mean max-Tanimoto across all test compounds is **0.53**, peaking squarely in the 0.45–0.55 range.

**The challenge here is not structural novelty — it is activity cliffs and potency tail prediction.** Our best Chemprop model produced a test prediction ceiling of **pEC50 = 5.94**, while the training set reaches **7.55**. Only 64 of 3,743 training compounds (1.7%) have pEC50 ≥ 6 — the model has very few high-potency scaffold anchors to extrapolate from, even when test compounds are structurally similar to training.

### Interactive Chemical Space Map (t-SNE)

We computed a t-SNE embedding of all 4,652 compounds (training + test) using:

```
ECFP4 fingerprints (2048-bit)
    → PCA (50 components)
    → t-SNE (perplexity=50, 500 iterations, PCA init)
```

**<a href="https://gashawmg.github.io/PXR-activity-pEC50-prediction/tsne_interactive_v2.html" target="_blank" rel="noopener noreferrer">→ Open interactive t-SNE map</a>**  
*(Hover over any compound to see its name and predicted pEC50. Viridis colour scale = training pEC50. Blue circles = test compounds.)*

![t-SNE chemical space](tsne_white.png)

**Key insight from the t-SNE:** Test compounds overlap well with training clusters throughout the chemical space. The prediction difficulty is concentrated in **potency extrapolation** — predicting fine-grained pEC50 differences within structurally similar analog series — not in structural coverage.

---

## Models

### Preliminary: Classical ML Baseline (ECFP4 + Meta-Learner)

Before building any deep learning pipeline, a classical ML stack was evaluated to establish a competitive baseline. Three models were trained on ECFP4 fingerprints and stacked via a meta-learner:

- **LightGBM (LGBM)** — gradient boosting on bit-vector fingerprints
- **HistGradientBoosting (HGB)** — sklearn's histogram-based boosting
- **Support Vector Regression (SVR)** — RBF kernel

| Metric | Internal Test | Leaderboard |
|---|---|---|
| MAE | 0.500 | 0.5196 |
| RAE | n/r | 0.6526 |
| R² | 0.67 | 0.457 |
| Spearman ρ | n/r | 0.7258 |
| Kendall's τ | n/r | 0.5280 |
| **Rank** | n/r | **~36 (at submission time)** |

*n/r = not recorded at the time of the experiment.*

The leaderboard MAE of **0.5196** was 16% worse than the final deep learning ensemble (0.4468). More telling is the Spearman drop: 0.7258 vs 0.8463 — classical ML struggled to rank compounds correctly within analog series, which is precisely the activity cliff problem that graph neural networks handle better through learned structural representations.

This result established that ECFP4 fingerprints + classical ML had hit a ceiling and motivated the switch to Chemprop D-MPNN and UniMol.

### Chemprop (2D Message-Passing Neural Network)

<a href="https://github.com/chemprop/chemprop" target="_blank" rel="noopener noreferrer">Chemprop</a> implements a directed message-passing neural network (D-MPNN) on 2D molecular graphs. We augmented it with **61 PXR-specific physicochemical descriptors** computed via RDKit:

- Lipophilicity & polarity: LogP, TPSA, polar surface ratio, MolMR
- Shape: Kappa1/2/3, NPR1/NPR2, Asphericity, FractionCSP3
- Rings: fused ring count, aromatic/saturated/aliphatic ring counts, steroid scaffold score
- PXR-relevant pharmacophores: sulfoxide count, ketone count, hydroxyl, Michael acceptors, electrophilic sp² centres
- Drug-likeness: QED, HBA, HBD, rotatable bonds, TPSA, Labute ASA

**Final configuration (v4_3_3):**

| Parameter | Value | Why |
|---|---|---|
| Loss | MSE | Huber and MAE/L1 both tested, both worse |
| Y-scaler | RobustScaler (1–99%) | Robust to high-pEC50 outliers |
| Sample weights | None | Uniform weighting is optimal |
| CV | 10-fold activity-stratified scaffold | Simulates novel scaffold prediction |
| Ensemble | 1/MAE weighted across folds | Lower-MAE folds contribute more |
| **OOF MAE** | **0.4420** | |
| **LB MAE** | **0.4622 (Rank 36)** | |

**Cross-validation fold count matters enormously.** We tested 10, 15, and 20 folds:

| Folds | Val set/fold | LB MAE | Rank |
|---|---|---|---|
| **10** | ~374 | **0.4622** | **36** ✅ |
| 15 | ~250 | 0.4702 | 48 ❌ |
| 20 | ~187 | 0.4750 | 52 ❌ |

Smaller validation sets make early stopping unreliable — the model terminates too early or too late based on noise, producing inconsistent fold quality.

### UniMol (3D Molecular Transformer)

<a href="https://github.com/dptech-corp/Uni-Mol" target="_blank" rel="noopener noreferrer">UniMol</a> is a universal 3D molecular pretraining framework. It operates on explicit 3D atomic coordinates (ETKDG v3 conformers, hydrogens retained), capturing spatial relationships invisible to 2D graph methods.

- **Pretraining:** 200 million molecular conformers (UniMol v2, 84M parameters)
- **Fine-tuning:** 3,743 competition compounds, 8-fold scaffold CV on Kaggle T4 × 2 GPU
- **LB MAE: 0.4615 (Rank 33), Spearman: 0.8306**

The same degradation pattern with fold count appeared: UniMol 12-fold (OOF MAE 0.3326 vs LB MAE 0.4788) showed an extreme train-test gap — a clear overfitting signal. 8-fold was the sweet spot for this dataset size.

---

## What We Tried (and What Failed)

This section documents every significant experiment, because understanding what *doesn't* work is as important as the final result.

*\* Classical ML was submitted earlier in the competition when fewer teams had submitted; current equivalent rank would be significantly lower given the final leaderboard standings.*

### The core challenge: OOF MAE is not a reliable guide

The most important methodological finding was a consistent pattern: **every technique that improved out-of-fold (OOF) cross-validation MAE worsened the leaderboard MAE**. This reflects a fundamental distribution shift — the training set (filtered by selectivity delta) is not representative enough of the analog-expansion test set to use OOF as a reliable proxy for generalisation.

| Experiment | OOF MAE | LB MAE | Rank | Key lesson |
|---|---|---|---|---|
| Classical ML (LGBM+HGB+SVR meta-learner) | n/r | 0.5196 | ~36* | ❌ ECFP4 fingerprints hit a ceiling — motivated switch to GNNs |
| v4_3_3 MSE baseline | 0.4420 | 0.4622 | 36 | ✅ Hard ceiling |
| MAE/L1 loss (v4_3_4) | 0.4369 ↓ | 0.4674 ↑ | 62 | Better OOF, worse LB — MAE loss median-pulls |
| QuantileTransformer scaler (v4_3_5) | 0.4546 | 0.4748 | 68 | Poor convergence (avg 24 vs 41 epochs) |
| SC binary pre-training (v4_3_6) | **0.4338** ↓ | **0.4943** ↑ | **108** | Worst experiment — see below |
| Hand-picked +5 compounds | 0.4396 ↓ | 0.4751 ↑ | 67 | OOF/LB gap = 0.036 — largest observed |
| + 37 ChEMBL compounds | 0.4463 | 0.5051 | 80 | Assay heterogeneity |
| Piecewise stretch post-processing | n/r | always worse | n/r | Compression is distributional, not calibrational |

### The SC pre-training experiment (v4_3_6) — a cautionary tale

The competition provides `single_concentration.csv`: 21,003 measurements (10,870 unique SMILES) from the primary PXR screen with binary hit/non-hit labels (criterion: log2_fc > 1 AND FDR-adjusted p < 0.05). We hypothesised that pre-training an MPNN binary classifier on this data, then transferring the message-passing weights into a regression fine-tuning run, would give the model prior knowledge of PXR-relevant structural features.

The results were sobering:

| Metric | SC pre-train | Baseline | Change |
|---|---|---|---|
| OOF MAE | 0.4338 | 0.4420 | ↓ Better |
| OOF R² | 0.7024 | ~0.51 | ↑ Suspicious jump |
| **LB MAE** | **0.4943** | **0.4622** | **↑ Much worse** |
| **LB Rank** | **108** | **36** | **↓ 72 places** |

Three root causes:

1. **Wrong training objective.** Binary hit/non-hit at 1–99 µM concentrations teaches the model to recognise "binder vs non-binder" — a coarse distinction. pEC50 regression requires learning *how potent* a binder is, which is a fundamentally different and more nuanced signal. Pre-training encodes the wrong gradient.

2. **Weight conflict during fine-tuning.** The transferred message-passing weights create a prior that fine-tuning must overcome. Best epoch per fold: [17, 49, 9, 29, 48, 10, 22, 33, 22, 14] — extreme variance. Some folds stop at epoch 9–10 before the prior is properly overridden; others run 48–49 epochs trying to fight it. Inconsistent fold models → poor ensemble quality.

3. **OOF R² jump 0.51 → 0.70 is an overfitting signature.** The model fits the training distribution better but the learned features don't generalise to blind test scaffolds.

> The right way to use SC data is **multi-task learning**: train a single MPNN simultaneously on pEC50 regression AND binary SC hit classification, with joint gradient flow through the shared encoder. All tasks update the message-passing weights together throughout training, rather than sequentially. This is on the Phase 2 agenda.

---

## Best Ensemble

The final best submission is a weighted average of three models:

| Model | Weight | Training data | Architecture |
|---|---|---|---|
| UniMol 8-fold (Kaggle T4 × 2) | **35%** | 3,743 compounds | 3D transformer (84M params) |
| Chemprop 10-fold | **35%** | 3,743 compounds | D-MPNN + 61 descriptors |
| UniMol 10-fold (2,948 compounds) | **30%** | 2,948 compounds | 3D transformer (84M params) |

The third model — UniMol trained on the smaller 2,948-compound set — was retained because it captures different error patterns from the 8-fold model (Pearson r = 0.937 between them; r = 0.922 between UniMol-8f and Chemprop). All pairwise correlations remain below 0.95, which is the practical threshold for meaningful ensemble benefit.

### Phase 1 leaderboard performance

| Metric | Best blend | Chemprop alone | UniMol alone |
|---|---|---|---|
| MAE | **0.4468** | 0.4622 | 0.4615 |
| RAE | **0.5606** | 0.5800 | 0.5793 |
| R² | **0.5459** | 0.5117 | 0.5515 |
| Spearman ρ | **0.8463** | 0.8137 | 0.8306 |
| Kendall's τ | **0.6567** | 0.6245 | 0.6391 |
| Rank | **35** | 36 | 33 |

Net improvement from initial Chemprop baseline to final blend: **~0.023 MAE units**, achieved entirely through data curation, architecture selection, hyperparameter optimisation, and ensemble design — no proprietary data, no additional assay measurements.

---

## What Top Teams Are Likely Doing Differently

The gap between our best (MAE 0.4468) and the leaderboard leader (MAE 0.3912) is 12.5%. More informatively, their **R² = 0.6634 vs our 0.5459** — they are explaining 11 additional percentage points of variance. This magnitude cannot come from hyperparameter tuning or more data of the same type; it requires capturing a fundamentally different source of molecular signal.

### 3D structure-based modelling

PXR has well-characterised crystal structures with bound ligands (PDB: 1ILG, 1M13, 3CTB, 3HVL and others). Top teams are almost certainly using **<a href="https://github.com/gnina/gnina" target="_blank" rel="noopener noreferrer">GNINA</a>** — an open-source neural network scoring function that generates docked poses and scores them using a CNN trained on protein-ligand affinity data. Unlike classical AutoDock Vina scores (raw force-field estimates), GNINA produces binding affinity estimates that are semantically comparable to pEC50.

We did try docking: AutoDock Vina scores as an additional descriptor caused rank to drop from 26 to 55. The lesson is that *how* you incorporate docking matters enormously. Vina raw scores add noise; learned scoring functions add signal.

### Multi-task learning (done correctly)

Training a single MPNN with separate output heads for pEC50 regression, log₂FC regression from single-concentration data, and binary hit classification — all simultaneously — allows the shared message-passing encoder to benefit from all three signals without the "wrong prior" problem of sequential pre-training.

### Newer foundation models

<a href="https://arxiv.org/abs/2406.14769" target="_blank" rel="noopener noreferrer">Uni-Mol2</a> (2024, scaling up to 1.1B parameters) and 3D-aware equivariant networks (MACE-OFF, NequIP) represent the current frontier. With 84M-parameter UniMol v2, we were already using a strong foundation, but the gap suggests the leading teams may be using larger or more recently fine-tuned models.


---

## Repo Contents

| File | Description |
|---|---|
| <a href="https://gashawmg.github.io/PXR-activity-pEC50-prediction/tsne_interactive_v2.html" target="_blank" rel="noopener noreferrer">tsne_interactive_v2.html</a> | Interactive Plotly t-SNE map of all 4,652 compounds — hover for name + pEC50 |
| [`tsne_white.png`](tsne_white.png) | Static t-SNE figure (white background, print-quality) |

---

## References & Resources

- **Challenge:** <a href="https://huggingface.co/spaces/openadmet/pxr-challenge" target="_blank" rel="noopener noreferrer">OpenADMET PXR Challenge HuggingFace Space</a>
- **Data:** <a href="https://huggingface.co/datasets/openadmet/pxr-challenge-train-test" target="_blank" rel="noopener noreferrer">HuggingFace dataset: openadmet/pxr-challenge-train-test</a>
- **Tutorial:** <a href="https://github.com/OpenADMET/PXR-Challenge-Tutorial" target="_blank" rel="noopener noreferrer">OpenADMET PXR Challenge Tutorial (GitHub)</a>
- **Chemprop:** <a href="https://github.com/chemprop/chemprop" target="_blank" rel="noopener noreferrer">github.com/chemprop/chemprop</a> · <a href="https://doi.org/10.1021/acs.jcim.3c01250" target="_blank" rel="noopener noreferrer">Heid et al., JCIM 2024</a>
- **UniMol:** <a href="https://github.com/dptech-corp/Uni-Mol" target="_blank" rel="noopener noreferrer">github.com/dptech-corp/Uni-Mol</a> · <a href="https://openreview.net/forum?id=6K2RM6wVqKu" target="_blank" rel="noopener noreferrer">Zhou et al., ICLR 2023</a>
- **GNINA:** <a href="https://github.com/gnina/gnina" target="_blank" rel="noopener noreferrer">github.com/gnina/gnina</a> · <a href="https://doi.org/10.1186/s13321-021-00522-2" target="_blank" rel="noopener noreferrer">McNutt et al., J Cheminform 2021</a>
- **RDKit:** <a href="https://www.rdkit.org" target="_blank" rel="noopener noreferrer">www.rdkit.org</a>

---

*Experimental PXR data generated by the team at Octant Bio (Sam Sabaat, Scott Simpkins, Yuning Shen, Bryan Jiang, Henry Chan, Jeff Tang, Ayesha Ghazali, Theo Tarver, Steven Edgar, Dominic Ky, and many others). Crystallographic data determined by Galen Correy, Nikhil Gupta, and the Fraser Lab at UCSF.*
                                                                                                                                                                                                                                                                                                                                                       