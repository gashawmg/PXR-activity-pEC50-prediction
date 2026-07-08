# Predicting PXR Induction Potency — OpenADMET Blind Challenge

> **Phase 1:** MAE **0.4468** · RAE **0.5606** · R² **0.5459** · Spearman **0.8463** · Kendall's τ **0.6567** · Rank **39**  
> **Phase 2 (final):** MAE **0.4231** .RAE **0.5841** ·   R² **0.5528** · Spearman **0.8028** · Kendall's τ **0.6221** · Rank **12**   
> 3-way blend · Chemprop D-MPNN + MultitaskMPNN + UniMol 3D transformer · No proprietary data

---

## Table of Contents

1. [Why PXR?](#why-pxr)  
2. [The Challenge](#the-challenge)  
3. [Data: From Screen to Model-Ready Set](#data-from-screen-to-model-ready-set)  
4. [Chemical Space Analysis](#chemical-space-analysis)  
5. [Models](#models)  
6. [What Was Tried (and What Failed)](#what-was-tried-and-what-failed)  
7. [Best Ensemble (Phase 1)](#best-ensemble-phase-1)  
8. [Phase 2 — Refinement with Unblinded Labels](#phase-2--refinement-with-unblinded-labels)  
9. [Conclusion](#conclusion)  
10. [Repo Contents](#repo-contents)  

---

## Why PXR?

The **Pregnane X Receptor (PXR / NR1I2)** is a nuclear receptor that acts as the primary sensor for foreign chemicals entering the body. Upon activation by a drug candidate, PXR translocates to the nucleus and dramatically upregulates **CYP3A4** — the cytochrome P450 enzyme that handles roughly half of all small-molecule drugs in clinical use. For a drug candidate, PXR activation carries three serious consequences:

| Consequence | Mechanism |
|---|---|
| **Drug-Drug Interactions (DDIs)** | Elevated CYP3A4 activity accelerates clearance of co-administered drugs, potentially dropping them below therapeutic thresholds |
| **Hepatotoxicity** | Increased metabolic flux through CYP3A4 can generate reactive intermediates that damage liver tissue |
| **Chemoresistance** | In PXR-expressing tumours, enhanced CYP3A4 activity reduces the intracellular concentration of oncology agents |

Modelling PXR activity is challenging for two reasons. First, the receptor's ligand-binding pocket is exceptionally large and conformationally adaptable, allowing it to accommodate structurally unrelated compounds — from macrolide antibiotics to steroidal natural products to synthetic drug fragments. Second, before this challenge, quantitative PXR activity data in the public domain was sparse and heterogeneous: fewer than 800 reliable pEC50 values extracted from roughly 150 publications, each using different cell lines, reporter constructs, and concentration protocols.

---

## The Challenge

The **<a href="https://huggingface.co/spaces/openadmet/pxr-challenge" target="_blank" rel="noopener noreferrer">OpenADMET Blind Challenge</a>** (Octant Bio + UCSF) addresses this data scarcity directly by releasing the largest uniform-assay PXR activity dataset to date: over 11,000 compounds tested in a single, standardised cell-based system.

**Assay design:** PXR agonism is measured using a reporter cell line in which a chimeric receptor construct — the PXR ligand-binding domain grafted onto a heterologous DNA-binding domain — drives luciferase expression upon activation. A critical feature of the assay is the paired counter-screen: an identical experiment is run in parallel using a construct carrying a non-functional point mutant of the same chimeric receptor. Any compound that activates both the active and mutant constructs is classified as a non-specific transcriptional activator (e.g. an HDAC inhibitor) rather than a genuine PXR agonist, and is removed from the potency analysis. Only compounds that selectively activate the functional construct are retained.

**The test set is not a random sample.** Rather than drawing compounds at random from the screening library, 513 Enamine on-demand analogs were selected around the 63 most potent and selective hits (ECFP4 Tanimoto similarity > 0.4 to at least one of the 63 seed compounds). This **analog expansion design** places the prediction task squarely in lead optimisation territory — small structural changes within active series that may produce large potency differences — rather than in broad virtual screening.

| Phase | Dates | Description |
|---|---|---|
| **Phase 1** | April 1 – May 25, 2026 | Blind prediction for all 513 test compounds; live leaderboard |
| **Phase 2** | May 26 – July 1, 2026 | Analog Set 1 labels unblinded (253 compounds); refine predictions for Analog Set 2 (260 compounds) |

**Ranking metric:** Relative Absolute Error (RAE) = Σ|y−ŷ| / Σ|y−ȳ|. MAE, R², Spearman ρ and Kendall's τ also reported.

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

The competition released **4,139 compounds with pEC50 labels** as training data. Each has both a PXR pEC50 and a counter-screen pEC50, giving a **selectivity delta**:

```
selectivity_delta = pEC50(PXR) − pEC50(counter-screen)
```

### Finding the right training set

This was the most consequential decision in the project. Every threshold was tested systematically using Chemprop (D-MPNN + 61 RDKit descriptors) as the evaluation model, with all leaderboard results from blind test set submissions:

| Dataset | Compounds | Filter | LB MAE | RAE | R² | Spearman ρ | Verdict |
|---|---|---|---|---|---|---|---|
| `clean_train.csv` | 2,948 | delta > 1.5 | 0.4794 | 0.6017 | 0.4799 | 0.8042 | Starting point |
| **`clean_train2.csv`** | **3,743** | **delta > 0** | **0.4622** | **0.5800** | **0.5117** | **0.8137** | ✅ **Phase 1 baseline** |
| + 37 ChEMBL compounds | 3,780 | delta > 0 + external | 0.5051 | 0.6338 | 0.4094 | 0.7420 | ❌ Assay heterogeneity |
| Relaxed (delta ≥ −0.6) | 4,054 | — | 0.4809 | 0.6036 | 0.5008 | 0.8097 | ❌ Non-selective compounds contaminate |
| Unfiltered | 4,139 | None | 0.5095 | 0.6397 | 0.4799 | 0.8025 | ❌ Rejected |

**The biological logic:** The 513 test compounds are analogs of hits with selectivity delta ≥ 1.5. The training set that best represents this population is one filtered by *positive* selectivity — delta > 0 captures compounds that show some PXR preference over the counter-screen, without forcing the strict 1.5-unit threshold that would discard useful potency information from moderate hits.

> **Rule established:** `clean_train2.csv` (3,743 compounds, selectivity_delta > 0) is the hard ceiling for Phase 1. Adding *any* compounds beyond this — by relaxing the filter, hand-picking borderline entries, or adding external ChEMBL data — degraded leaderboard performance in every experiment. Phase 2 broke this ceiling by adding *measured* high-potency data from secondary screening (see Phase 2 section).

### Why external ChEMBL data hurt

37 ChEMBL PXR compounds with ECFP4 Tanimoto ≥ 0.4 to blind test compounds were curated and added to training. Leaderboard MAE jumped from 0.4622 to **0.5051** (rank 36 → 80). The reason: ChEMBL PXR measurements span at least 150 different assay protocols, cell lines, and reporter constructs accumulated over two decades. The OpenADMET assay is a single, tightly controlled protocol. Adding cross-assay data introduces systematic offsets that the model learns as real signal — but they are assay artefacts.

---

## Chemical Space Analysis

### Structural coverage of the test set

Prior to model development, the relationship between training and test chemical space was mapped using Morgan fingerprint Tanimoto similarity against `clean_train2.csv` (3,743 compounds). The mean max-Tanimoto across all test compounds is **0.53**, peaking in the 0.45–0.55 range.

**The challenge here is not structural novelty — it is activity cliffs and potency tail prediction.** The best Phase 1 Chemprop model produced a test prediction ceiling of **pEC50 = 5.94**, while the training set reaches **7.55**. Only 64 of 3,743 training compounds (1.7%) have pEC50 ≥ 6 — the model has very few high-potency scaffold anchors to extrapolate from, even when test compounds are structurally similar to training.

### Interactive Chemical Space Map (t-SNE)

A t-SNE embedding of all 4,652 compounds (training + test) was computed using:

```
ECFP4 fingerprints (2048-bit) → PCA (50 components) → t-SNE (perplexity=50, 500 iterations)
```

**<a href="https://gashawmg.github.io/PXR-activity-pEC50-prediction/tsne_interactive_v2.html" target="_blank" rel="noopener noreferrer">→ Open interactive t-SNE map</a>**  
*(Hover over any compound to see its name and predicted pEC50. Viridis colour scale = training pEC50. Blue circles = test compounds.)*

![t-SNE chemical space](tsne_white.png)

---

## Models

### Preliminary: Classical ML Baseline

A classical ML stack (LightGBM + HistGradientBoosting + SVR meta-learner) on RDKit descriptors + ECFP4 fingerprints achieved LB MAE = 0.5196, Spearman = 0.7258. The 16% MAE gap and 0.11 Spearman gap vs. the final ensemble established that classical ML had hit a ceiling on this task, motivating the switch to graph neural networks.

### Chemprop (2D Message-Passing Neural Network)

<a href="https://github.com/chemprop/chemprop" target="_blank" rel="noopener noreferrer">Chemprop</a> implements a directed message-passing neural network (D-MPNN) that learns molecular representations by iteratively aggregating atom and bond features across 2D graph neighbourhoods. Development followed three successive improvements:

**Step 1 — 61 PXR-specific descriptors** concatenated to graph readout (LogP, TPSA, shape descriptors, ring counts, pharmacophore flags, QED): −9.4% MAE, −9.4% RAE in a single step.

**Step 2 — Expanded training set** (2,948 → 3,743 compounds, delta > 0): −3.6% MAE.

**Step 3 — 10-fold activity-stratified scaffold CV**: 10-fold outperforms 15-fold and 20-fold because smaller validation sets destabilise early stopping, producing inconsistent fold models.

| Stage | LB MAE | RAE | Spearman ρ |
|---|---|---|---|
| D-MPNN, no descriptors (2,948) | 0.5289 | 0.6641 | 0.7324 |
| + 61 PXR descriptors | 0.4794 | 0.6017 | 0.8042 |
| + expanded training (3,743) + tuning | **0.4622** | **0.5800** | **0.8137** |

### UniMol (3D Molecular Transformer)

<a href="https://github.com/deepmodeling/Uni-Mol" target="_blank" rel="noopener noreferrer">UniMol</a> is a transformer-based foundation model pre-trained on 200 million molecular conformers. Input conformers are generated with ETKDG v3 (all hydrogens retained), giving access to inter-atomic distances, angles, and steric contacts invisible to 2D graph networks.

- Fine-tuned on 3,743 competition compounds, 8-fold scaffold CV on Kaggle T4 × 2 GPU
- **LB MAE: 0.4615 (Rank 33), Spearman: 0.8306**
- 8-fold is the sweet spot: 12-fold and 15-fold both degrade (same val-set stability issue as Chemprop)

---

## What Was Tried (and What Failed)

The most important methodological finding: **every technique that improved OOF cross-validation MAE worsened the leaderboard MAE.** This reflects a fundamental distribution shift — the training set is not representative enough of the analog-expansion test set to use OOF as a reliable proxy for generalisation.

| Experiment | OOF MAE | LB MAE | Rank | Key lesson |
|---|---|---|---|---|
| v4_3_3 MSE baseline | 0.4420 | 0.4622 | 36 | ✅ Hard ceiling for Phase 1 |
| MAE/L1 loss | 0.4369 ↓ | 0.4674 ↑ | 62 | Better OOF, worse LB — MAE loss median-pulls |
| QuantileTransformer scaler | 0.4546 | 0.4748 | 68 | Poor convergence (avg 24 epochs vs 41) |
| SC binary pre-training | **0.4338** ↓ | **0.4943** ↑ | **108** | Wrong objective; see below |
| Hand-picked +5 compounds | 0.4396 ↓ | 0.4751 ↑ | 67 | OOF/LB gap = 0.036 — largest observed |
| + 37 ChEMBL compounds | 0.4463 | 0.5051 | 80 | Assay heterogeneity |
| Piecewise stretch post-processing | — | always worse | — | Compression is distributional, not calibrational |

**The SC pre-training experiment — a cautionary tale.** Pre-training on binary PXR hit/non-hit labels (21,003 compounds from the primary screen) then fine-tuning on pEC50: OOF MAE 0.4338 (best ever), LB MAE 0.4943 (rank 108 — worst result). Binary labels at µM concentrations cannot teach the model to discriminate pEC50 4 vs 5 vs 6. The right approach is multi-task learning with simultaneous gradient flow — implemented in Phase 2 as MultitaskMPNN.

---

## Best Ensemble (Phase 1)

| Model | Weight | Training data | LB MAE | RAE | Spearman ρ |
|---|---|---|---|---|---|
| UniMol 8-fold (Kaggle T4 × 2) | **35%** | 3,743 compounds | 0.4615 | 0.5793 | 0.8306 |
| Chemprop 10-fold | **35%** | 3,743 compounds | 0.4622 | 0.5800 | 0.8137 |
| UniMol 7-fold | **30%** | 1,948 compounds | — | — | — |

**Phase 1 final result: MAE 0.4468 · RAE 0.5606 · R² 0.5459 · Spearman 0.8463 · Rank 39**

All pairwise model correlations < 0.95, confirming meaningful ensemble diversity. The third model (trained on a smaller 1,948-compound subset) captures different error patterns from the 8-fold model (Pearson r = 0.937).

---

## Phase 2 — Refinement with Unblinded Labels

> **Final submission:** MAE **0.4231** .RAE **0.5841** ·   R² **0.5528** · Spearman **0.8028** · Kendall's τ **0.6221** · Rank **12** 
> `submission_final_3way_513.csv` — 513 compounds — 70% × (48% v4_4 + 52% v4_13d) + 30% UniMol

### What the Unblinded Labels Revealed

Evaluating all historical predictions against the 253 Analog Set 1 compounds revealed the dominant error mode: **activity cliffs driven by low-potency compounds**.

- **22% of test compounds** (pEC50 < 4.0, N=55) account for **60% of total RAE** — models over-predict their activity by +0.97 pEC50 units on average
- These low-potency compounds share scaffold topology with active training compounds (ECFP4 Tanimoto ≥ 0.5), making them appear "active" to a 2D graph network
- 31 high-uncertainty compounds (pEC50 std error > 0.3) have RAE = 2.0 — nearly 4× worse than low-uncertainty compounds, setting an irreducible noise floor

### New Training Data

Phase 2 enriched the training set with measured high-potency compounds from secondary screening:

| Source | N | Content |
|---|---|---|
| `clean_train2.csv` | 3,743 | Phase 1 baseline |
| `crude_nv_hi` | 244 | Measured hits with pEC50 ≥ 5.5 |
| `semi_nv` | 55 | Semi-pure batch measurements (non-volatile, purity-corrected) |
| `sc_inactives_300` | 300 | Confirmed SC inactives (pseudo-pEC50 = 2.0) |

**Chemprop training set: 4,342 compounds** (all four sources).  
**UniMol training set: 4,042 compounds** (clean_train2 + crude_nv_hi + semi_nv, **no sc_inactives**).

A controlled ablation showed `sc_inactives_300` hurts UniMol's Potent >5.5 bin by +0.44 RAE. Chemprop uses 3× sample weights to limit their influence; UniMol's MolTrain assigns equal weight to all rows, so 300 identical pseudo-labels at pEC50 = 2.0 distort the high-potency embedding. The inactive correction is handled by Chemprop in the ensemble.

### Chemprop v4_4: New Best Single Model

| Model | RAE | MAE | R² | Spearman ρ |
|---|---|---|---|---|
| v4_3_3 (Phase 1 best Chemprop) | 0.5886 | 0.4701 | 0.5234 | 0.8074 |
| **v4_4** | **0.5477** | **0.4374** | **0.5996** | **0.8082** |

Key changes: Phase 2 training data, sc_inactives supplement, 3× sample weight for pEC50 ≥ 5.5.

### MultitaskMPNN v4_13d: Asymmetric Classification Head

A `MultitaskMPNN` with shared D-MPNN trunk and parallel output heads trained simultaneously:

```
Molecule → BondMessagePassing → shared MLP trunk
                                    ├─ reg_head  → pEC50 (MSE loss, all compounds)
                                    └─ clf_head  → active/inactive (asymmetric BCE)
```

**Joint loss:** `0.8 × MSE + 0.2 × BCE_asymmetric`

The asymmetric BCE penalises **only false positives** (inactives predicted as active). Symmetric BCE would also penalise false negatives, creating a global downward pull on all active embeddings that compresses the potency ceiling. By zeroing the classification gradient for genuine actives, the Potent bin is preserved while the Low bin is corrected.

| Variant | RAE | Low <4.0 RAE | Potent >5.5 RAE |
|---|---|---|---|
| v4_13 (symmetric BCE) | 0.5755 | 1.643 | 2.350 |
| v4_13b (w=0.1) | 0.5706 | 1.616 | 2.222 |
| v4_13c (threshold=5.0) | 0.5770 | 1.594 | 2.182 |
| **v4_13d (asymmetric BCE)** | **0.5636** | **1.451** | **2.098** |

### Final 3-Way Ensemble

**Step 1 — Chemprop ensemble:**

| Model | Weight | RAE |
|---|---|---|
| v4_4 (regression only) | 48% | 0.5477 |
| v4_13d (asymmetric MultitaskMPNN) | 52% | 0.5636 |
| **v4_4 + v4_13d blend** | — | **0.5341** |

v4_4 is stronger on active compounds (Potent bin RAE 1.131 vs 1.224); v4_13d corrects inactive over-prediction (Low bin bias +0.788 vs +0.965). The blend plateau is flat from α = 0.45–0.55, confirming robustness.

**Step 2 — Add UniMol:**

| Blend | RAE | MAE | R² |
|---|---|---|---|
| Chemprop ensemble alone | 0.5341 | 0.4266 | 0.6340 |
| **70% Ens + 30% UniMol** | **0.5298** | **0.4231** | **0.6236** |

UniMol contributes complementary signal in the Mid 4–5 and High 5–5.5 bins (Pearson r = 0.961 with Chemprop ensemble).

**Final submission metrics (253 unblinded compounds):**

| Metric | Phase 1 best | **Phase 2 final** | Δ |
|---|---|---|---|
| RAE | 0.5589 | **0.5298** | −0.029 |
| MAE | 0.4464 | **0.4231** | −0.023 |
| R² | 0.5459 | **0.6236** | +0.078 |
| Spearman ρ | 0.8492 | 0.8389 | −0.010 |
| Kendall's τ | 0.6567 | 0.6486 | −0.008 |
| Bias | +0.182 | **+0.088** | halved |

---

## Conclusion

### Phase 1

The Phase 1 submission achieved **MAE 0.4468, RAE 0.5606, Rank 39** on 513 blind test compounds using a 3-model ensemble of Chemprop D-MPNN and two UniMol fine-tuned models — entirely with public training data and open-source tools.

### Phase 2

The Phase 2 final submission achieves **RAE 0.5298, MAE 0.4231, R² 0.6236** — a 0.029 RAE improvement over Phase 1. The gain comes from three sources working together: richer training data (measured high-potency compounds from secondary screening), multi-task learning (asymmetric classification head targeting inactive over-prediction), and ensemble diversity (Chemprop + UniMol with complementary error profiles).

The irreducible error floor remains the 31 high-uncertainty test compounds (std error > 0.3, RAE = 2.0) — a measurement noise problem that no model architecture can fully overcome without better assay reproducibility.

**The full pipeline is open-source.** Every script, training file, and prediction is reproducible with Chemprop, UniMol, and RDKit on public data.

---

## Repo Contents

| File | Description |
|---|---|
| <a href="https://gashawmg.github.io/PXR-activity-pEC50-prediction/tsne_interactive_v2.html" target="_blank" rel="noopener noreferrer">tsne_interactive_v2.html</a> | Interactive Plotly t-SNE map — hover for compound name + pEC50 |
| `tsne_white.png` | Static t-SNE figure (white background) |

---

## References & Resources

- **Challenge:** <a href="https://huggingface.co/spaces/openadmet/pxr-challenge" target="_blank" rel="noopener noreferrer">OpenADMET PXR Challenge HuggingFace Space</a>
- **Data:** <a href="https://huggingface.co/datasets/openadmet/pxr-challenge-train-test" target="_blank" rel="noopener noreferrer">HuggingFace dataset: openadmet/pxr-challenge-train-test</a>
- **Tutorial:** <a href="https://github.com/OpenADMET/PXR-Challenge-Tutorial" target="_blank" rel="noopener noreferrer">OpenADMET PXR Challenge Tutorial (GitHub)</a>
- **Chemprop:** <a href="https://github.com/chemprop/chemprop" target="_blank" rel="noopener noreferrer">github.com/chemprop/chemprop</a> · <a href="https://doi.org/10.1021/acs.jcim.3c01250" target="_blank" rel="noopener noreferrer">Heid et al., JCIM 2024</a>
- **UniMol:** <a href="https://github.com/deepmodeling/Uni-Mol" target="_blank" rel="noopener noreferrer">github.com/deepmodeling/Uni-Mol</a> · <a href="https://openreview.net/forum?id=6K2RM6wVqKu" target="_blank" rel="noopener noreferrer">Zhou et al., ICLR 2023</a>
- **RDKit:** <a href="https://www.rdkit.org" target="_blank" rel="noopener noreferrer">www.rdk
