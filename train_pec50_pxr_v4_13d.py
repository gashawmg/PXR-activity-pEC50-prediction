"""
pEC50 PXR -- Pipeline v4_13d
==============================
Base    : v4_13 (multi-task MPNN, regression + classification head)

Changelog v4_13 -> v4_13d
---------------------------
CHANGED  Asymmetric classification loss (false-positive penalty only).

  v4_13 used standard symmetric BCE:
    loss = (1-w)*MSE + w*BCE(clf_logit, is_active)
  This penalizes BOTH:
    - False positives : inactive predicted as active  (what we want to fix)
    - False negatives : active predicted as inactive  (unintended side effect)
  The false-negative penalty pulled the entire active distribution downward,
  compressing the ceiling and hurting the Potent 5-6 bin (+0.31-0.34 RAE).

  v4_13d uses ASYMMETRIC BCE -- false-positive penalty only:
    clf_loss computed ONLY on inactive compounds (pEC50 < act_thresh)
    Active compounds contribute zero classification gradient.
  Result: the trunk only learns "don't encode inactives like actives."
  Genuine actives are never pulled toward the inactive zone.

KEPT  All v4_13 architecture, data pipeline, and hyperparameters.
      clf_weight=0.2, act_thresh=4.0 (same defaults as v4_13).

Usage (Windows CMD)
-------------------
  python train_pec50_pxr_v4_13d.py --mode BOTH
"""

import argparse, glob, json, os, pickle, random, re, time, warnings
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
from chemprop.nn.message_passing import BondMessagePassing
from chemprop.nn.agg import MeanAggregation
from chemprop.nn.predictors import RegressionFFN
from chemprop.models import MPNN

from rdkit import Chem, rdBase
from rdkit.Chem import (
    Descriptors, rdMolDescriptors as rdMD,
    GraphDescriptors, MolSurf, Crippen,
    rdFingerprintGenerator, DataStructs, AllChem,
)
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser, Uncharger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_CSV      = "train_unimol_e3_ct2_crhi_semi.csv"
INACTIVE_SUPPLEMENT_CSV = "sc_inactives_300.csv"  # 300 SC confirmed inactives (pEC50=2.0) — appended at train time
TEST_CSV       = "test.csv"
SMILES_COL     = "SMILES"
TARGET_COL     = "pEC50"
NAME_COL       = "Molecule Name"
MODEL_SAVE_DIR = "models_pxr_v4_13d"
OUTPUT_CSV     = "predictions_pxr_v4_13d.csv"

GLOBAL_SEED           = 42
N_FOLDS               = 10
OUTLIER_MAD_THRESHOLD = 3.5

SELECTIVITY_COL = "selectivity_delta"
STD_ERROR_COL   = "pEC50_std_error"
CI_LOWER_COL    = "pEC50_ci_lower"
CI_UPPER_COL    = "pEC50_ci_upper"

BUTINA_THRESHOLD  = 0.4
KMEANS_N_CLUSTERS = 15

# Tail upweight cap restored to 5x (same as v4_1 rank-26 run).
# Set to 1.0 to disable tail upweighting entirely.
TAIL_WEIGHT_MAX = 5.0

# Activity cliff upweighting (set CLIFF_WEIGHT_MULTIPLIER = 1.0 to disable)
CLIFF_WEIGHT_MULTIPLIER  = 2.0
CLIFF_TANIMOTO_THRESHOLD = 0.5
CLIFF_DELTA_THRESHOLD    = 1.0

ADAPTIVE_CONFIG = {
    'small':      {'threshold': 300,          'hidden_dim': 384, 'depth': 4, 'n_layers': 3,
                   'dropout': 0.30, 'max_epochs': 150, 'patience': 20, 'batch_size': 32,
                   'augmentation_factor': 5,  'use_augmentation': True},
    'medium':     {'threshold': 800,          'hidden_dim': 512, 'depth': 5, 'n_layers': 3,
                   'dropout': 0.25, 'max_epochs': 200, 'patience': 20, 'batch_size': 64,
                   'augmentation_factor': 3,  'use_augmentation': True},
    'large':      {'threshold': 2000,         'hidden_dim': 768, 'depth': 5, 'n_layers': 3,
                   'dropout': 0.20, 'max_epochs': 200, 'patience': 20, 'batch_size': 64,
                   'augmentation_factor': 2,  'use_augmentation': True},
    'very_large': {'threshold': float('inf'), 'hidden_dim': 768, 'depth': 6, 'n_layers': 4,
                   'dropout': 0.20, 'max_epochs': 300, 'patience': 25, 'batch_size': 64,
                   'augmentation_factor': 1,  'use_augmentation': True},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# =============================================================================
# PXR DESCRIPTOR LIST — 61 descriptors
# =============================================================================

_PXR_RAW = [
    # ── Hydrophobicity (9) ──────────────────────────────────────────────────
    'logp', 'MolMR', 'hydrophobic_vsa', 'aromatic_proportion',
    'aromatic_carbon_count', 'halogen_count', 'num_Cl', 'num_F', 'num_Br',

    # ── Size and weight (7) ─────────────────────────────────────────────────
    'mw', 'heavy_atom_count', 'rings', 'six_membered_rings',
    'five_membered_rings', 'fused_ring_count', 'bertz_ct',

    # ── H-bond acceptors (7) ────────────────────────────────────────────────
    'HBA', 'num_carbonyl', 'num_ketone', 'num_ether_O',
    'num_hydroxyl', 'num_sulfoxide', 'hba_vsa',

    # ── H-bond donors (3) ───────────────────────────────────────────────────
    'HBD', 'num_NH', 'hbd_vsa',

    # ── Topological shape (13) ──────────────────────────────────────────────
    'Kappa1', 'Kappa2', 'Kappa3', 'Chi0v', 'Chi1v', 'Chi2v',
    'HallKierAlpha', 'fractionCSP3', 'sp3_carbon_count', 'rotatable',
    'NPR1', 'NPR2', 'Asphericity',

    # ── Ring systems (5) ────────────────────────────────────────────────────
    'num_aromatic_rings', 'num_saturated_rings', 'num_aliphatic_rings',
    'steroid_scaffold_score', 'num_fused_aromatic',

    # ── Polarity and surface (6) ────────────────────────────────────────────
    'tpsa', 'labute_asa', 'polar_surface_ratio',
    'MaxPartialCharge', 'MinPartialCharge', 'charge',

    # ── Electronic (3) ──────────────────────────────────────────────────────
    'MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex',

    # ── SMR VSA (2) ─────────────────────────────────────────────────────────
    'SMR_VSA5', 'SMR_VSA7',

    # ── PXR-specific (3) ────────────────────────────────────────────────────
    'num_ring_hydroxyl', 'num_carboxyl', 'bile_acid_score',

    # ── Reactivity (2) ──────────────────────────────────────────────────────
    'num_michael_acceptors', 'num_electrophilic_sp2',

    # ── Drug-likeness (1) ───────────────────────────────────────────────────
    'qed',
]

PXR_DESCRIPTOR_LIST = list(dict.fromkeys(_PXR_RAW))
assert len(PXR_DESCRIPTOR_LIST) == len(set(PXR_DESCRIPTOR_LIST)), \
    "Duplicate descriptors"
assert len(PXR_DESCRIPTOR_LIST) == 61, \
    f"Expected 61 descriptors, got {len(PXR_DESCRIPTOR_LIST)}"
print(f"PXR descriptor count: {len(PXR_DESCRIPTOR_LIST)}")


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_global_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    pl.seed_everything(seed, workers=True)

set_global_seed(GLOBAL_SEED)
torch.set_float32_matmul_precision('medium')


# =============================================================================
# MOLECULE STANDARDIZER
# =============================================================================

class MoleculeStandardizer:
    def __init__(self):
        self.lfc = LargestFragmentChooser()
        self.uc  = Uncharger()

    def sanitize(self, smiles):
        try:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return None
            return self.uc.uncharge(self.lfc.choose(mol))
        except Exception:
            return None


# =============================================================================
# SMILES AUGMENTER
# =============================================================================

class SMILESAugmenter:
    def __init__(self, random_seed=GLOBAL_SEED):
        random.seed(random_seed)

    def enumerate_smiles(self, smiles, n_variants=5, include_original=True):
        variants = {smiles} if include_original else set()
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return list(variants)
            atoms    = list(range(mol.GetNumAtoms()))
            attempts = 0
            target   = n_variants + (1 if include_original else 0)
            while len(variants) < target and attempts < n_variants * 10:
                s = Chem.MolToSmiles(mol, rootedAtAtom=random.choice(atoms),
                                     canonical=False, doRandom=True)
                if s:
                    variants.add(s)
                attempts += 1
        except Exception:
            pass
        return list(variants)

    def augment_dataset(self, smiles_list, y_values, augmentation_factor=3,
                        weights=None):
        aug_smiles, aug_y, aug_w = [], [], []
        for i, (smi, y) in enumerate(zip(smiles_list, y_values)):
            for v in self.enumerate_smiles(smi, n_variants=augmentation_factor-1,
                                           include_original=True):
                aug_smiles.append(v)
                aug_y.append(y)
                if weights is not None:
                    aug_w.append(weights[i])
        return (aug_smiles, np.array(aug_y),
                np.array(aug_w) if weights is not None else None)


# =============================================================================
# PXR DESCRIPTOR CALCULATOR — 61 descriptors
# =============================================================================

class PXRDescriptorCalculator:
    """
    Calculates the 61-descriptor PXR feature vector for each molecule.
    No docking score — works with any SMILES input, no external files needed.
    """

    _SMARTS = {
        'carbonyl':          '[CX3]=[OX1]',
        'ketone':            '[#6][CX3](=O)[#6]',
        'ether_O':           '[OX2;!$([OX2H]);!$(OC=O)]',
        'hydroxyl':          '[OX2H]',
        'sulfoxide':         '[SX3](=O)',
        'NH':                '[NX3;H1,H2;!$(NC=O)]',
        'ring_hydroxyl':     '[OX2H;$([OX2H]~[r])]',
        'carboxyl':          '[CX3](=O)[OX2H1]',
        'michael_acceptor':  '[CX3;H0](=[OX1])[CX3]=[CX3]',
        'electrophilic_sp2': '[CX3;$([CX3](=[OX1])[OX2,NX3])]',
    }

    def __init__(self, descriptor_list, scaler_type='robust'):
        self.descriptor_list = list(descriptor_list)
        if len(self.descriptor_list) != len(set(self.descriptor_list)):
            raise ValueError("Duplicate descriptors in list")
        self.scaler = (RobustScaler(quantile_range=(5.0, 95.0))
                       if scaler_type == 'robust' else StandardScaler())
        self._compiled = {k: Chem.MolFromSmarts(v) for k, v in self._SMARTS.items()}

    def _match(self, mol, key):
        pat = self._compiled.get(key)
        try:
            return len(mol.GetSubstructMatches(pat)) if pat else 0
        except Exception:
            return 0

    def _calc(self, mol):
        if mol is None:
            return {}
        d = {}
        try:
            # ── Hydrophobicity ────────────────────────────────────────────────
            d['logp']  = Crippen.MolLogP(mol)
            d['MolMR'] = Crippen.MolMR(mol)
            sv = MolSurf.SlogP_VSA_(mol)
            d['hydrophobic_vsa'] = float(sum(sv[5:]))
            nh = Descriptors.HeavyAtomCount(mol)
            na = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
            d['aromatic_proportion']    = na / nh if nh > 0 else 0.0
            d['aromatic_carbon_count']  = float(sum(
                1 for a in mol.GetAtoms()
                if a.GetAtomicNum() == 6 and a.GetIsAromatic()))
            d['num_F']  = float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9))
            d['num_Cl'] = float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 17))
            d['num_Br'] = float(sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 35))
            d['halogen_count'] = d['num_F'] + d['num_Cl'] + d['num_Br']

            # ── Size ─────────────────────────────────────────────────────────
            d['mw']               = Descriptors.MolWt(mol)
            d['heavy_atom_count'] = float(nh)
            ri  = mol.GetRingInfo()
            rs  = [len(r) for r in ri.AtomRings()]
            d['rings']               = float(len(rs))
            d['six_membered_rings']  = float(sum(1 for s in rs if s == 6))
            d['five_membered_rings'] = float(sum(1 for s in rs if s == 5))
            rl = ri.AtomRings()
            d['fused_ring_count']    = float(sum(
                1 for i, r1 in enumerate(rl)
                for r2 in rl[i+1:] if len(set(r1) & set(r2)) >= 2))
            d['bertz_ct'] = GraphDescriptors.BertzCT(mol)

            # ── H-bond acceptors ──────────────────────────────────────────────
            d['HBA']           = float(Descriptors.NumHAcceptors(mol))
            d['num_carbonyl']  = float(self._match(mol, 'carbonyl'))
            d['num_ketone']    = float(self._match(mol, 'ketone'))
            d['num_ether_O']   = float(self._match(mol, 'ether_O'))
            d['num_hydroxyl']  = float(self._match(mol, 'hydroxyl'))
            d['num_sulfoxide'] = float(self._match(mol, 'sulfoxide'))
            pv = MolSurf.PEOE_VSA_(mol)
            d['hba_vsa'] = float(sum(pv[7:]))
            d['hbd_vsa'] = float(sum(pv[:7]))

            # ── H-bond donors ─────────────────────────────────────────────────
            d['HBD']    = float(Descriptors.NumHDonors(mol))
            d['num_NH'] = float(self._match(mol, 'NH'))

            # ── Topological shape ─────────────────────────────────────────────
            d['Kappa1']           = GraphDescriptors.Kappa1(mol)
            d['Kappa2']           = GraphDescriptors.Kappa2(mol)
            d['Kappa3']           = GraphDescriptors.Kappa3(mol)
            d['Chi0v']            = GraphDescriptors.Chi0v(mol)
            d['Chi1v']            = GraphDescriptors.Chi1v(mol)
            d['Chi2v']            = Descriptors.Chi2v(mol)
            d['HallKierAlpha']    = GraphDescriptors.HallKierAlpha(mol)
            d['fractionCSP3']     = Descriptors.FractionCSP3(mol)
            d['sp3_carbon_count'] = float(sum(
                1 for a in mol.GetAtoms()
                if a.GetAtomicNum() == 6
                and a.GetHybridization().name == 'SP3'))
            d['rotatable'] = float(Descriptors.NumRotatableBonds(mol))
            try:
                d['NPR1']        = rdMD.CalcNPR1(mol)
                d['NPR2']        = rdMD.CalcNPR2(mol)
                d['Asphericity'] = rdMD.CalcAsphericity(mol)
            except Exception:
                d['NPR1'] = d['NPR2'] = d['Asphericity'] = 0.0

            # ── Ring systems ──────────────────────────────────────────────────
            d['num_aromatic_rings']  = float(rdMD.CalcNumAromaticRings(mol))
            d['num_saturated_rings'] = float(rdMD.CalcNumSaturatedRings(mol))
            d['num_aliphatic_rings'] = float(rdMD.CalcNumAliphaticRings(mol))
            six = [set(r) for r in ri.AtomRings() if len(r) == 6]
            d['steroid_scaffold_score'] = float(sum(
                1 for i, r1 in enumerate(six)
                for r2 in six[i+1:] if len(r1 & r2) == 2))
            ar = [set(r) for r in ri.AtomRings()
                  if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in r)]
            d['num_fused_aromatic'] = float(sum(
                1 for i, r1 in enumerate(ar)
                for r2 in ar[i+1:] if len(r1 & r2) >= 2))

            # ── Polarity and surface ───────────────────────────────────────────
            d['tpsa']       = Descriptors.TPSA(mol)
            d['labute_asa'] = rdMD.CalcLabuteASA(mol)
            d['polar_surface_ratio'] = (d['tpsa'] / d['labute_asa']
                                        if d['labute_asa'] > 0 else 0.0)
            try:
                d['MaxPartialCharge'] = Descriptors.MaxPartialCharge(mol)
                d['MinPartialCharge'] = Descriptors.MinPartialCharge(mol)
            except Exception:
                d['MaxPartialCharge'] = d['MinPartialCharge'] = 0.0
            d['charge'] = float(Chem.GetFormalCharge(mol))

            # ── Electronic ────────────────────────────────────────────────────
            d['MaxEStateIndex']    = Descriptors.MaxEStateIndex(mol)
            d['MinEStateIndex']    = Descriptors.MinEStateIndex(mol)
            d['MaxAbsEStateIndex'] = Descriptors.MaxAbsEStateIndex(mol)

            # ── SMR VSA ───────────────────────────────────────────────────────
            sm = MolSurf.SMR_VSA_(mol)
            d['SMR_VSA5'] = sm[4] if len(sm) > 4 else 0.0
            d['SMR_VSA7'] = sm[6] if len(sm) > 6 else 0.0

            # ── PXR-specific ──────────────────────────────────────────────────
            d['num_ring_hydroxyl'] = float(self._match(mol, 'ring_hydroxyl'))
            d['num_carboxyl']      = float(self._match(mol, 'carboxyl'))
            d['bile_acid_score']   = (d['logp']
                                      + d['num_ring_hydroxyl'] * 0.5
                                      + d['num_carboxyl'])

            # ── Reactivity ────────────────────────────────────────────────────
            d['num_michael_acceptors'] = float(self._match(mol, 'michael_acceptor'))
            d['num_electrophilic_sp2'] = float(self._match(mol, 'electrophilic_sp2'))

            # ── Drug-likeness ─────────────────────────────────────────────────
            try:
                d['qed'] = Descriptors.qed(mol)
            except Exception:
                d['qed'] = 0.0

        except Exception:
            pass
        return d

    def fit_transform(self, smiles_list):
        return self.scaler.fit_transform(self._matrix(smiles_list))

    def transform(self, smiles_list):
        return self.scaler.transform(self._matrix(smiles_list))

    def _matrix(self, smiles_list):
        rows = []
        for smi in smiles_list:
            mol  = Chem.MolFromSmiles(smi)
            desc = self._calc(mol)
            rows.append([float(desc.get(n, 0.0)) for n in self.descriptor_list])
        return np.nan_to_num(np.array(rows, dtype=np.float32), nan=0.0)


# =============================================================================
# SPLITTING STRATEGIES
# =============================================================================

def split_scaffold(smiles_list, n_folds=N_FOLDS, seed=GLOBAL_SEED):
    scaffolds = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                sc = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False)
            except Exception:
                sc = smi
            scaffolds[sc].append(idx)
    groups = sorted([scaffolds[k] for k in sorted(scaffolds)],
                    key=lambda x: (-len(x), min(x)))
    folds = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for g in groups:
        i = int(np.argmin(sizes))
        folds[i].extend(g)
        sizes[i] += len(g)
    return [sorted(f) for f in folds]


def split_butina(smiles_list, n_folds=N_FOLDS, threshold=BUTINA_THRESHOLD,
                 seed=GLOBAL_SEED):
    print(f"  Computing Butina clusters (threshold={threshold})...")
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(gen.GetFingerprint(mol) if mol else None)
    n = len(fps)
    dists = []
    for i in range(1, n):
        fp_i = fps[i]
        if fp_i is None:
            dists.extend([1.0] * i)
            continue
        for j in range(i):
            fp_j = fps[j]
            dists.append(1.0 - DataStructs.TanimotoSimilarity(fp_i, fp_j)
                         if fp_j else 1.0)
    from rdkit.ML.Cluster import Butina
    clusters = Butina.ClusterData(dists, n, threshold, isDistData=True)
    print(f"  Butina: {len(clusters)} clusters from {n} compounds")
    groups = sorted([list(c) for c in clusters], key=lambda x: (-len(x), min(x)))
    folds  = [[] for _ in range(n_folds)]
    sizes  = [0] * n_folds
    for g in groups:
        i = int(np.argmin(sizes))
        folds[i].extend(g)
        sizes[i] += len(g)
    return [sorted(f) for f in folds]


def split_descriptor(smiles_list, desc_matrix, n_folds=N_FOLDS,
                     n_clusters=KMEANS_N_CLUSTERS, seed=GLOBAL_SEED):
    print(f"  Computing k-means clusters (k={n_clusters}) on descriptors...")
    X  = StandardScaler().fit_transform(desc_matrix)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    clusters = defaultdict(list)
    for idx, lbl in enumerate(labels):
        clusters[lbl].append(idx)
    groups = sorted(clusters.values(), key=lambda x: (-len(x), min(x)))
    folds  = [[] for _ in range(n_folds)]
    sizes  = [0] * n_folds
    for g in groups:
        i = int(np.argmin(sizes))
        folds[i].extend(g)
        sizes[i] += len(g)
    return [sorted(f) for f in folds]


def split_activity_scaffold(smiles_list, y_values, n_folds=N_FOLDS,
                             seed=GLOBAL_SEED):
    """Activity-stratified scaffold split [DEFAULT]."""
    scaffold_to_idx = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            try:
                sc = MurckoScaffold.MurckoScaffoldSmiles(
                    mol=mol, includeChirality=False)
            except Exception:
                sc = smi
            scaffold_to_idx[sc].append(idx)

    quantiles = np.quantile(y_values, np.linspace(0, 1, n_folds + 1))
    def get_bin(y):
        for i, q in enumerate(quantiles[1:]):
            if y <= q:
                return i
        return n_folds - 1

    folds  = [[] for _ in range(n_folds)]
    sizes  = [0] * n_folds
    groups = sorted(scaffold_to_idx.values(), key=lambda x: (-len(x), min(x)))
    for g in groups:
        if len(g) == 1:
            i = int(np.argmin(sizes))
            folds[i].extend(g)
            sizes[i] += 1
        else:
            g_sorted   = sorted(g, key=lambda idx: get_bin(y_values[idx]))
            fold_order = np.argsort(sizes)
            for k, idx in enumerate(g_sorted):
                fi = int(fold_order[k % n_folds])
                folds[fi].append(idx)
                sizes[fi] += 1
    return [sorted(f) for f in folds]


def get_folds(strategy, smiles_list, y_values=None, desc_matrix=None,
              n_folds=N_FOLDS, seed=GLOBAL_SEED):
    strategy = strategy.lower()
    if strategy == 'scaffold':
        print(f"\n  Split strategy: Murcko scaffold ({n_folds} folds)")
        folds = split_scaffold(smiles_list, n_folds, seed)
    elif strategy == 'butina':
        print(f"\n  Split strategy: Butina cluster ({n_folds} folds)")
        folds = split_butina(smiles_list, n_folds, seed=seed)
    elif strategy == 'descriptor':
        print(f"\n  Split strategy: descriptor k-means ({n_folds} folds)")
        if desc_matrix is None:
            raise ValueError("desc_matrix required for descriptor split")
        folds = split_descriptor(smiles_list, desc_matrix, n_folds, seed=seed)
    elif strategy in ('activity_scaffold', 'activity'):
        print(f"\n  Split strategy: activity-stratified scaffold ({n_folds} folds)")
        if y_values is None:
            raise ValueError("y_values required for activity_scaffold split")
        folds = split_activity_scaffold(smiles_list, y_values, n_folds, seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")
    sizes = [len(f) for f in folds]
    print(f"  Fold sizes: min={min(sizes)}  max={max(sizes)}  "
          f"mean={np.mean(sizes):.0f}")
    return folds


# =============================================================================
# UTILITIES
# =============================================================================

def get_adaptive_config(n):
    for k in ('small', 'medium', 'large', 'very_large'):
        c = ADAPTIVE_CONFIG[k]
        if n < c['threshold']:
            out = c.copy(); out['n_samples'] = n; return out
    out = ADAPTIVE_CONFIG['very_large'].copy(); out['n_samples'] = n; return out


def remove_outliers(y, threshold=OUTLIER_MAD_THRESHOLD):
    if threshold is None:
        return np.ones(len(y), dtype=bool)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad == 0:
        return np.ones(len(y), dtype=bool)
    return np.abs(0.6745 * (y - med) / mad) < threshold


def compute_cliff_weights(smiles_list, y_values,
                           tanimoto_threshold=CLIFF_TANIMOTO_THRESHOLD,
                           delta_threshold=CLIFF_DELTA_THRESHOLD,
                           multiplier=CLIFF_WEIGHT_MULTIPLIER):
    """Upweight compounds involved in activity cliffs (similar structure, large DeltapEC50)."""
    from rdkit.Chem import AllChem, DataStructs
    n, fps = len(smiles_list), []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048) if mol else None)
    is_cliff, n_pairs = np.zeros(n, dtype=bool), 0
    for i in range(n):
        if fps[i] is None:
            continue
        valid_j = [j for j in range(i + 1, n) if fps[j] is not None]
        bulk = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in valid_j])
        for j, sim in zip(valid_j, bulk):
            if sim >= tanimoto_threshold and \
               abs(float(y_values[i]) - float(y_values[j])) >= delta_threshold:
                is_cliff[i] = is_cliff[j] = True
                n_pairs += 1
    nc = int(is_cliff.sum())
    print(f"  Cliffs: {n_pairs} pairs, {nc} compounds ({nc/n*100:.1f}%) "
          f"upweighted {multiplier}x")
    return np.where(is_cliff, multiplier, 1.0).astype(np.float64)


def compute_sample_weights(df, min_w=0.1, max_w=5.0):
    """
    Combined sample weighting:
      1. Selectivity weight — rewards confirmed PXR-selective compounds
      2. CI width weight    — rewards high-precision measurements
      3. Tail upweighting   — capped at TAIL_WEIGHT_MAX (5x, matching v4_1)
      4. Cliff upweighting  — capped at CLIFF_WEIGHT_MULTIPLIER (2x default)

    Pass sample_weights=None to train() to skip weighting entirely.
    """
    n      = len(df)
    sel_w  = np.ones(n, dtype=np.float64)
    ci_w   = np.ones(n, dtype=np.float64)
    tail_w = np.ones(n, dtype=np.float64)

    if SELECTIVITY_COL in df.columns:
        has_sel = df[SELECTIVITY_COL].notna()
        if STD_ERROR_COL in df.columns:
            sel_w[has_sel.values] = (
                df.loc[has_sel, SELECTIVITY_COL].clip(lower=0).values /
                df.loc[has_sel, STD_ERROR_COL].clip(lower=0.05).values)
        else:
            sel_w[has_sel.values] = (
                df.loc[has_sel, SELECTIVITY_COL].clip(lower=0).values)

    if CI_LOWER_COL in df.columns and CI_UPPER_COL in df.columns:
        has_ci = df[CI_LOWER_COL].notna() & df[CI_UPPER_COL].notna()
        if has_ci.sum() > 10:
            ci_width = (df.loc[has_ci, CI_UPPER_COL] -
                        df.loc[has_ci, CI_LOWER_COL]).clip(lower=0.05).values
            ci_w[has_ci.values] = np.median(ci_width) / ci_width

    if TARGET_COL in df.columns:
        y_vals = df[TARGET_COL].values.astype(np.float64)
        y_mean = np.nanmean(y_vals)
        y_std  = np.nanstd(y_vals) + 1e-8
        tail_w = np.clip(
            1.0 + np.abs(y_vals - y_mean) / y_std,
            1.0, TAIL_WEIGHT_MAX)          # 5x cap (restored from v4_1)
        n_tail = (np.abs(y_vals - y_mean) / y_std > 1).sum()
        print(f"  Tail upweighting: {n_tail} compounds with |z|>1  "
              f"weight=[{tail_w.min():.2f}, {tail_w.max():.2f}]  "
              f"(cap={TAIL_WEIGHT_MAX}x)")

    cliff_w = np.ones(n, dtype=np.float64)
    if CLIFF_WEIGHT_MULTIPLIER > 1.0 and SMILES_COL in df.columns and TARGET_COL in df.columns:
        try:
            cliff_w = compute_cliff_weights(df[SMILES_COL].tolist(), df[TARGET_COL].values)
        except Exception as e:
            print(f"  WARNING: cliff weights failed: {e}")

    combined = np.clip(sel_w * ci_w * tail_w * cliff_w, min_w, max_w)
    combined = combined / combined.mean()
    return combined


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_version_checkpoints(model_dir, version='latest'):
    ckpts = glob.glob(os.path.join(model_dir, 'fold_*.ckpt'))
    if not ckpts:
        return []
    fv = {}
    for p in ckpts:
        m = re.match(r'fold_(\d+)(?:_\w+)?(?:-v(\d+))?\.ckpt',
                     os.path.basename(p))
        if m:
            fn = m.group(1)
            v  = int(m.group(2)) if m.group(2) else 0
            fv.setdefault(fn, {})[v] = p
    out = []
    for fn in sorted(fv, key=lambda x: int(x) if x.isdigit() else 0):
        vs = fv[fn]; chosen = max(vs)
        out.append(vs[chosen])
        print(f"    Fold {fn}: {'v'+str(chosen) if chosen else 'v0'}")
    return out


# =============================================================================
# MPNN PREDICTOR
# =============================================================================


# =============================================================================
# MULTI-TASK MPNN  (regression + binary classification)
# =============================================================================


class MultitaskMPNN(pl.LightningModule):
    """
    Chemprop MPNN with two output heads trained jointly:
      - Regression head : scaled pEC50  (MSE loss, primary)
      - Classification  : binary active (pEC50 >= act_thresh, BCE loss)
    Total loss = (1 - clf_weight) * MSE_reg + clf_weight * BCE_clf
    At inference, only the regression head is used.
    """

    def __init__(self, mp, agg, hidden_dim, n_layers, dropout, input_extra,
                 clf_weight=0.2, act_thresh_scaled=0.0, lr=1e-4):
        super().__init__()
        self.mp          = mp
        self.agg         = agg
        self.clf_weight  = clf_weight
        self.act_thresh  = act_thresh_scaled   # threshold in SCALED space
        self._lr         = lr

        in_dim = mp.output_dim + input_extra   # e.g. 768 + 61 = 829
        # Shared trunk: (n_layers - 1) hidden layers
        layers, prev = [], in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(prev, hidden_dim),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self._trunk_out = hidden_dim if n_layers > 1 else in_dim

        self.reg_head = nn.Linear(self._trunk_out, 1)   # regression
        self.clf_head = nn.Linear(self._trunk_out, 1)   # binary classification

    def _encode(self, bmg, V_d, X_d):
        H = self.mp(bmg, V_d)
        H = self.agg(H, bmg.batch)
        if X_d is not None:
            H = torch.cat([H, X_d], dim=-1)
        return H

    def forward(self, bmg, V_d=None, X_d=None):
        """Regression-only forward for chemprop trainer.predict()."""
        H = self._encode(bmg, V_d, X_d)
        return self.reg_head(self.trunk(H))

    def training_step(self, batch, batch_idx):
        bmg, V_d, X_d, targets, weights, *_ = batch
        H      = self._encode(bmg, V_d, X_d)
        shared = self.trunk(H)

        reg_pred  = self.reg_head(shared).squeeze(-1)
        clf_logit = self.clf_head(shared).squeeze(-1)

        y_sc = targets.squeeze(-1)
        w    = (weights.squeeze(-1)
                if weights is not None else torch.ones_like(y_sc))

        # Regression loss: weighted MSE (all compounds)
        reg_loss = (w * (reg_pred - y_sc) ** 2).mean()

        # Asymmetric classification loss -- FALSE POSITIVE PENALTY ONLY.
        # Only inactive compounds (pEC50 < act_thresh) contribute to clf_loss.
        # Active compounds generate zero classification gradient, so genuine
        # actives are never pulled toward the inactive zone.
        inactive_mask = (y_sc <= self.act_thresh)
        if inactive_mask.any():
            # Target = 0 (inactive) for all compounds in this subset.
            # Loss fires when clf_logit > 0 (model thinks "active") for an
            # inactive compound -- i.e., a false positive.
            fp_logits  = clf_logit[inactive_mask]
            fp_targets = torch.zeros_like(fp_logits)
            fp_weights = w[inactive_mask]
            clf_loss = F.binary_cross_entropy_with_logits(
                fp_logits, fp_targets, weight=fp_weights)
        else:
            clf_loss = torch.tensor(0.0, device=y_sc.device,
                                    requires_grad=False)

        loss = (1.0 - self.clf_weight) * reg_loss + self.clf_weight * clf_loss
        self.log_dict(
            {'train_loss': loss, 'reg_loss': reg_loss, 'clf_loss': clf_loss},
            prog_bar=False, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        bmg, V_d, X_d, targets, weights, *_ = batch
        H        = self._encode(bmg, V_d, X_d)
        reg_pred = self.reg_head(self.trunk(H)).squeeze(-1)
        y_sc     = targets.squeeze(-1)
        val_loss = F.mse_loss(reg_pred, y_sc)   # regression MSE only
        self.log('val_loss', val_loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        return val_loss

    def predict_step(self, batch, batch_idx):
        """Called by trainer.predict() -- returns regression output only."""
        bmg, V_d, X_d, *_ = batch
        H = self._encode(bmg, V_d, X_d)
        return self.reg_head(self.trunk(H))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._lr)


class PXRMPNNPredictor:
    """
    Single-task MPNN for PXR pEC50.

    v4_3_3 vs v4_3_1:
      - MSE loss (same as v4_3_1)
      - Tail cap 5x (same as v4_3_1)
      - Sample weight BUG FIXED: weights now passed to MoleculeDatapoint(weight=)
      - --no_sample_weights flag lets you disable weighting to A/B test
    """

    def __init__(self, model_dir=MODEL_SAVE_DIR, checkpoint_version='latest'):
        self.model_dir           = model_dir
        self.checkpoint_version  = checkpoint_version
        os.makedirs(self.model_dir, exist_ok=True)
        self.mpnn_models:     list                 = []
        self.desc_calc:       Optional[object]     = None
        self.scaler_y                              = RobustScaler(
            quantile_range=(1.0, 99.0))
        self.augmenter                             = SMILESAugmenter(
            random_seed=GLOBAL_SEED)
        self.adaptive_config: Optional[dict]       = None
        self.oof_predictions: Optional[np.ndarray] = None
        self.oof_targets:     Optional[np.ndarray] = None
        self.fold_maes:       List[float]           = []
        self.best_epochs:     List[int]             = []
        self.smiles_clean:    Optional[List[str]]   = None

    def _build(self, cfg, input_extra=0):
        hd, d, nl, do = (cfg['hidden_dim'], cfg['depth'],
                         cfg['n_layers'],   cfg['dropout'])
        mp  = BondMessagePassing(d_h=hd, depth=d, dropout=do, activation='relu')
        agg = MeanAggregation()
        return MultitaskMPNN(
            mp               = mp,
            agg              = agg,
            hidden_dim       = hd,
            n_layers         = nl,
            dropout          = do,
            input_extra      = input_extra,
            clf_weight       = getattr(self, '_clf_weight',        0.2),
            act_thresh_scaled= getattr(self, '_act_thresh_scaled', 0.0),
            lr               = 1e-4,
        )

    def _save_model_config(self, cfg, input_extra):
        """Persist architecture params so _load_from_ckpt can rebuild the model."""
        info = {
            'hidden_dim':         cfg['hidden_dim'],
            'depth':              cfg['depth'],
            'n_layers':           cfg['n_layers'],
            'dropout':            cfg['dropout'],
            'input_extra':        input_extra,
            'clf_weight':         getattr(self, '_clf_weight',        0.2),
            'act_thresh_scaled':  getattr(self, '_act_thresh_scaled', 0.0),
        }
        with open(os.path.join(self.model_dir, 'model_config.json'), 'w') as fh:
            json.dump(info, fh, indent=2)

    def _load_from_ckpt(self, ckpt_path):
        """Reconstruct MultitaskMPNN from saved config + checkpoint weights."""
        cfg_path = os.path.join(self.model_dir, 'model_config.json')
        with open(cfg_path) as fh:
            mcfg = json.load(fh)
        mp  = BondMessagePassing(d_h=mcfg['hidden_dim'],
                                  depth=mcfg['depth'])
        agg = MeanAggregation()
        model = MultitaskMPNN(
            mp               = mp,
            agg              = agg,
            hidden_dim       = mcfg['hidden_dim'],
            n_layers         = mcfg['n_layers'],
            dropout          = mcfg['dropout'],
            input_extra      = mcfg['input_extra'],
            clf_weight       = mcfg['clf_weight'],
            act_thresh_scaled= mcfg['act_thresh_scaled'],
            lr               = 1e-4,
        )
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE,
                              weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt['state_dict'])
        return model


    def train(self, smiles_list, y_primary,
              descriptor_list=PXR_DESCRIPTOR_LIST,
              sample_weights=None,
              split_strategy='activity_scaffold',
              fold_tag=''):
        use_weights = sample_weights is not None
        print(f"\n{'='*60}")
        print(f"Training PXR MPNN v4_13d |  N={len(smiles_list)}  "
              f"folds={N_FOLDS}  strategy={split_strategy}")
        print(f"Descriptors: {len(descriptor_list)}  "
              f"Loss: MSE  TailCap: {TAIL_WEIGHT_MAX}x  "
              f"SampleWeights: {'ON (bug-fixed)' if use_weights else 'OFF'}")
        print(f"{'='*60}")

        self._seed = GLOBAL_SEED
        self.adaptive_config = cfg = get_adaptive_config(len(smiles_list))
        print(f"Config: hidden={cfg['hidden_dim']}  depth={cfg['depth']}  "
              f"layers={cfg['n_layers']}  dropout={cfg['dropout']}")

        if not fold_tag:
            stale = glob.glob(os.path.join(self.model_dir, 'fold_*.ckpt'))
            if stale:
                for s in stale:
                    os.remove(s)
                print(f"  Purged {len(stale)} stale checkpoint(s)")

        mask         = remove_outliers(y_primary)
        smiles_clean = [s for s, m in zip(smiles_list, mask) if m]
        y_clean      = y_primary[mask]
        w_clean      = sample_weights[mask] if use_weights else None
        self.smiles_clean = smiles_clean

        print(f"pEC50: [{y_clean.min():.3f}, {y_clean.max():.3f}]  "
              f"std={y_clean.std():.3f}  N={len(y_clean)}")

        y_sc = self.scaler_y.fit_transform(y_clean.reshape(-1, 1)).ravel()
        print(f"Scaled y: [{y_sc.min():.3f}, {y_sc.max():.3f}]")

        # Compute scaled activity threshold for classification head
        _act_thresh_raw = getattr(self, '_act_thresh', 4.0)
        self._act_thresh_scaled = float(
            self.scaler_y.transform([[_act_thresh_raw]])[0, 0])
        print(f"Act threshold: {_act_thresh_raw:.1f} pEC50 "
              f"-> {self._act_thresh_scaled:.4f} (scaled)")

        print(f"Computing {len(descriptor_list)} descriptors ...")
        self.desc_calc = PXRDescriptorCalculator(descriptor_list, 'robust')
        X_desc_full    = self.desc_calc.fit_transform(smiles_clean)
        print(f"Descriptor matrix: {X_desc_full.shape}")

        # Save architecture config once (same for all folds)
        self._save_model_config(
            get_adaptive_config(len(smiles_clean)), len(descriptor_list))

        folds = get_folds(
            strategy    = split_strategy,
            smiles_list = smiles_clean,
            y_values    = y_clean,
            desc_matrix = X_desc_full,
            n_folds     = N_FOLDS,
            seed        = GLOBAL_SEED,
        )

        oof_preds    = np.zeros(len(smiles_clean))
        fold_metrics = []
        self.fold_maes = []

        for fold_idx, val_indices in enumerate(folds):
            set_global_seed(self._seed + fold_idx)
            print(f"\n--- Fold {fold_idx+1}/{N_FOLDS}  "
                  f"(val={len(val_indices)}) ---")

            tr_mask = np.ones(len(smiles_clean), dtype=bool)
            tr_mask[val_indices] = False

            tr_smi  = [smiles_clean[i] for i in np.where(tr_mask)[0]]
            tr_y    = y_sc[tr_mask]
            tr_w    = w_clean[tr_mask] if w_clean is not None else None
            vl_smi  = [smiles_clean[i] for i in val_indices]
            vl_y    = y_sc[val_indices]
            vl_desc = X_desc_full[val_indices]

            if cfg['use_augmentation'] and cfg['augmentation_factor'] > 1:
                aug_smi, aug_y, aug_w = self.augmenter.augment_dataset(
                    tr_smi, tr_y, cfg['augmentation_factor'], tr_w)
            else:
                aug_smi, aug_y, aug_w = tr_smi, tr_y, tr_w

            aug_desc = self.desc_calc.transform(aug_smi)

            # Sample weights correctly forwarded to MoleculeDatapoint (bug fix)
            train_dps = [
                MoleculeDatapoint(
                    mol=Chem.MolFromSmiles(s),
                    y=[float(aug_y[i])],
                    x_d=aug_desc[i].astype(np.float32),
                    weight=float(aug_w[i]) if aug_w is not None else 1.0,
                )
                for i, s in enumerate(aug_smi) if Chem.MolFromSmiles(s)
            ]
            val_dps = [
                MoleculeDatapoint(
                    mol=Chem.MolFromSmiles(s),
                    y=[float(vl_y[i])],
                    x_d=vl_desc[i].astype(np.float32),
                )
                for i, s in enumerate(vl_smi) if Chem.MolFromSmiles(s)
            ]

            tr_loader = build_dataloader(MoleculeDataset(train_dps),
                                          batch_size=cfg['batch_size'],
                                          shuffle=True, num_workers=0)
            vl_loader = build_dataloader(MoleculeDataset(val_dps),
                                          batch_size=cfg['batch_size'],
                                          shuffle=False, num_workers=0)

            model   = self._build(cfg, input_extra=len(descriptor_list))
            ckpt_fn = f'fold_{fold_idx}{fold_tag}'
            ckpt_cb = ModelCheckpoint(
                dirpath=self.model_dir, filename=ckpt_fn,
                monitor='val_loss', mode='min', save_top_k=1)
            trainer = pl.Trainer(
                accelerator='gpu' if DEVICE == 'cuda' else 'cpu',
                devices=1,
                max_epochs=cfg['max_epochs'],
                callbacks=[
                    EarlyStopping(monitor='val_loss',
                                  patience=cfg['patience'], mode='min'),
                    ckpt_cb,
                ],
                enable_progress_bar=True,
                logger=False,
                gradient_clip_val=1.0,
                deterministic=True,
            )
            trainer.fit(model, tr_loader, vl_loader)

            # Record best epoch (early stopping stops `patience` epochs after best)
            best_ep = max(0, trainer.current_epoch - cfg['patience'])
            self.best_epochs.append(best_ep)
            print(f"  Stopped epoch={trainer.current_epoch}  "
                  f"Best epoch≈{best_ep}")

            if not ckpt_cb.best_model_path or \
               not os.path.exists(ckpt_cb.best_model_path):
                print(f"  WARNING: no checkpoint for fold {fold_idx}")
                continue
            best = self._load_from_ckpt(ckpt_cb.best_model_path)
            if best is None:
                print(f"  WARNING: load failed for fold {fold_idx}")
                continue
            best.eval()
            best.to(torch.device(DEVICE))

            pred_trainer = pl.Trainer(
                accelerator='gpu' if DEVICE == 'cuda' else 'cpu',
                devices=1, logger=False, enable_progress_bar=False)
            with torch.inference_mode():
                raw = pred_trainer.predict(best, vl_loader)
            raw_np = np.concatenate([p.cpu().numpy() for p in raw]).flatten()

            vl_y_orig    = self.scaler_y.inverse_transform(
                vl_y.reshape(-1, 1)).ravel()
            vl_pred_orig = self.scaler_y.inverse_transform(
                raw_np.reshape(-1, 1)).ravel()

            rmse = float(np.sqrt(np.mean((vl_y_orig - vl_pred_orig)**2)))
            mae  = float(np.mean(np.abs(vl_y_orig - vl_pred_orig)))
            r2   = float(1 - np.sum((vl_y_orig - vl_pred_orig)**2)
                         / (np.sum((vl_y_orig - vl_y_orig.mean())**2) + 1e-12))
            print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")
            fold_metrics.append({'fold': fold_idx, 'rmse': rmse,
                                  'mae': mae, 'r2': r2})
            self.fold_maes.append(mae)

            valid_val_idx = [val_indices[i] for i, s in enumerate(vl_smi)
                             if Chem.MolFromSmiles(s)]
            for i, idx in enumerate(valid_val_idx):
                if i < len(raw_np):
                    oof_preds[idx] = raw_np[i]

        self.oof_predictions = self.scaler_y.inverse_transform(
            oof_preds.reshape(-1, 1)).ravel()
        self.oof_targets = y_clean

        oof_min  = self.oof_predictions.min()
        oof_max  = self.oof_predictions.max()
        tr_min   = float(y_clean.min())
        tr_max   = float(y_clean.max())
        coverage = (oof_max - oof_min) / (tr_max - tr_min + 1e-8)
        print(f"\nOOF range: [{oof_min:.3f}, {oof_max:.3f}]  "
              f"(training: [{tr_min:.3f}, {tr_max:.3f}])  "
              f"coverage={coverage:.1%}")
        if coverage < 0.75:
            print(f"  *** WARNING: OOF coverage {coverage:.1%} < 75%")

        save_pickle(self.scaler_y,
                    os.path.join(self.model_dir, 'scaler_y.pkl'))
        save_pickle(descriptor_list,
                    os.path.join(self.model_dir, 'desc_list.pkl'))
        save_pickle(self.desc_calc.scaler,
                    os.path.join(self.model_dir, 'desc_scaler.pkl'))
        save_pickle(self.fold_maes,
                    os.path.join(self.model_dir, 'fold_maes.pkl'))
        save_pickle(self.best_epochs,
                    os.path.join(self.model_dir, 'best_epochs.pkl'))
        avg_ep = int(np.mean(self.best_epochs)) if self.best_epochs else 0
        print(f"\nBest epochs per fold : {self.best_epochs}")
        print(f"Average best epoch   : {avg_ep}  "
              f"(use this for --mode FULL --full_epochs {avg_ep + 5})")
        save_pickle(self.oof_predictions,
                    os.path.join(self.model_dir, 'oof_predictions.pkl'))
        save_pickle(self.oof_targets,
                    os.path.join(self.model_dir, 'oof_targets.pkl'))
        save_pickle({'n_folds': N_FOLDS, 'split_strategy': split_strategy,
                     'n_descriptors': len(descriptor_list),
                     'tail_weight_max': TAIL_WEIGHT_MAX,
                     'sample_weights': use_weights,
                     'training_style': 'pxr_v4_3_3'},
                    os.path.join(self.model_dir, 'meta.pkl'))

        oof_y    = self.oof_targets
        oof_p    = self.oof_predictions
        oof_rmse = float(np.sqrt(np.mean((oof_y - oof_p)**2)))
        oof_mae  = float(np.mean(np.abs(oof_y - oof_p)))
        oof_r2   = float(1 - np.sum((oof_y - oof_p)**2)
                         / (np.sum((oof_y - oof_y.mean())**2) + 1e-12))

        print(f"\n{'='*60}")
        print(f"OOF Results ({split_strategy})  [SampleWeights={'ON' if use_weights else 'OFF'}]:")
        print(f"  RMSE : {oof_rmse:.4f}")
        print(f"  MAE  : {oof_mae:.4f}")
        print(f"  R2   : {oof_r2:.4f}")
        print(f"{'='*60}")
        for fm in fold_metrics:
            print(f"  Fold {fm['fold']+1}: "
                  f"RMSE={fm['rmse']:.4f}  "
                  f"MAE={fm['mae']:.4f}  "
                  f"R2={fm['r2']:.4f}")

        return {'oof_rmse': oof_rmse, 'oof_mae': oof_mae, 'oof_r2': oof_r2,
                'fold_metrics': fold_metrics}

    def load_models(self, checkpoint_version=None):
        version = checkpoint_version or self.checkpoint_version
        try:
            self.scaler_y  = load_pickle(
                os.path.join(self.model_dir, 'scaler_y.pkl'))
            desc_list      = load_pickle(
                os.path.join(self.model_dir, 'desc_list.pkl'))
            self.desc_calc = PXRDescriptorCalculator(desc_list, 'robust')
            self.desc_calc.scaler = load_pickle(
                os.path.join(self.model_dir, 'desc_scaler.pkl'))
            self.fold_maes = load_pickle(
                os.path.join(self.model_dir, 'fold_maes.pkl'))
            ckpts = get_version_checkpoints(self.model_dir, version)
            if not ckpts:
                print("  No checkpoints found.")
                return False
            self.mpnn_models = []
            for c in ckpts:
                m = self._load_from_ckpt(c)
                if m:
                    m.eval()
                    m.to(torch.device(DEVICE))
                    self.mpnn_models.append(m)
            print(f"  Loaded {len(self.mpnn_models)} model(s)")
            return len(self.mpnn_models) > 0
        except Exception as e:
            print(f"  Error loading models: {e}")
            return False

    def train_full(self, smiles_list, y_primary,
                   descriptor_list=PXR_DESCRIPTOR_LIST,
                   sample_weights=None,
                   fixed_epochs=None):
        """
        Train ONE model on ALL training data (no train/val split).
        fixed_epochs: average best epoch from CV + small buffer.
        No early stopping — every compound contributes to learning.
        """
        use_weights = sample_weights is not None

        if fixed_epochs is None:
            ep_path = os.path.join(self.model_dir, 'best_epochs.pkl')
            if os.path.exists(ep_path):
                saved = load_pickle(ep_path)
                fixed_epochs = int(np.mean(saved)) + 5
                print(f"  Loaded best epochs from CV: {saved}")
                print(f"  Fixed epoch budget: {fixed_epochs} "
                      f"(mean={int(np.mean(saved))} + 5 buffer)")
            else:
                fixed_epochs = 100
                print(f"  No saved CV epochs found — using default {fixed_epochs}")

        print(f"\n{'='*60}")
        print(f"FULL RETRAIN  |  N={len(smiles_list)}  epochs={fixed_epochs}")
        print(f"SampleWeights: {'ON' if use_weights else 'OFF'}")
        print(f"{'='*60}")

        self._seed = GLOBAL_SEED
        set_global_seed(self._seed)
        self.adaptive_config = cfg = get_adaptive_config(len(smiles_list))
        print(f"Config: hidden={cfg['hidden_dim']}  depth={cfg['depth']}  "
              f"layers={cfg['n_layers']}  dropout={cfg['dropout']}")

        mask         = remove_outliers(y_primary)
        smiles_clean = [s for s, m in zip(smiles_list, mask) if m]
        y_clean      = y_primary[mask]
        w_clean      = sample_weights[mask] if use_weights else None

        print(f"pEC50: [{y_clean.min():.3f}, {y_clean.max():.3f}]  "
              f"std={y_clean.std():.3f}  N={len(y_clean)}")

        y_sc = self.scaler_y.fit_transform(y_clean.reshape(-1, 1)).ravel()

        # Compute scaled activity threshold for classification head
        _act_thresh_raw = getattr(self, '_act_thresh', 4.0)
        self._act_thresh_scaled = float(
            self.scaler_y.transform([[_act_thresh_raw]])[0, 0])
        print(f"Act threshold: {_act_thresh_raw:.1f} pEC50 "
              f"-> {self._act_thresh_scaled:.4f} (scaled)")

        print(f"Computing {len(descriptor_list)} descriptors ...")
        self.desc_calc = PXRDescriptorCalculator(descriptor_list, 'robust')
        X_desc_full    = self.desc_calc.fit_transform(smiles_clean)

        if cfg['use_augmentation'] and cfg['augmentation_factor'] > 1:
            aug_smi, aug_y, aug_w = self.augmenter.augment_dataset(
                smiles_clean, y_sc, cfg['augmentation_factor'], w_clean)
        else:
            aug_smi, aug_y, aug_w = smiles_clean, y_sc, w_clean

        aug_desc  = self.desc_calc.transform(aug_smi)
        train_dps = [
            MoleculeDatapoint(
                mol=Chem.MolFromSmiles(s),
                y=[float(aug_y[i])],
                x_d=aug_desc[i].astype(np.float32),
                weight=float(aug_w[i]) if aug_w is not None else 1.0,
            )
            for i, s in enumerate(aug_smi) if Chem.MolFromSmiles(s)
        ]
        tr_loader = build_dataloader(MoleculeDataset(train_dps),
                                     batch_size=cfg['batch_size'],
                                     shuffle=True, num_workers=0)

        model   = self._build(cfg, input_extra=len(descriptor_list))
        self._save_model_config(cfg, len(descriptor_list))
        trainer = pl.Trainer(
            accelerator='gpu' if DEVICE == 'cuda' else 'cpu',
            devices=1,
            max_epochs=fixed_epochs,
            enable_checkpointing=False,   # no val set to monitor
            logger=False,
            enable_progress_bar=True,
            gradient_clip_val=1.0,
            deterministic=True,
        )
        trainer.fit(model, tr_loader)    # no val_loader

        full_ckpt = os.path.join(self.model_dir, 'full_retrain.ckpt')
        trainer.save_checkpoint(full_ckpt)
        print(f"\nFull retrain model saved → {full_ckpt}")

        save_pickle(self.scaler_y,
                    os.path.join(self.model_dir, 'scaler_y.pkl'))
        save_pickle(descriptor_list,
                    os.path.join(self.model_dir, 'desc_list.pkl'))
        save_pickle(self.desc_calc.scaler,
                    os.path.join(self.model_dir, 'desc_scaler.pkl'))
        self.fold_maes = [1.0]          # single model — equal weight

        model.eval()
        model.to(torch.device(DEVICE))
        self.mpnn_models = [model]
        print(f"Full retrain complete — {fixed_epochs} epochs, "
              f"{len(smiles_clean)} compounds (all data)")
        return model

    def predict(self, smiles_list, use_weighted=False):
        """Ensemble prediction. Default: simple average (use_weighted=False).
        Set use_weighted=True for 1/MAE fold weighting (negligible difference
        in practice — simple average confirmed equal or marginally better)."""
        if not self.mpnn_models:
            return np.zeros(len(smiles_list))

        X_desc = self.desc_calc.transform(smiles_list)
        dps, valid_idx = [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                dps.append(MoleculeDatapoint(
                    mol=mol, x_d=X_desc[i].astype(np.float32)))
                valid_idx.append(i)

        if not dps:
            return np.zeros(len(smiles_list))

        loader = build_dataloader(MoleculeDataset(dps), batch_size=64,
                                  shuffle=False, num_workers=0)
        fold_preds = []
        for m in self.mpnn_models:
            m.eval()
            pt = pl.Trainer(
                accelerator='gpu' if DEVICE == 'cuda' else 'cpu',
                devices=1, logger=False, enable_progress_bar=False)
            with torch.inference_mode():
                bp = pt.predict(m, loader)
            raw  = np.concatenate([b.cpu().numpy() for b in bp]).flatten()
            orig = self.scaler_y.inverse_transform(raw.reshape(-1, 1)).ravel()
            full = np.zeros(len(smiles_list))
            for idx, val in zip(valid_idx, orig):
                full[idx] = val
            fold_preds.append(full)

        fold_preds = np.array(fold_preds)
        if use_weighted and self.fold_maes and \
           len(self.fold_maes) == len(fold_preds):
            w = np.array([1.0 / mae for mae in self.fold_maes])
            w = w / w.sum()
            return np.average(fold_preds, axis=0, weights=w)
        return fold_preds.mean(axis=0)


# =============================================================================
# MAIN
# =============================================================================


def save_training_log(log: dict, model_dir: str, output_csv: str) -> None:
    """Write training log to {model_dir}/training_log.json and alongside output CSV."""
    def _write(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Training log saved -> {path}")
    _write(os.path.join(model_dir, "training_log.json"))
    stem = os.path.splitext(output_csv)[0]
    _write(stem + "_training_log.json")


def main():
    parser = argparse.ArgumentParser(description="PXR pEC50 MPNN v4_13d")
    parser.add_argument('--mode',
                        choices=['TRAIN', 'INFERENCE', 'BOTH', 'FULL'],
                        default='BOTH',
                        help='FULL = retrain on 100%% data using avg CV epoch, '
                             'then predict test set')
    parser.add_argument('--train_csv',    default=TRAIN_CSV)
    parser.add_argument('--test_csv',     default=TEST_CSV)
    parser.add_argument('--model_dir',    default=MODEL_SAVE_DIR)
    parser.add_argument('--output',       default=OUTPUT_CSV)
    parser.add_argument('--ckpt_version', default='latest')
    parser.add_argument('--seed', type=int, default=GLOBAL_SEED)
    parser.add_argument('--split_strategy',
                        choices=['scaffold', 'butina', 'descriptor',
                                 'activity_scaffold'],
                        default='activity_scaffold')
    parser.add_argument('--full_epochs', type=int, default=None,
                        help='Fixed epoch count for --mode FULL. '
                             'If omitted, uses avg best epoch from CV + 5.')
    parser.add_argument('--cliff_weight', type=float, default=CLIFF_WEIGHT_MULTIPLIER,
                        help='Upweight multiplier for activity cliff compounds '
                             '(default: 2.0, set 1.0 to disable)')
    parser.add_argument('--no_sample_weights', action='store_true', default=True,
                        help='Disable selectivity/CI/tail sample weighting (default: OFF). '
                             'Best Phase 1 result used no weights.')
    parser.add_argument('--high_tail_mult', type=float, default=3.0,
                        help='Upweight multiplier applied to compounds with pEC50 >= '
                             '--high_tail_thresh (default: 1.0 = OFF). Applied on top of '
                             'any sample weights (or alone when --no_sample_weights). '
                             'Targets prediction ceiling without affecting inactives.')
    parser.add_argument('--high_tail_thresh', type=float, default=5.5,
                        help='pEC50 threshold for high-tail asymmetric upweighting '
                             '(default: 5.5). Covers near-ceiling as well as extreme actives.')
    parser.add_argument('--inactive_supplement_csv',
                        default=INACTIVE_SUPPLEMENT_CSV,
                        help='CSV of confirmed-inactive compounds to append to training '
                             '(default: sc_inactives_300.csv). Set to "" to disable. '
                             'Corrects inactive over-prediction without removing '
                             'natural low-activity compounds from the base dataset.')
    parser.add_argument('--clf_weight', type=float, default=0.2,
                        help='Weight of classification loss in joint training: '
                             'total = (1-clf_weight)*MSE + clf_weight*BCE. '
                             'Default 0.2 = 80%%%% regression, 20%%%% classification.')
    parser.add_argument('--act_thresh', type=float, default=4.0,
                        help='pEC50 threshold separating active/inactive for the '
                             'classification head (default 4.0).')
    args = parser.parse_args()

    import sys
    _seed = args.seed
    set_global_seed(_seed)
    sys.modules[__name__].GLOBAL_SEED = _seed
    sys.modules[__name__].CLIFF_WEIGHT_MULTIPLIER = args.cliff_weight
    print(f"Cliff upweighting : {args.cliff_weight}x  "
          f"(Tan>={CLIFF_TANIMOTO_THRESHOLD}, DpEC50>={CLIFF_DELTA_THRESHOLD})")
    print(f"Global seed       : {_seed}")
    print(f"Loss              : MSE (chemprop default)")
    print(f"Tail weight cap   : {TAIL_WEIGHT_MAX}x")
    print(f"Sample weights    : {'DISABLED (--no_sample_weights)' if args.no_sample_weights else 'ENABLED (bug-fixed)'}")
    print(f"High-tail upwt    : {args.high_tail_mult}x  (pEC50 >= {args.high_tail_thresh})"
          if args.high_tail_mult > 1.0 else "High-tail upwt    : OFF")

    t0  = time.time()
    std = MoleculeStandardizer()

    # ── TRAINING ──────────────────────────────────────────────────────────────
    if args.mode in ('TRAIN', 'BOTH'):
        print(f"\nLoading: {args.train_csv}")
        df = pd.read_csv(args.train_csv).reset_index(drop=True)
        df.columns = [c.strip() for c in df.columns]

        smiles_actual = next(
            (c for c in df.columns if c.lower() == 'smiles'), SMILES_COL)
        if smiles_actual != SMILES_COL:
            df = df.rename(columns={smiles_actual: SMILES_COL})

        if TARGET_COL not in df.columns:
            raise ValueError(f"Column '{TARGET_COL}' not found. "
                             f"Available: {list(df.columns)}")

        df['mol']          = df[SMILES_COL].apply(std.sanitize)
        df['SMILES_clean'] = df['mol'].apply(
            lambda m: Chem.MolToSmiles(m, canonical=True) if m else None)
        df = df.dropna(
            subset=['SMILES_clean', TARGET_COL]).reset_index(drop=True)
        print(f"Valid rows: {len(df)}  "
              f"pEC50=[{df[TARGET_COL].min():.2f},"
              f"{df[TARGET_COL].max():.2f}]  "
              f"std={df[TARGET_COL].std():.3f}")

        # ── Inactive supplement (SC confirmed inactives) ─────────────────
        if args.inactive_supplement_csv:
            try:
                sup = pd.read_csv(args.inactive_supplement_csv)
                sup.columns = [c.strip() for c in sup.columns]
                sup["mol"] = sup[SMILES_COL].apply(std.sanitize)
                sup["SMILES_clean"] = sup["mol"].apply(
                    lambda m: Chem.MolToSmiles(m, canonical=True) if m else None)
                sup = sup.dropna(subset=["SMILES_clean", TARGET_COL]).reset_index(drop=True)
                df = pd.concat([df, sup], ignore_index=True)
                n_inact = (df[TARGET_COL] < 3.5).sum()
                print(f"Inactive supplement: +{len(sup)} rows from "
                      f"{args.inactive_supplement_csv}  "
                      f"total_inact<3.5={n_inact}  total={len(df)}")
            except Exception as e:
                print(f"WARNING: could not load inactive supplement: {e}")

        if args.no_sample_weights:
            sw = None
            print("Sample weights: DISABLED")
        else:
            sw = compute_sample_weights(df)
            print(f"Sample weights: min={sw.min():.2f}  max={sw.max():.2f}")

        # Asymmetric high-tail upweighting (independent of --no_sample_weights)
        if args.high_tail_mult > 1.0 and TARGET_COL in df.columns:
            y_vals = df[TARGET_COL].values.astype(np.float64)
            ht_mask = y_vals >= args.high_tail_thresh
            ht_w = np.where(ht_mask, args.high_tail_mult, 1.0)
            n_ht = ht_mask.sum()
            if sw is None:
                sw = ht_w
            else:
                sw = np.clip(sw * ht_w, 0.1, 10.0)
                sw = sw / sw.mean()   # renormalise
            print(f"High-tail upweighting: {n_ht} compounds (pEC50 >= {args.high_tail_thresh}) "
                  f"x{args.high_tail_mult}  sw range=[{sw.min():.2f},{sw.max():.2f}]")

        predictor = PXRMPNNPredictor(model_dir=args.model_dir)
        predictor._clf_weight = args.clf_weight
        predictor._act_thresh = args.act_thresh
        print(f"Multitask clf_weight  : {args.clf_weight}  "
              f"(0=pure regression, 1=pure classification)")
        print(f"Activity threshold    : {args.act_thresh} pEC50")
        cv_results = predictor.train(
            smiles_list     = df['SMILES_clean'].tolist(),
            y_primary       = df[TARGET_COL].values.astype(np.float32),
            descriptor_list = PXR_DESCRIPTOR_LIST,
            sample_weights  = sw,
            split_strategy  = args.split_strategy,
        )
        print(f"\nTraining done in {(time.time()-t0)/60:.1f} min")
        train_elapsed = (time.time() - t0) / 60

        y_vals = df[TARGET_COL].values.astype(np.float64)
        _log_cv = {
            "script": os.path.basename(__file__),
            "version": "v4_13d",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {
                "train_csv": args.train_csv,
                "inactive_supplement": args.inactive_supplement_csv or "disabled",
                "n_training_rows": int(len(df)),
                "ha_ge_5_5": int((y_vals >= 5.5).sum()),
                "ha_ge_6_0": int((y_vals >= 6.0).sum()),
                "inact_lt_3_5": int((y_vals < 3.5).sum()),
                "split_strategy": args.split_strategy,
                "sample_weights": not args.no_sample_weights,
                "high_tail_mult": args.high_tail_mult,
                "high_tail_thresh": args.high_tail_thresh,
                "seed": args.seed,
                "mode": args.mode,
            },
            "oof": {
                "rmse": cv_results["oof_rmse"],
                "mae": cv_results["oof_mae"],
                "r2": cv_results["oof_r2"],
                "rae": float(
                    np.sum(np.abs(np.array(predictor.oof_targets) - np.array(predictor.oof_predictions)))
                    / np.sum(np.abs(np.array(predictor.oof_targets) - np.array(predictor.oof_targets).mean()))
                ),
            },
            "folds": [
                {
                    "fold": fm["fold"] + 1,
                    "rmse": fm["rmse"],
                    "mae": fm["mae"],
                    "r2": fm["r2"],
                    "best_epoch": predictor.best_epochs[fm["fold"]]
                              if fm["fold"] < len(predictor.best_epochs) else None,
                }
                for fm in cv_results["fold_metrics"]
            ],
            "best_epochs": predictor.best_epochs,
            "avg_best_epoch": int(np.mean(predictor.best_epochs)) if predictor.best_epochs else None,
            "training_time_min": round(train_elapsed, 1),
            "model_dir": args.model_dir,
            "output_csv": args.output,
        }

    # ── FULL RETRAIN ──────────────────────────────────────────────────────────
    if args.mode == 'FULL':
        print(f"\nLoading: {args.train_csv}")
        df = pd.read_csv(args.train_csv).reset_index(drop=True)
        df.columns = [c.strip() for c in df.columns]
        smiles_actual = next(
            (c for c in df.columns if c.lower() == 'smiles'), SMILES_COL)
        if smiles_actual != SMILES_COL:
            df = df.rename(columns={smiles_actual: SMILES_COL})
        df['mol']          = df[SMILES_COL].apply(std.sanitize)
        df['SMILES_clean'] = df['mol'].apply(
            lambda m: Chem.MolToSmiles(m, canonical=True) if m else None)
        df = df.dropna(
            subset=['SMILES_clean', TARGET_COL]).reset_index(drop=True)
        print(f"Valid rows: {len(df)}  "
              f"pEC50=[{df[TARGET_COL].min():.2f},{df[TARGET_COL].max():.2f}]")

        predictor = PXRMPNNPredictor(model_dir=args.model_dir)
        predictor.train_full(
            smiles_list     = df['SMILES_clean'].tolist(),
            y_primary       = df[TARGET_COL].values.astype(np.float32),
            descriptor_list = PXR_DESCRIPTOR_LIST,
            sample_weights  = None,      # no weights (confirmed best)
            fixed_epochs    = args.full_epochs,
        )

        # Immediately predict test set with the full-retrain model
        print(f"\nLoading test: {args.test_csv}")
        tdf = pd.read_csv(args.test_csv)
        tdf.columns = [c.strip() for c in tdf.columns]
        smiles_actual = next(
            (c for c in tdf.columns if c.lower() == 'smiles'), SMILES_COL)
        if smiles_actual != SMILES_COL:
            tdf = tdf.rename(columns={smiles_actual: SMILES_COL})
        tdf['mol']          = tdf[SMILES_COL].apply(std.sanitize)
        tdf['SMILES_clean'] = tdf['mol'].apply(
            lambda m: Chem.MolToSmiles(m, canonical=True) if m else None)
        tdf['SMILES_clean'] = tdf['SMILES_clean'].fillna(tdf[SMILES_COL])
        test_smiles = tdf['SMILES_clean'].tolist()

        preds = predictor.predict(test_smiles, use_weighted=False)
        print(f"Full-retrain predictions: "
              f"[{preds.min():.3f}, {preds.max():.3f}]  "
              f"mean={preds.mean():.3f}  std={preds.std():.3f}")

        sub = pd.DataFrame()
        name_col = next(
            (c for c in tdf.columns
             if c.lower().replace(' ', '_') == 'molecule_name'), None)
        sub['Molecule Name'] = (tdf[name_col].values if name_col
                                else [f"compound_{i+1}" for i in range(len(tdf))])
        sub['SMILES'] = tdf[SMILES_COL].values
        sub['pEC50']  = preds
        sub.to_csv(args.output, index=False)
        print(f"Saved → {args.output}")

        # ── Save training log for FULL retrain ────────────────────────────
        full_log = {
            "script":           os.path.basename(__file__),
            "version":          "v4_4",
            "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode":             "FULL",
            "train_csv":        args.train_csv,
            "inactive_supplement": args.inactive_supplement_csv or "disabled",
            "full_epochs":      args.full_epochs,
            "model_dir":        args.model_dir,
            "output_csv":       args.output,
            "test_predictions": {
                "n":    int(len(preds)),
                "min":  round(float(preds.min()), 4),
                "max":  round(float(preds.max()), 4),
                "mean": round(float(preds.mean()), 4),
                "std":  round(float(preds.std()), 4),
            },
            "total_time_min":   round((time.time() - t0) / 60, 1),
        }
        save_training_log(full_log, args.model_dir, args.output)

        print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
        return

    # ── INFERENCE ─────────────────────────────────────────────────────────────
    if args.mode in ('INFERENCE', 'BOTH'):
        print(f"\nLoading test: {args.test_csv}")
        tdf = pd.read_csv(args.test_csv)
        tdf.columns = [c.strip() for c in tdf.columns]

        smiles_actual = next(
            (c for c in tdf.columns if c.lower() == 'smiles'), SMILES_COL)
        if smiles_actual != SMILES_COL:
            tdf = tdf.rename(columns={smiles_actual: SMILES_COL})

        tdf['mol']          = tdf[SMILES_COL].apply(std.sanitize)
        tdf['SMILES_clean'] = tdf['mol'].apply(
            lambda m: Chem.MolToSmiles(m, canonical=True) if m else None)
        tdf['SMILES_clean'] = tdf['SMILES_clean'].fillna(tdf[SMILES_COL])
        test_smiles = tdf['SMILES_clean'].tolist()

        predictor = PXRMPNNPredictor(
            model_dir=args.model_dir,
            checkpoint_version=args.ckpt_version)
        if not predictor.load_models():
            print("  Models not loaded — aborting.")
            return

        print("Generating MPNN predictions (simple average across folds) ...")
        preds = predictor.predict(test_smiles, use_weighted=False)
        print(f"  Range: [{preds.min():.3f}, {preds.max():.3f}]  "
              f"mean={preds.mean():.3f}  std={preds.std():.3f}")

        sub = pd.DataFrame()
        name_col = next(
            (c for c in tdf.columns
             if c.lower().replace(' ', '_') == 'molecule_name'), None)
        if name_col:
            sub['Molecule Name'] = tdf[name_col].values
        else:
            sub['Molecule Name'] = [f"compound_{i+1}"
                                    for i in range(len(tdf))]
            print("  WARNING: 'Molecule Name' not found — using sequential IDs")
        sub['SMILES'] = tdf[SMILES_COL].values
        sub['pEC50']  = preds


        sub.to_csv(args.output, index=False)
        print(f"Saved -> {args.output}")

        # ── Finalise and save training log (BOTH mode) ──────────────────────
        if args.mode == 'BOTH' and '_log_cv' in dir():
            _log_cv["test_predictions"] = {
                "n":    int(len(preds)),
                "min":  round(float(preds.min()), 4),
                "max":  round(float(preds.max()), 4),
                "mean": round(float(preds.mean()), 4),
                "std":  round(float(preds.std()), 4),
            }
            _log_cv["total_time_min"] = round((time.time() - t0) / 60, 1)
            save_training_log(_log_cv, args.model_dir, args.output)

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
