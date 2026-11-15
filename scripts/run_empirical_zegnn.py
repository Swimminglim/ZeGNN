# ZeGNN empirical analysis.
# Execute from repository root; see README.md.

import os
import random
import json
import math
import time
import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings("ignore")

# ----------------------------
# Publication-style (matplotlib-only)
# ----------------------------
def set_publication_style():
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif", "serif"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["lines.linewidth"] = 2.0
    plt.rcParams["lines.markersize"] = 7
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"

set_publication_style()

# ----------------------------
# Torch (MPS-first)
# ----------------------------
import torch
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

def log(msg: str):
    print(msg, flush=True)

# ============================================================
# Config: edit these for your environment
# ============================================================
# Default manifest path (will also search /mnt/data basename)
DEFAULT_MANIFEST_PATH = os.environ.get("MASTER_MANIFEST_PATH", "./inputs/ZENTROPY_sweep.csv")

# Default data paths
HOUSING_CSV_PATH = os.environ.get("HOUSING_CSV_PATH", "./data/kc_house_data.csv")
MENTAL_CSV_PATH  = os.environ.get("MENTAL_CSV_PATH", "./data/Mental_health.csv")
PM_CSV_PATH      = os.environ.get("PM_CSV_PATH", "./data/PM25_June2023_Mean.csv")

# Output root (default: alongside manifest if possible)
DEFAULT_OUT_ROOT = os.environ.get("MASTER_OUT_ROOT", "./outputs/empirical")  # default output root

# Case definitions (edit if needed)
CASES = {
    "PM": dict(
        csv_path=PM_CSV_PATH,
        y_col="pm",
        coord_cols=("x","y"),
        E_cols=["2m temperature","2m dewpoint temperature","Surface solar radiation downwards","Mean sea level pressure"],
        S_cols=["Total precipitation","Boundary layer height","10m u-component of wind","10m v-component of wind"],
        pm_keep_month=None,
        y_transform="none",  # stabilize heavy-tail and outliers
    ),
    "Housing": dict(
        csv_path=HOUSING_CSV_PATH,
        y_col="price",
        coord_cols=("long", "lat"),
        E_cols=["bedrooms","bathrooms","sqft_living","sqft_lot","floors","sqft_above","sqft_basement","sqft_living15","sqft_lot15"],
        S_cols=["waterfront","view","condition","grade","yr_built","yr_renovated"],
        pm_keep_month=None,
        y_transform="none",
    ),
    "MentalHealth": dict(
        csv_path=MENTAL_CSV_PATH,
        y_col="Mental_health_prevelence",
        coord_cols=("X","Y"),
        E_cols=["PopulationDensity","PopulationFemale","Population65","NoHealthInsurance","Neighbor_Disadvantage",
                "Beer","Casinos","DrinkingPlaces","X.HouseRent"],
        S_cols=["Generalhosptials","NatureParks","MentalHealthPati"],
        pm_keep_month=None,
        y_transform="none",
    ),
}
# CV settings
N_SPLITS = 5
SPATIAL_GRID_N = 5  # 5x5 blocks -> GroupKFold(5) by block groups

# Moran residual settings (in-sample)
MORAN_KNN_K = 8

# Torch training settings (Zentropy)
EPOCHS = 1500
PATIENCE = 80          # [IMPROVED] increased from 60 for more stable convergence
LR = 1e-3
WD = 1e-4
GRAD_CLIP = 1.0
DROPOUT = 0.1
HIDDEN = 64            # keep 64; 128 tended to over-parameterize sparse PM data
USE_LR_SCHEDULER = True  # [IMPROVED] enable ReduceLROnPlateau (factor=0.5, patience=20)

# ----------------------------
# PM-only hyperparameter selection (SpatialCV)
#
# Core rule: use the same epoch and patience settings for search and final runs
# ----------------------------
N_SPLITS_SELECT = 5   # folds used for hyperparameter search (final CV also uses 5)

# ----------------------------
# Sparse Dirichlet prior design
#
# K_UPPER=5 is the upper bound on the number of regimes. The active regime count is learned.
# lam_balance was removed and replaced with a sparse Dirichlet prior.
#
# Rationale:
#   Dirichlet(α) with α < 1: mode at simplex vertices → sparse regime assignment
#   Loss term: lam_sparse × (1 − ALPHA_DIR) × Σ_k log(mean_p_k)
#     → negative contribution; more negative at sparser solutions
#     → optimizer naturally pushed toward sparse regimes
#
#   n_eff = exp(H(p̄)) = exp(−Σ_k p̄_k log p̄_k)


# ----------------------------
K_UPPER   = 5
ALPHA_DIR = 0.1   # Dirichlet concentration < 1 → sparse (0.1 strongly, 0.5 weakly)

# SpatialCV HP re-selection grids
GRAPH_K_GRID_V2 = (8, 10,12,14, 16)
LAM_SPARSE_GRID = (1e-3, 1e-2, 1e-1)   # sparse prior strength
LAM_MAG_GRID_V2 = (1e-4, 1e-3, 1e-2)   # E_k/S_k scale control

# Random seed
SEED = 0

# ----------------------------
# the entropy-adjusted model stabilization knobs
# ----------------------------
ENTROPY_NORM_LOGK = True
TEFF_CLIP_MIN = None
TEFF_CLIP_MAX = None

# ----------------------------
# HP search strategy
# ----------------------------
HP_SEARCH_MODE = "auto"
HP_MAX_TRIALS  = 27

# ============================================================
# Utilities
# ============================================================
def resolve_path(p: str) -> str:
    if p is None:
        raise ValueError("Path is None")
    if os.path.exists(p):
        return p
    b = os.path.basename(p)
    alt = os.path.join("/mnt/data", b)
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"Not found: {p} (also tried {alt})")

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def set_all_seeds(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def zentropy_model_name(variant: str) -> str:
    """Human-readable key used in CSV/fig outputs."""
    v = (variant or "").strip().lower()
    if v in ("v2_entropy_complete", "v2entropycomplete", "entropy_complete", "entropyinf", "copilot17_19", "copilot"):
        return "ZentropyV2_EntropyInF"
    return "ZentropyV2"
set_all_seeds(SEED)
# ============================================================
# Regime label alignment ("swap") for real data CV
# - No ground-truth regimes in PM, so we align fold-specific regimes
#   to a reference model fitted on the full data using a soft overlap criterion.
# - This affects ONLY diagnostics/plots of p, E_k, S_k, etc. Predictions F are permutation-invariant.
# ============================================================
from itertools import permutations

def best_perm_overlap(p_ref: np.ndarray, p_fold: np.ndarray, idx: np.ndarray) -> Tuple[int, ...]:
    """Find permutation that maximizes overlap sum_i sum_k p_fold[i,k] * p_ref[i,perm(k)] over idx."""
    idx = np.asarray(idx, dtype=int)
    K = p_ref.shape[1]
    # overlap matrix M[a,b] = sum_i p_fold[i,a] * p_ref[i,b]
    M = np.zeros((K, K), dtype=float)
    pr = p_ref[idx]
    pf = p_fold[idx]
    for a in range(K):
        for b in range(K):
            M[a, b] = float(np.sum(pf[:, a] * pr[:, b]))
    # brute force for small K (<=7 is safe); fallback to greedy otherwise
    if K <= 7:
        best_score = -1e18
        best = tuple(range(K))
        for perm in permutations(range(K)):
            score = 0.0
            for a, b in enumerate(perm):
                score += M[a, b]
            if score > best_score:
                best_score = score
                best = perm
        return best
    # greedy fallback
    used_b = set()
    perm = [-1] * K
    for a in range(K):
        b = int(np.argmax([M[a, j] if j not in used_b else -1e18 for j in range(K)]))
        perm[a] = b
        used_b.add(b)
    return tuple(perm)

def apply_perm_out(out_np: dict, perm: Tuple[int, ...]) -> dict:
    """Apply mapping fold-regime a -> ref-regime perm[a] to p and per-regime tensors."""
    K = len(perm)
    out2 = dict(out_np)
    # p: (N,K)
    if "p" in out_np:
        p = np.asarray(out_np["p"])
        p2 = np.zeros_like(p)
        for a, b in enumerate(perm):
            p2[:, b] = p[:, a]
        out2["p"] = p2
    # E_k, S_k, F_k: (N,K)
    for key in ["E_k", "S_k"]:
        if key in out_np:
            A = np.asarray(out_np[key])
            A2 = np.zeros_like(A)
            for a, b in enumerate(perm):
                A2[:, b] = A[:, a]
            out2[key] = A2
    # T, mu: (K,)
    for key in ["T", "mu"]:
        if key in out_np:
            v = np.asarray(out_np[key])
            v2 = np.zeros_like(v)
            for a, b in enumerate(perm):
                v2[b] = v[a]
            out2[key] = v2
    return out2

def _perm_map_from_order(order: np.ndarray) -> Tuple[int, ...]:
    """Given order (new positions contain old indices), return mapping old_index -> new_index."""
    order = list(map(int, list(order)))
    K = len(order)
    inv = [0] * K
    for new_pos, old_idx in enumerate(order):
        inv[old_idx] = new_pos
    return tuple(inv)

def canonical_perm_from_p(p: np.ndarray, mode: str = "mean_p_desc") -> Tuple[int, ...]:
    """
    Create a deterministic canonical regime ordering for a reference solution.
    Returns mapping old_index -> new_index.
    Default: sort regimes by descending mean probability.
    """
    p = np.asarray(p, dtype=float)
    K = p.shape[1]
    mean_p = np.nanmean(p, axis=0)
    if mode == "mean_p_desc":
        order = np.argsort(-mean_p)  # new positions
    else:
        order = np.arange(K)
    return _perm_map_from_order(order)

def swap_overlap_score(p_ref: np.ndarray, p_fold: np.ndarray, idx: np.ndarray) -> float:
    """Soft overlap score used for swap: sum_i sum_k p_fold(i,k) * p_ref(i,k) over idx."""
    idx = np.asarray(idx, dtype=int)
    pr = np.asarray(p_ref, float)[idx]
    pf = np.asarray(p_fold, float)[idx]
    return float(np.sum(pf * pr))



# ============================================================
# Spatial groups: 5x5 grid -> 25 groups -> GroupKFold(5)
# ============================================================
def make_spatial_groups(coords: np.ndarray, grid_n: int = 5) -> np.ndarray:
    """
    Spatial CV groups for GroupKFold.

    Partition the coordinates into contiguous spatial folds.
    The number of folds is determined by grid_n.

    - grid_n specifies the number of folds.
    - Group IDs range from 0 to grid_n - 1.
    - Quantiles along the first PCA axis define the spatial groups.
      This encourages geographically contiguous holdout regions.

    Usage:
      groups = make_spatial_groups(coords, grid_n=5)
      gkf = GroupKFold(n_splits=5)
      splits = list(gkf.split(X_std, y_std, groups=groups))
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must be (N,2) array")

    n_folds = int(max(2, grid_n))
    n = coords.shape[0]
    if n < n_folds:

        return np.zeros(n, dtype=int)


    try:
        from sklearn.decomposition import PCA
        z = PCA(n_components=1, random_state=0).fit_transform(coords).reshape(-1)
    except Exception:

        z = coords[:, 0].copy()

    # quantile cut -> group ids 0..n_folds-1
    qs = np.linspace(0.0, 1.0, n_folds + 1)
    bins = np.quantile(z, qs)



    if np.unique(bins).size < bins.size:
        rng = np.random.default_rng(0)
        z = z + rng.normal(scale=1e-9, size=z.shape[0])
        bins = np.quantile(z, qs)


    inner = bins[1:-1]
    groups = np.digitize(z, inner, right=True).astype(int)  # 0..n_folds-1


    if np.unique(groups).size < 2:
        z2 = coords[:, 0] + coords[:, 1]
        bins2 = np.quantile(z2, qs)
        if np.unique(bins2).size < bins2.size:
            rng = np.random.default_rng(1)
            z2 = z2 + rng.normal(scale=1e-9, size=z2.shape[0])
            bins2 = np.quantile(z2, qs)
        groups = np.digitize(z2, bins2[1:-1], right=True).astype(int)

    return groups

# ============================================================
# Graph utilities (KNN edges + EDGE-SPMM diffusion)
# edge_index: shape (2,E) with [dst, src]
# ============================================================
def build_knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    n = coords.shape[0]
    k_eff = int(min(k + 1, n))
    nnbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(coords)
    _, idx = nnbrs.kneighbors(coords, return_distance=True)
    idx = idx[:, 1:]  # drop self
    dst = np.repeat(np.arange(n), idx.shape[1])
    src = idx.reshape(-1)
    edge_index = np.vstack([dst, src]).astype(np.int64)
    return edge_index

def edges_to_device(edges_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(edges_np, dtype=torch.long, device=device)

def spmm_edge(edges_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dst = edges_t[0]
    src = edges_t[1]
    out = torch.zeros_like(x)
    out.index_add_(0, dst, x[src])
    deg = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
    out = out / deg.clamp_min(1.0).unsqueeze(1)
    return out

# ============================================================
# Moran's I (global) from KNN edges (row-standardized)
# ============================================================
def global_moran_I_from_edges(residual: np.ndarray, edges_full: np.ndarray) -> float:
    r = np.asarray(residual, dtype=float).reshape(-1)
    n = r.size
    if n < 3:
        return float("nan")
    z = r - np.mean(r)

    dst = edges_full[0].astype(np.int64)
    src = edges_full[1].astype(np.int64)

    wz = np.zeros(n, dtype=float)
    deg = np.zeros(n, dtype=float)
    np.add.at(wz, dst, z[src])
    np.add.at(deg, dst, 1.0)
    deg = np.clip(deg, 1.0, None)
    wz = wz / deg

    num = float(np.dot(z, wz))
    den = float(np.dot(z, z))
    if den <= 0:
        return float("nan")
    return num / den

# ============================================================
# Data preprocessing
# - y_transform: none | log1p (PM)
# - y standardized after transform
# - Housing binary handling: waterfront, yr_renovated (no scaling)
# ============================================================
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ----------------------------
# y transform helper
# ----------------------------
@dataclass
class YTransform:
    name: str

    def forward(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.name == "log1p":
            # robust to small negatives: clip to 0 before log1p
            return np.log1p(np.clip(y, a_min=0.0, a_max=None))
        return y

    def inverse(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)
        if self.name == "log1p":
            return np.expm1(y)
        return y


def _looks_like_lonlat(coords_xy: np.ndarray) -> bool:
    if coords_xy is None or len(coords_xy) == 0:
        return False
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return False
    return (np.nanmax(np.abs(x)) <= 180.0) and (np.nanmax(np.abs(y)) <= 90.0)


def maybe_project_lonlat_to_5070(coords_xy: np.ndarray):
    """
    Returns: (coords_out, did_project: bool, crs_tag: str)
    """
    coords_xy = np.asarray(coords_xy, dtype=np.float64)
    if coords_xy.ndim != 2 or coords_xy.shape[1] != 2:
        raise ValueError("coords must be (N,2) array")

    if not _looks_like_lonlat(coords_xy):
        return coords_xy.astype(np.float32), False, "as_is"

    try:
        from pyproj import Transformer
        tfm = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
        xx, yy = tfm.transform(coords_xy[:, 0], coords_xy[:, 1])
        out = np.vstack([xx, yy]).T.astype(np.float32)
        return out, True, "EPSG:5070"
    except Exception as e:
        return coords_xy.astype(np.float32), False, f"lonlat_fallback({type(e).__name__})"


def preprocess_case(df: pd.DataFrame, case: str, cfg: dict):
    try:
        _log = log
    except Exception:
        _log = print


    if str(case).startswith("PM") and cfg.get("pm_keep_month"):
        keep = str(cfg["pm_keep_month"])
        month_col = None
        for c in ["month", "Month", "YYYY-MM", "yyyy_mm", "year_month"]:
            if c in df.columns:
                month_col = c
                break
        if month_col is not None:
            df = df.loc[df[month_col].astype(str) == keep].copy()

    if case == "Housing":
        if "waterfront" in df.columns:
            df["waterfront"] = (pd.to_numeric(df["waterfront"], errors="coerce").fillna(0.0) > 0).astype(int)
        if "yr_renovated" in df.columns:
            df["yr_renovated"] = (pd.to_numeric(df["yr_renovated"], errors="coerce").fillna(0.0) > 0).astype(int)

    y_col = cfg["y_col"]
    coord_cols = cfg["coord_cols"]
    E_cols = cfg["E_cols"]
    S_cols = cfg["S_cols"]

    feat_cols = [c for c in (E_cols + S_cols) if c in df.columns]
    if len(feat_cols) == 0:
        raise ValueError(f"[{case}] No feature columns found. Check E_cols/S_cols.")

    y_raw = pd.to_numeric(df[y_col], errors="coerce")
    if y_raw.isna().all():
        raise ValueError(f"[{case}] target '{y_col}' all-NaN.")
    y_raw = y_raw.fillna(y_raw.median()).values.astype(np.float32)

    coords_raw = df.loc[:, list(coord_cols)].values.astype(np.float32)
    coords, did_project, crs_tag = maybe_project_lonlat_to_5070(coords_raw)


    if str(case).startswith("PM") and _looks_like_lonlat(coords_raw.astype(np.float64)) and (not did_project):
        raise RuntimeError(
            f"[{case}] coords look like lon/lat but projection failed ({crs_tag}). "
            "Install pyproj or provide projected coordinates."
        )

    _log(f"[{case}] coords_crs={crs_tag} | did_project={did_project} | "
         f"x[{coords[:,0].min():.1f},{coords[:,0].max():.1f}] y[{coords[:,1].min():.1f},{coords[:,1].max():.1f}]")

    # X build
    X_df = df.loc[:, feat_cols].copy()
    for c in X_df.columns:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X = X_df.values.astype(np.float32)


    bin_cols = []
    if case == "Housing":
        for c in ["waterfront", "yr_renovated"]:
            if c in feat_cols:
                bin_cols.append(c)
    for j, c in enumerate(feat_cols):
        if c in bin_cols:
            continue
        uq = set(np.unique(X[:, j]))
        if uq.issubset({0.0, 1.0}) and len(uq) <= 2:
            bin_cols.append(c)

    cont_cols = [c for c in feat_cols if c not in bin_cols]
    cont_idx = [feat_cols.index(c) for c in cont_cols]

    X_std = X.copy()
    if len(cont_idx) > 0:
        x_scaler = StandardScaler()
        X_std[:, cont_idx] = x_scaler.fit_transform(X_std[:, cont_idx]).astype(np.float32)


    ytf = YTransform(cfg.get("y_transform", "none"))
    y_t = ytf.forward(y_raw).astype(np.float32)

    y_scaler = StandardScaler()
    y_std = y_scaler.fit_transform(y_t.reshape(-1, 1)).reshape(-1).astype(np.float32)

    c_scaler = StandardScaler()
    C_std = c_scaler.fit_transform(coords).astype(np.float32)

    E_idx = [feat_cols.index(c) for c in E_cols if c in feat_cols]
    S_idx = [feat_cols.index(c) for c in S_cols if c in feat_cols]

    return dict(
        df=df, y_raw=y_raw, y_trans=y_t, ytf=ytf, y_scaler=y_scaler, y_std=y_std,
        coords=coords, C_std=C_std,
        X=X, X_std=X_std,
        feat_cols=feat_cols, cont_cols=cont_cols, bin_cols=bin_cols,
        E_idx=E_idx, S_idx=S_idx,
        coords_crs=crs_tag, did_project=did_project
    )

# ============================================================
# Torch models: DNN, GNN, ZentropyV2
# ============================================================

class MLPBlock(nn.Module):
    def __init__(self, d_in, hidden=128, drop=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop),
        )
    def forward(self, x):
        return self.net(x)


class DeepZentropyV2(nn.Module):
    """
    Matches user's V2 code (EDGE-SPMM diffusion on gating encoder).
    """
    def __init__(self, d_in, E_idx, S_idx, K=4, hidden=128, drop=0.10, gate_use_coords=True, entropy_in_F: bool = False):
        super().__init__()
        self.K = int(K)
        self.E_idx = np.array(E_idx, dtype=np.int64)
        self.S_idx = np.array(S_idx, dtype=np.int64)
        self.gate_use_coords = bool(gate_use_coords)
        self.entropy_in_F = bool(entropy_in_F)

        d_gate = d_in + (2 if gate_use_coords else 0)
        self.gate_enc = MLPBlock(d_gate, hidden=hidden, drop=drop)
        self.gate_head = nn.Linear(hidden, self.K)

        self.E_enc = MLPBlock(len(self.E_idx), hidden=hidden, drop=drop)
        self.S_enc = MLPBlock(len(self.S_idx), hidden=hidden, drop=drop)
        self.E_head = nn.Linear(hidden, self.K)
        self.S_head = nn.Linear(hidden, self.K)

        self.T_raw = nn.Parameter(torch.zeros(self.K))

    def forward(self, X, coords, edges_t):
        if self.gate_use_coords:
            gate_in = torch.cat([X, coords], dim=1)
        else:
            gate_in = X

        h_gate = self.gate_enc(gate_in)
        h_gate = spmm_edge(edges_t, h_gate)
        logits = self.gate_head(h_gate)
        p = torch.softmax(logits, dim=1)

        hE = self.E_enc(X[:, self.E_idx])
        hS = self.S_enc(X[:, self.S_idx])
        E_k = self.E_head(hE)
        S_k = self.S_head(hS)

        T = F.softplus(self.T_raw) + 1e-6
        F_k = E_k - (S_k * T.unsqueeze(0))

        # Base mixture free energy (MoE-style)
        F_mix = torch.sum(p * F_k, dim=1)
        
        # the entropy-adjusted model: include regime-mixing entropy in the predicted free energy (optional).
        # Effective temperature per node: T_eff(i) = sum_k p_ik * T_k
        # Mixing entropy per node: H(i) = - sum_k p_ik log p_ik
        # Entropy-complete free energy: F_i = sum_k p_ik F_ik + T_eff(i) * H(i)
        eps = 1e-9
        H_raw = -torch.sum(p * torch.log(p + eps), dim=1)

        # (A-only) Scale-stabilized mixing entropy: H_norm = H / log(K).
        K_eff = int(p.shape[1])
        if K_eff > 1:
            logK = torch.log(torch.tensor(float(K_eff), device=H_raw.device, dtype=H_raw.dtype))
            H_use = H_raw / (logK + 1e-12)
        else:
            H_use = torch.zeros_like(H_raw)

        # Effective temperature per node (as in pp.17–19)
        T_eff = torch.sum(p * T.unsqueeze(0), dim=1)

        if self.entropy_in_F:
            F_mix = F_mix + T_eff * H_use

        
        E_mix = torch.sum(p * E_k, dim=1)
        S_mix = torch.sum(p * S_k, dim=1)
        
        return {"F": F_mix, "E": E_mix, "S": S_mix, "p": p, "E_k": E_k, "S_k": S_k, "T": T,
        "H_node": H_raw, "H_node_norm": H_use, "T_eff": T_eff, "entropy_in_F": self.entropy_in_F}





def zentropy_loss(out, y_std_t, mask_t, lam_sparse, lam_magnitude):
    """
    Loss for ZentropyV2 with Sparse Dirichlet Prior.

    - MSE: prediction loss
    - Sparse Dirichlet prior (lam_sparse):
        log p(π̄ | Dir(α)) ∝ (α−1) Σ_k log π̄_k
        α = ALPHA_DIR < 1  →  mode at simplex vertices  →  sparse regimes
        Loss term = lam_sparse × (1 − ALPHA_DIR) × Σ_k (−log mean_p_k)
        = lam_sparse × (1 − α) × Σ_k (−log mean_p_k)
        Minimizing loss pushes mean_p_k toward 0/1 (sparse), NOT uniform.
    - lam_magnitude: L2 penalty on E_k, S_k (scale control)

    K is an upper bound; report the effective number of regimes using n_eff = exp(H(p_bar)).
    """
    mse = torch.mean((out["F"][mask_t] - y_std_t[mask_t])**2)

    p = out["p"][mask_t]
    eps = 1e-9
    mean_p = torch.mean(p, dim=0)  # (K,) marginal regime usage

    # Sparse Dirichlet prior: (1 - α) × Σ_k (−log mean_p_k)
    # α < 1 → (1−α) > 0 → penalizes uniform mean_p → pushes toward sparse
    sparse_penalty = (1.0 - ALPHA_DIR) * torch.sum(-torch.log(mean_p + eps))

    Ek = out["E_k"][mask_t]
    Sk = out["S_k"][mask_t]
    mag = torch.mean(Ek**2) + torch.mean(Sk**2)

    loss = mse + float(lam_sparse) * sparse_penalty + float(lam_magnitude) * mag

    # n_eff = exp(H(p̄)): effective number of active regimes
    H_mean_p = -torch.sum(mean_p * torch.log(mean_p + eps))
    n_eff = float(torch.exp(H_mean_p).item())

    parts = {
        "mse": float(mse.item()),
        "sparse_penalty": float(sparse_penalty.item()),
        "mag": float(mag.item()),
        "n_eff": n_eff,
        "mean_p": mean_p.detach().cpu().numpy().tolist(),
    }
    return loss, parts


def split_train_val(idx: np.ndarray, val_ratio: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.array(idx, dtype=int)
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_ratio))
    va = idx[:n_val]
    tr = idx[n_val:]
    return tr, va


def train_zentropy_fullgraph(
    X_std, C_std, y_std, edges_np,
    E_idx, S_idx, K,
    tr_idx, va_idx,
    max_epochs, patience, lr, wd,
    lam_sparse, lam_magnitude,
    device,
    variant: str = "v2",
):
    n, d = X_std.shape
    X_t = torch.tensor(X_std, device=device)
    C_t = torch.tensor(C_std, device=device)
    y_t = torch.tensor(y_std, device=device)
    edges_t = edges_to_device(edges_np, device=device)

    tr_mask = torch.zeros(n, dtype=torch.bool, device=device); tr_mask[tr_idx] = True
    va_mask = torch.zeros(n, dtype=torch.bool, device=device); va_mask[va_idx] = True

    vlow = str(variant).lower()
    if vlow in ["v2_entropy_complete", "entropy_complete", "copilot17_19", "copilot1719"]:
        model = DeepZentropyV2(d_in=d, E_idx=E_idx, S_idx=S_idx, K=K, hidden=HIDDEN, drop=DROPOUT, gate_use_coords=True, entropy_in_F=True).to(device)
    else:
        model = DeepZentropyV2(d_in=d, E_idx=E_idx, S_idx=S_idx, K=K, hidden=HIDDEN, drop=DROPOUT, gate_use_coords=True, entropy_in_F=False).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = None
    if USE_LR_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20, min_lr=1e-5)

    best = 1e18
    best_state = None
    wait = 0

    for ep in range(1, max_epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)

        out = model(X_t, C_t, edges_t)
        loss, _ = zentropy_loss(out, y_t, tr_mask, lam_sparse, lam_magnitude)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

        model.eval()
        with torch.no_grad():
            outv = model(X_t, C_t, edges_t)
            vloss = torch.mean((outv["F"][va_mask] - y_t[va_mask])**2).item()

        if scheduler is not None:
            scheduler.step(vloss)

        if vloss < best - 1e-9:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out = model(X_t, C_t, edges_t)

    out_np = {k: v.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(v) else v for k, v in out.items()}
    return model, out_np

def safe_train_zentropy(*args, **kwargs):
    try:
        return train_zentropy_fullgraph(*args, **kwargs)
    except RuntimeError as e:
        msg = str(e).lower()
        if ("mps" in msg) or ("out of memory" in msg) or ("metal" in msg):
            log("    [WARN] device runtime error -> fallback CPU")
            kwargs2 = dict(kwargs)
            kwargs2["device"] = torch.device("cpu")
            return train_zentropy_fullgraph(*args, **kwargs2)
        raise

# ============================================================
# FASTGWR (fast local WLS with kNN bandwidth)
# - Select bw ONCE per case (spatial CV on y_std)
# ============================================================
def _wls_predict_one(X_nb: np.ndarray, y_nb: np.ndarray, w: np.ndarray, x0: np.ndarray, ridge: float = 1e-6) -> float:
    # Weighted ridge regression beta = (X'WX + ridge I)^-1 X'Wy
    W = w.reshape(-1, 1)
    Xw = X_nb * W
    XtX = X_nb.T @ Xw
    XtX.flat[:: XtX.shape[0] + 1] += ridge
    Xty = X_nb.T @ (y_nb * w)
    beta = np.linalg.solve(XtX, Xty)
    return float(x0 @ beta)



# ============================================================
# Manifest reader (Zentropy hyperparams)
# ============================================================
def read_manifest(manifest_path: str) -> pd.DataFrame:
    p = resolve_path(manifest_path)
    df = pd.read_csv(p)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # expected columns (flexible)
    # case, graph_k, K, lam_sparse, lam_magnitude
    return df

def get_case_hparams_from_manifest(df: pd.DataFrame, case: str) -> dict:
    # try case matches
    cols = {c.lower(): c for c in df.columns}
    case_col = cols.get("case", None)
    if case_col is None:
        # sometimes "dataset" or "Case"
        for k in ["dataset", "Case", "CASE"]:
            if k in df.columns:
                case_col = k
                break
    if case_col is None:
        raise ValueError("Manifest must contain a 'case' (or 'dataset') column.")

    sub = df.loc[df[case_col].astype(str) == str(case)].copy()
    if sub.empty:
        # fallback: partial match
        sub = df.loc[df[case_col].astype(str).str.contains(str(case), regex=False)].copy()
    if sub.empty:
        raise ValueError(f"Manifest has no row for case='{case}'")

    row = sub.iloc[0]
    def pick(*names, default=None):
        for nm in names:
            if nm in df.columns:
                v = row[nm]
                return v
            if nm.lower() in cols:
                v = row[cols[nm.lower()]]
                return v
        return default

    graph_k = int(pick("graph_k", "k_graph", "knn_k"))
    K = int(pick("K", "regime_K", "n_regimes"))
    lam_sparse = float(pick("lam_sparse", "lambda_sparse", default=0.01))
    lam_magnitude = float(pick("lam_magnitude", "lambda_magnitude", "lambda_mag", default=0.001))


    K = int(K_UPPER)
    return dict(graph_k=graph_k, K=K, lam_sparse=lam_sparse, lam_magnitude=lam_magnitude)

# ============================================================
# Metrics helpers (supports PM log transform export)
# ============================================================
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(a, b)))

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return dict(RMSE=rmse(y_true, y_pred), R2=float(r2_score(y_true, y_pred)))


def regime_usage_metrics(p_arr: Optional[np.ndarray]) -> Dict[str, float]:
    """Return regime-utilization diagnostics from probability matrix p (N,K)."""
    out = {"n_eff_global": float("nan"), "n_eff_local": float("nan"), "argmax_share_max": float("nan")}
    try:
        if p_arr is None:
            return out
        p = np.asarray(p_arr, dtype=float)
        if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
            return out
        mask = np.all(np.isfinite(p), axis=1)
        p = p[mask]
        if p.shape[0] == 0:
            return out
        eps = 1e-12
        mean_p = np.nanmean(p, axis=0)
        mean_p = np.clip(mean_p, eps, None)
        mean_p = mean_p / np.sum(mean_p)
        H_global = -np.sum(mean_p * np.log(mean_p + eps))
        out["n_eff_global"] = float(np.exp(H_global))

        H_local = -np.sum(p * np.log(np.clip(p, eps, None)), axis=1)
        out["n_eff_local"] = float(np.nanmean(np.exp(H_local)))

        argmax_idx = np.argmax(p, axis=1)
        counts = np.bincount(argmax_idx, minlength=p.shape[1]).astype(float)
        shares = counts / max(1.0, float(p.shape[0]))
        out["argmax_share_max"] = float(np.max(shares))
    except Exception:
        pass
    return out

def inv_to_original_space(y_pred_std: np.ndarray, y_scaler: StandardScaler, ytf: YTransform) -> np.ndarray:
    y_pred_trans = y_scaler.inverse_transform(y_pred_std.reshape(-1, 1)).reshape(-1)
    y_pred_raw = ytf.inverse(y_pred_trans)
    return y_pred_raw.astype(np.float64)


# ============================================================
# SpatialCV-based HP selection
# ============================================================
def _oof_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    return float(r2_score(y_true[m], y_pred[m]))

def select_hp_spatialcv(
    case: str,
    X_std: np.ndarray,
    C_std: np.ndarray,
    y_std: np.ndarray,
    coords: np.ndarray,
    data: dict,
    hp_zen: dict,
    spatial_splits,
    zentropy_variant: str,
) -> Tuple[dict, List[dict], Optional[dict]]:
    """
    SpatialCV HP selection for ZentropyV2 (entropy-in-F).

    Hyperparameter selection:
    - Use the same epoch and patience settings for tuning and final estimation.
      This avoids differences caused by training budgets.
    - Use a smaller selection CV while retaining five folds for final evaluation.
    - Select graph_k, lam_sparse, and lam_magnitude with K_UPPER fixed.
    """
    base_hp = dict(hp_zen)
    v = "v2_entropy_complete"
    rows: List[dict] = []


    base_hp["K"] = int(K_UPPER)
    base_hp.setdefault("graph_k", 12)
    base_hp.setdefault("lam_sparse", 0.01)
    base_hp.setdefault("lam_magnitude", 0.001)

    gk_grid  = list(GRAPH_K_GRID_V2) if GRAPH_K_GRID_V2 else [int(base_hp["graph_k"])]
    sp_grid  = list(LAM_SPARSE_GRID)
    mag_grid = list(LAM_MAG_GRID_V2)

    total_full = max(1, len(gk_grid) * len(sp_grid) * len(mag_grid))
    cap = int(HP_MAX_TRIALS) if HP_MAX_TRIALS and HP_MAX_TRIALS > 0 else 0
    mode = str(HP_SEARCH_MODE).strip().lower()

    def _reservoir_sample(it, k, seed):
        if k <= 0:
            return list(it)
        rng = random.Random(int(seed))
        res = []
        for n, combo in enumerate(it, 1):
            if n <= k:
                res.append(combo)
            else:
                j = rng.randrange(n)
                if j < k:
                    res[j] = combo
        return res

    if mode == "fullgrid" or cap <= 0 or total_full <= cap:
        combos = list(itertools.product(gk_grid, sp_grid, mag_grid))
        mode_used = "fullgrid"
    else:
        combos = _reservoir_sample(itertools.product(gk_grid, sp_grid, mag_grid), cap, SEED + 123)
        mode_used = "random"


    from sklearn.model_selection import GroupKFold as _GKF
    _groups = make_spatial_groups(coords, grid_n=SPATIAL_GRID_N)
    _select_splits = list(_GKF(n_splits=N_SPLITS_SELECT).split(X_std, y_std, _groups))

    log(f"[SELECT] SpatialCV HP selection for {case}: "
        f"K={K_UPPER}(fixed, upper bound, alpha={ALPHA_DIR}), "
        f"gk={gk_grid}, lam_sparse={sp_grid}, lam_magnitude={mag_grid} "
        f"({len(combos)}/{total_full} combos, mode={mode_used}, "
        f"epochs={EPOCHS}, folds={N_SPLITS_SELECT})")

    best = None
    best_r2 = -1e18
    best_artifacts = None
    edges_cache = {}

    for t, (gk, lam_sp, lam_mag) in enumerate(combos, 1):
        gk = int(gk)
        lam_sp = float(lam_sp); lam_mag = float(lam_mag)

        if gk not in edges_cache:
            edges_cache[gk] = build_knn_edges(coords, k=gk)
        edges_zen = edges_cache[gk]

        trial = dict(base_hp)
        trial["graph_k"] = gk
        trial["lam_sparse"] = lam_sp
        trial["lam_magnitude"] = lam_mag

        try:
            oof_F, oof_p, oof_ent = run_cv_zentropy(
                X_std, C_std, y_std, edges_zen,
                data["E_idx"], data["S_idx"],
                trial, _select_splits,
                zentropy_variant=v,
                p_ref=None,
            )
            r2 = _oof_r2(y_std, oof_F)
        except Exception as e:
            oof_F, oof_p, oof_ent = None, None, None
            r2 = float("nan")
            log(f"    [SELECT-WARN] combo {t}/{len(combos)} failed "
                f"(gk={gk}, sparse={lam_sp}, mag={lam_mag}) "
                f"({type(e).__name__}: {e})")

        rows.append({"variant": v, "graph_k": gk, "K": K_UPPER,
                     "lam_sparse": lam_sp, "lam_magnitude": lam_mag, "r2_oof": r2})

        if np.isfinite(r2) and r2 > best_r2:
            best_r2 = r2
            best = (gk, lam_sp, lam_mag)
            best_artifacts = {
                "oof_F": None if oof_F is None else np.asarray(oof_F, dtype=float).copy(),
                "oof_p": None if oof_p is None else np.asarray(oof_p, dtype=float).copy(),
                "oof_ent": None if oof_ent is None else np.asarray(oof_ent, dtype=float).copy(),
                "splits_n": int(N_SPLITS_SELECT),
                "r2_oof": float(r2),
            }

        if len(combos) >= 5 and (t % max(1, len(combos) // 10) == 0 or t == len(combos)):
            log(f"    [SELECT] {t}/{len(combos)} | best_r2={best_r2:.4f}")

    return_hp = dict(base_hp)
    if best is not None:
        gk, lam_sp, lam_mag = best
        return_hp["graph_k"] = int(gk)
        return_hp["lam_sparse"] = float(lam_sp)
        return_hp["lam_magnitude"] = float(lam_mag)
        log(f"[SELECT] Best HP → graph_k={gk}, K={K_UPPER}(fixed), "
            f"lam_sparse={lam_sp}, lam_magnitude={lam_mag} | r2_oof={best_r2:.4f}")
    else:
        log("[SELECT] No valid combo found; using base HP.")

    return return_hp, rows, best_artifacts

# ============================================================
# CV runners (Random / Spatial)
# ============================================================





def run_cv_zentropy(X_std: np.ndarray, C_std: np.ndarray, y_std: np.ndarray, edges_np: np.ndarray,
                    E_idx: List[int], S_idx: List[int],
                    hp: dict,
                    cv_splits,
                    zentropy_variant: str = "v2_entropy_complete",
                    p_ref: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (oof_F, oof_p_aligned, oof_entropy_aligned) in y_std space.
    Notes:
      - oof_F is the OOF prediction (permutation-invariant).
      - oof_p_aligned / entropy are for diagnostics only (regime label 'swap' aligned to p_ref if provided).
    """
    n = len(y_std)
    K = int(hp["K"])
    oof_F = np.zeros(n, dtype=float)
    oof_p = np.full((n, K), np.nan, dtype=float)
    oof_ent = np.full(n, np.nan, dtype=float)

    for fold, (tr_idx, te_idx) in enumerate(cv_splits, 1):
        tr2, va2 = split_train_val(tr_idx, val_ratio=0.2, seed=SEED + fold)
        _, out_np = safe_train_zentropy(
            X_std=X_std, C_std=C_std, y_std=y_std, edges_np=edges_np,
            E_idx=E_idx, S_idx=S_idx, K=int(hp["K"]),
            tr_idx=tr2, va_idx=va2,
            max_epochs=EPOCHS, patience=PATIENCE, lr=LR, wd=WD,
            lam_sparse=float(hp.get("lam_sparse", 0.01)),
            lam_magnitude=float(hp.get("lam_magnitude", 0.001)),
            device=DEVICE,
            variant=zentropy_variant,
        )

        # OOF prediction (invariant to regime permutation)
        oof_F[te_idx] = out_np["F"][te_idx]

        # Diagnostics: align regimes to reference p_ref if provided
        if "p" in out_np:
            p_fold = np.asarray(out_np["p"], dtype=float)
            if (p_ref is not None) and (p_ref.shape == p_fold.shape):
                score0 = swap_overlap_score(p_ref, p_fold, idx=tr_idx)
                perm = best_perm_overlap(p_ref, p_fold, idx=tr_idx)
                out_np2 = apply_perm_out(out_np, perm)
                p_use = out_np2["p"]
                score1 = swap_overlap_score(p_ref, p_use, idx=tr_idx)
                log(f"[SWAP] fold={fold} perm={perm} overlap(train): {score0:.4f} -> {score1:.4f}")

            else:
                p_use = p_fold

            oof_p[te_idx, :] = p_use[te_idx, :]

            # normalized entropy in [0,1]
            eps = 1e-12
            ent = -np.sum(p_use[te_idx, :] * np.log(p_use[te_idx, :] + eps), axis=1) / max(np.log(K + eps), eps)
            oof_ent[te_idx] = ent

    return oof_F, oof_p, oof_ent

# ============================================================
# In-sample fit for each model (full data)
# ============================================================
def fit_predict_insample_all_models(case: str, cfg: dict, hp_zen: dict, out_case_dir: str, data: dict, zentropy_variant: str = "v2_entropy_complete") -> Dict[str, np.ndarray]:
    """
    Trains ONLY Zentropy on full data (y_std) and returns predictions in y_std space.
    Keeps downstream CSV + figure pipeline intact, while removing non-Zentropy baselines.
    """
    coords = data["coords"]
    X_std = data["X_std"]
    y_std = data["y_std"]
    C_std = data["C_std"]

    # Zentropy KNN graph for in-sample training
    edges_zen = build_knn_edges(coords, k=int(hp_zen["graph_k"]))

    preds: Dict[str, np.ndarray] = {}

    idx_all = np.arange(len(y_std))
    tr2, va2 = idx_all, idx_all

    model_zen, out_np = safe_train_zentropy(
        X_std=X_std, C_std=C_std, y_std=y_std, edges_np=edges_zen,
        E_idx=data["E_idx"], S_idx=data["S_idx"], K=int(hp_zen["K"]),
        tr_idx=tr2, va_idx=va2,
        max_epochs=EPOCHS, patience=PATIENCE, lr=LR, wd=WD,
        lam_sparse=float(hp_zen.get("lam_sparse", 0.01)),
        lam_magnitude=float(hp_zen.get("lam_magnitude", 0.001)),
        device=DEVICE,
        variant=zentropy_variant,
    )

    # Canonicalize in-sample regime ordering for stable figures
    try:
        if isinstance(out_np, dict) and ("p" in out_np):
            perm_can = canonical_perm_from_p(out_np["p"], mode="mean_p_desc")
            out_np = apply_perm_out(out_np, perm_can)
    except Exception as _e:
        log(f"[WARN] in-sample canonicalization skipped: {_e}")

    preds[zkey] = out_np["F"].astype(np.float64)
    # regime probabilities stored for n_eff calculation in summary
    if "p" in out_np:
        preds[zkey + "_p"] = out_np["p"].astype(np.float64)

    # Save thermo diagnostics (global + local) and paper figures (A/B/C...)
    try:
        save_zentropy_thermo_diagnostics(case, out_case_dir, data, hp_zen, model_zen, edges_zen, out_np)
    except Exception as e:
        log(f"[WARN] thermo diagnostics skipped due to error: {e}")

    return preds

# ============================================================
# Zentropy thermo diagnostics (global importance + local sensitivity)
# ============================================================
# ============================================================
# Zentropy thermo diagnostics (global importance + local sensitivity)
# ============================================================
# ============================================================
# PAPER FIGURES (match your example: layout + colormaps + bigger points)
# - EPSG:5070 plotting coords
# - only make points bigger via MAP_S
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def _legend_handles_rowmajor_for_matplotlib(handles, labels, ncol: int):
    """
    Matplotlib fills legend entries column-first by default.
    Reorder handles and labels to display entries row-first.
    Preserve the original legend elements.
    """
    n = len(handles)
    ncol = int(max(1, ncol))
    nrow = int(np.ceil(n / ncol))

    grid_h = [[None] * ncol for _ in range(nrow)]
    grid_l = [[None] * ncol for _ in range(nrow)]

    it = iter(range(n))
    # desired display order: row-major fill
    for r in range(nrow):
        for c in range(ncol):
            try:
                i = next(it)
            except StopIteration:
                break
            grid_h[r][c] = handles[i]
            grid_l[r][c] = labels[i]

    # convert to the order legend uses internally (column-major)
    out_h, out_l = [], []
    for c in range(ncol):
        for r in range(nrow):
            if grid_h[r][c] is not None:
                out_h.append(grid_h[r][c])
                out_l.append(grid_l[r][c])

    return out_h, out_l
# ---- you asked: "point size only bigger" ----
MAP_S   = 20
MAP_ALP = 0.6
# Optional render-only resource limits used by the one-click reproducibility runner.
# Defaults preserve the original analysis notebook when this script is run directly.
SAVE_DPI = int(os.environ.get("ZEGNN_SAVE_DPI", "400"))
if "ZEGNN_CANVAS_DPI" in os.environ:
    plt.rcParams["figure.dpi"] = int(os.environ["ZEGNN_CANVAS_DPI"])

def _axis_map_clean(ax, keep_frame=False):
    ax.set_xticks([]); ax.set_yticks([])
    if keep_frame:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(1.0)
    else:
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax.set_aspect("equal", adjustable="box")

def _sym_lim(v, q=0.985, minv=1e-9):
    v = np.asarray(v, float)
    a = np.abs(v[np.isfinite(v)])
    if a.size == 0:
        return (-1, 1)
    m = float(np.quantile(a, q))
    m = max(m, minv)
    return (-m, m)

def _zscore(v):
    v = np.asarray(v, float)
    mu = np.nanmean(v); sd = np.nanstd(v)
    if not np.isfinite(sd) or sd < 1e-12:
        return v * 0.0
    return (v - mu) / sd

def _maybe_project_for_plot(coords_xy):
    # ensure EPSG:5070 if coords look lon/lat
    try:
        c2, _, _ = maybe_project_lonlat_to_5070(coords_xy.astype(np.float64))
        return c2.astype(np.float32)
    except Exception:
        return coords_xy.astype(np.float32)

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def fig_A_regime_entropy_FES(case, out_fig_dir, coords_xy, p, F_std, E_mix, S_mix, K,
                             fname_prefix="FIG_A_InSample_Decomposition"):
    coords = _maybe_project_for_plot(coords_xy)
    reg = np.argmax(p, axis=1).astype(int)
    ent = (-np.sum(p * np.log(p + 1e-12), axis=1)) / max(np.log(K + 1e-12), 1e-12)

    Fz = np.asarray(F_std, float)
    Ez = _zscore(E_mix)
    Sz = _zscore(S_mix)

    base_cols = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    reg_cols = base_cols[:K]
    reg_cmap = ListedColormap(reg_cols)
    ent_cmap = plt.cm.magma
    div_cmap = plt.cm.RdBu_r

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        height_ratios=[1.0, 0.18, 1.0],
        width_ratios=[1.0, 1.25],
        wspace=0.18, hspace=0.18
    )

    # --- Top row: Regimes | Entropy ---
    ax_reg = fig.add_subplot(gs[0, 0])
    ax_ent = fig.add_subplot(gs[0, 1])

    sc0 = ax_reg.scatter(coords[:, 0], coords[:, 1], c=reg, cmap=reg_cmap,
                         s=MAP_S, alpha=MAP_ALP, linewidths=0, rasterized=True)
    ax_reg.set_title("Identified Regimes", fontweight="bold")
    _axis_map_clean(ax_reg)

    sc1 = ax_ent.scatter(coords[:, 0], coords[:, 1], c=ent, cmap=ent_cmap,
                         vmin=0.0, vmax=1.0, s=MAP_S, alpha=MAP_ALP,
                         linewidths=0, rasterized=True)
    ax_ent.set_title("Gating Uncertainty (Entropy)", fontweight="bold")
    _axis_map_clean(ax_ent)


    div1 = make_axes_locatable(ax_ent)
    cax_ent = div1.append_axes("right", size="3.5%", pad=0.06)
    cb1 = fig.colorbar(sc1, cax=cax_ent)
    cb1.set_ticks([0, .2, .4, .6, .8, 1.0])

    # --- Middle legend row (under regime map) ---
    ax_leg = fig.add_subplot(gs[1, 0])
    ax_leg.axis("off")

    handles = [Patch(facecolor=reg_cols[k], edgecolor="none", label=f"Regime {k+1}") for k in range(K)]
    labels = [f"Regime {k+1}" for k in range(K)]


    ncol = min(3, K)
    h2, l2 = _legend_handles_rowmajor_for_matplotlib(handles, labels, ncol=ncol)

    ax_leg.legend(
        h2, l2, title="Regimes", ncol=ncol, frameon=False,
        loc="center", bbox_to_anchor=(0.5, 0.5)
    )

    # right middle cell is empty
    ax_blank = fig.add_subplot(gs[1, 1]); ax_blank.axis("off")

    # --- Bottom row: F/E/S ---
    gs_bot = gs[2, :].subgridspec(1, 3, wspace=0.22)

    ax_F = fig.add_subplot(gs_bot[0, 0])
    ax_E = fig.add_subplot(gs_bot[0, 1])
    ax_S = fig.add_subplot(gs_bot[0, 2])

    vminF, vmaxF = _sym_lim(Fz)
    vminE, vmaxE = _sym_lim(Ez)
    vminS, vmaxS = _sym_lim(Sz)

    scF = ax_F.scatter(coords[:, 0], coords[:, 1], c=Fz, cmap=div_cmap,
                       vmin=vminF, vmax=vmaxF, s=MAP_S, alpha=MAP_ALP,
                       linewidths=0, rasterized=True)
    ax_F.set_title(r"Free Energy $F(x)$", fontweight="bold")
    _axis_map_clean(ax_F)
    divF = make_axes_locatable(ax_F)
    caxF = divF.append_axes("right", size="3.5%", pad=0.06)
    fig.colorbar(scF, cax=caxF)

    scE = ax_E.scatter(coords[:, 0], coords[:, 1], c=Ez, cmap=div_cmap,
                       vmin=vminE, vmax=vmaxE, s=MAP_S, alpha=MAP_ALP,
                       linewidths=0, rasterized=True)
    ax_E.set_title(r"Burden $E(x)$", fontweight="bold")
    _axis_map_clean(ax_E)
    divE = make_axes_locatable(ax_E)
    caxE = divE.append_axes("right", size="3.5%", pad=0.06)
    fig.colorbar(scE, cax=caxE)

    scS = ax_S.scatter(coords[:, 0], coords[:, 1], c=Sz, cmap=div_cmap,
                       vmin=vminS, vmax=vmaxS, s=MAP_S, alpha=MAP_ALP,
                       linewidths=0, rasterized=True)
    ax_S.set_title(r"Capacity $S(x)$", fontweight="bold")
    _axis_map_clean(ax_S)
    divS = make_axes_locatable(ax_S)
    caxS = divS.append_axes("right", size="3.5%", pad=0.06)
    fig.colorbar(scS, cax=caxS)

    out_png = os.path.join(out_fig_dir, f"{fname_prefix}_{case}.png")
    out_pdf = os.path.join(out_fig_dir, f"{fname_prefix}_{case}.pdf")
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def fig_B_scatter_box(case, out_fig_dir, y_raw, F_raw, E_std_like, S_std_like, p, K,
                      fname_prefix="FIG_B_Obs_vs_FES"):
    reg = np.argmax(p, axis=1).astype(int)
    base_cols = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
    reg_cols = base_cols[:K]

    def _corr(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3:
            return np.nan
        return float(np.corrcoef(a[m], b[m])[0, 1])

    rF = _corr(y_raw, F_raw)
    rE = _corr(y_raw, E_std_like)
    rS = _corr(y_raw, S_std_like)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # --- top row: scatters ---
    ax = axes[0, 0]
    for k in range(K):
        m = (reg == k)
        ax.scatter(y_raw[m], F_raw[m], s=max(10, MAP_S*0.6), color=reg_cols[k],
                   alpha=0.75, linewidths=0, label=f"Regime {k+1}")
    ax.set_title(f"Observed vs Free Energy $F(x)$\n(r={rF:.2f})")
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted $F(x)$")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[0, 1]
    for k in range(K):
        m = (reg == k)
        ax.scatter(y_raw[m], E_std_like[m], s=max(10, MAP_S*0.6), color=reg_cols[k],
                   alpha=0.75, linewidths=0)
    ax.set_title(f"Observed vs Burden $E(x)$\n(r={rE:.2f})")
    ax.set_xlabel("Observed"); ax.set_ylabel(r"$E(x)$ (std)")

    ax = axes[0, 2]
    for k in range(K):
        m = (reg == k)
        ax.scatter(y_raw[m], S_std_like[m], s=max(10, MAP_S*0.6), color=reg_cols[k],
                   alpha=0.75, linewidths=0)
    ax.set_title(f"Observed vs Capacity $S(x)$\n(r={rS:.2f})")
    ax.set_xlabel("Observed"); ax.set_ylabel(r"$S(x)$ (std)")

    # --- bottom row: boxplots by regime ---
    def _box(ax, values, title, ylab):
        data = [np.asarray(values[reg == k], float) for k in range(K)]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for k, box in enumerate(bp["boxes"]):
            box.set_facecolor(reg_cols[k])
            box.set_alpha(0.6)
            box.set_edgecolor("black")
        for line_key in ["whiskers", "caps", "medians"]:
            for ln in bp[line_key]:
                ln.set_color("black")
                ln.set_linewidth(1.2)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xticklabels([f"R{k+1}" for k in range(K)])

    _box(axes[1, 0], y_raw, "Outcome y Distribution", "y")
    _box(axes[1, 1], E_std_like, r"Burden $E(x)$ Distribution", r"$E(x)$ (std)")
    _box(axes[1, 2], S_std_like, r"Capacity $S(x)$ Distribution", r"$S(x)$ (std)")

    out_png = os.path.join(out_fig_dir, f"{fname_prefix}_{case}.png")
    out_pdf = os.path.join(out_fig_dir, f"{fname_prefix}_{case}.pdf")
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def fig_C_prob_maps(case, out_fig_dir, coords_xy, p, K, fname_prefix="FIG_C_Regime_ProbMaps"):
    coords = _maybe_project_for_plot(coords_xy)
    ncols = 3
    nrows = int(np.ceil(K / ncols))
    fig = plt.figure(figsize=(18, 5.2 * nrows), constrained_layout=True)
    gs = fig.add_gridspec(nrows=nrows, ncols=2*ncols, width_ratios=[1, 0.05]*ncols, wspace=0.20, hspace=0.25)

    for k in range(K):
        r = k // ncols
        c = k % ncols
        ax = fig.add_subplot(gs[r, 2*c])
        cax = fig.add_subplot(gs[r, 2*c + 1])

        sc = ax.scatter(coords[:, 0], coords[:, 1], c=p[:, k], cmap="viridis",
                        vmin=0.0, vmax=1.0, s=MAP_S, alpha=MAP_ALP,
                        linewidths=0, rasterized=True)
        ax.set_title(f"P(Regime {k+1} | x)")
        _axis_map_clean(ax, keep_frame=True)
        fig.colorbar(sc, cax=cax)

    # blank cells
    for k in range(K, nrows*ncols):
        r = k // ncols
        c = k % ncols
        ax = fig.add_subplot(gs[r, 2*c]); ax.axis("off")
        ax = fig.add_subplot(gs[r, 2*c + 1]); ax.axis("off")

    out_png = os.path.join(out_fig_dir, f"{fname_prefix}_{case}.png")
    out_pdf = os.path.join(out_fig_dir, f"{fname_prefix}_{case}.pdf")
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import textwrap

def fig_D_mechanism_2x2(case, out_fig_dir, feat_cols, grads_F, grads_E, grads_S,
                        top_n=10, fname_prefix="FIG_D_ThermoMechanism_2x2",
                        # --- label control ---
                        wrap_width=22,
                        label_max_len=18,
                        annotate_max=8,
                        use_adjust_text=True,
                        # --- name map ---
                        name_map=None,
                        save_name_map_csv=True,
                        name_map_csv_prefix="TABLE_S_VariableNameMap"):
    """
    FIG D: (A) global importance barh, (B) phase space, (C) role heterogeneity, (D) violin
    Handling long feature names:
      - Map full names to shorter display labels.
      - Wrap y-axis labels.
      - Reduce overlap between annotations.
      - Export a table matching short and full labels.
    """

    # -----------------------
    # Default name maps (extend freely)
    # -----------------------
    DEFAULT_NAME_MAP = {

        "Mean sea level pressure": "MSLP",
        "2m temperature": "T2m",
        "2m dewpoint temperature": "D2m",
        "Boundary layer height": "BLH",
        "10m u-component of wind": "U10",
        "10m v-component of wind": "V10",
        "Total precipitation": "TP",
        "Surface solar radiation downwards": "SSRD",
        "Surface solar radiation downward": "SSRD",
        "Surface pressure": "SP",
        "Total cloud cover": "TCC",
        "Low cloud cover": "LCC",
        "Medium cloud cover": "MCC",
        "High cloud cover": "HCC",
        "10m wind speed": "WS10",
        "Wind speed": "WS",
        "Relative humidity": "RH",
        "Specific humidity": "q",
        "Evaporation": "E",
        "Geopotential": "Z",
        "Temperature": "T",

    }


    merged_map = dict(DEFAULT_NAME_MAP)
    if isinstance(name_map, dict):
        merged_map.update(name_map)

    feat_cols = list(feat_cols)
    gF = np.asarray(grads_F, float)
    gE = np.asarray(grads_E, float)
    gS = np.asarray(grads_S, float)

    os.makedirs(out_fig_dir, exist_ok=True)

    # -----------------------
    # helpers: alias / wrap / ellipsis
    # -----------------------
    def _alias(s: str) -> str:
        return merged_map.get(s, s)

    def _wrap(s: str) -> str:
        return textwrap.fill(s, width=int(wrap_width)) if wrap_width and wrap_width > 0 else s

    def _ellip(s: str, max_len: int) -> str:
        if max_len is None or max_len <= 0:
            return s
        return s if len(s) <= max_len else (s[:max_len - 1] + "…")

    # -----------------------
    # Global importances
    # -----------------------
    I_F = np.nanmedian(np.abs(gF), axis=0)
    I_E = np.nanmedian(np.abs(gE), axis=0)
    I_S = np.nanmedian(np.abs(gS), axis=0)
    D_E = I_E / (I_E + I_S + 1e-12)  # burden dominance in [0,1]

    top_n = min(int(top_n), len(feat_cols))
    idx = np.argsort(I_F)[::-1][:top_n]


    f_top_full = [feat_cols[i] for i in idx]
    f_top = [_alias(v) for v in f_top_full]
    f_top_y = [_wrap(v) for v in f_top]
    f_top_sc = [_ellip(v, int(label_max_len)) for v in f_top]

    IF = I_F[idx]; IE = I_E[idx]; IS = I_S[idx]; DE = D_E[idx]

    # -----------------------
    # Save name-map table for Supplementary (top variables only)
    # -----------------------
    if save_name_map_csv:
        df_map = pd.DataFrame({
            "case": [case] * len(f_top_full),
            "feature_original": f_top_full,
            "feature_display": f_top,
        })
        out_csv = os.path.join(out_fig_dir, f"{name_map_csv_prefix}_{case}_top{top_n}.csv")
        df_map.to_csv(out_csv, index=False)

    # -----------------------
    # Colors
    # -----------------------
    col_cap = "#4C72B0"
    col_bur = "#C44E52"
    col_mid = "#9E9E9E"

    bar_cols = []
    for d in DE:
        if d > 0.55:
            bar_cols.append(col_bur)
        elif d < 0.45:
            bar_cols.append(col_cap)
        else:
            bar_cols.append(col_mid)

    # -----------------------
    # Role reversal index (sign flip rate)
    # -----------------------
    rri = []
    for j in idx:
        v = gF[:, j]
        s0 = np.sign(np.nanmean(v))
        if s0 == 0:
            rri.append(0.5)
        else:
            rri.append(float(np.mean(np.sign(v) != s0)))
    rri = np.asarray(rri, float)

    # -----------------------
    # Robust symmetric x-lim for violin
    # -----------------------
    all_v = np.concatenate([np.asarray(gF[:, j], float) for j in idx], axis=0)
    all_v = all_v[np.isfinite(all_v)]
    if all_v.size == 0:
        xlim = 1.0
    else:
        xlim = float(np.quantile(np.abs(all_v), 0.99))
        xlim = max(xlim, 1e-6)

    # -----------------------
    # Figure layout (left margin auto)
    # -----------------------
    max_line_len = 0
    for s in f_top_y:
        for line in s.split("\n"):
            max_line_len = max(max_line_len, len(line))
    left = min(0.38, max(0.18, 0.16 + 0.006 * max_line_len))

    fig, axes = plt.subplots(2, 2, figsize=(19, 11))
    fig.subplots_adjust(left=left, right=0.98, bottom=0.08, top=0.93, wspace=0.22, hspace=0.28)

    # (A) Global Feature Importance
    ax = axes[0, 0]
    y = np.arange(len(f_top_y))
    ax.barh(y, IF, color=bar_cols, alpha=0.85, edgecolor="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(f_top_y)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.invert_yaxis()
    ax.set_title("Global Feature Importance", fontweight="bold")
    ax.set_xlabel(r"Net Thermodynamic Importance ($I_F$)" "\n" r"(Median $|\partial F/\partial x|$)")
    ax.grid(True, axis="x", alpha=0.30, linestyle="--")

    ax.legend(handles=[
        Patch(facecolor=col_cap, edgecolor="none", label="Capacity-Dominant"),
        Patch(facecolor=col_bur, edgecolor="none", label="Burden-Dominant"),
        Patch(facecolor=col_mid, edgecolor="none", label="Mixed"),
    ], frameon=False, loc="lower right")

    # (B) Phase space
    ax = axes[0, 1]
    s = 1100 * (IF / (IF.max() + 1e-12)) + 120
    sc = ax.scatter(IE, IS, s=s, c=DE, cmap="coolwarm", vmin=0, vmax=1,
                    edgecolor="black", linewidths=0.7, alpha=0.90)

    lim = max(float(np.max(IE)), float(np.max(IS))) * 1.15 + 1e-12
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.margins(x=0.12, y=0.12)

    ax.plot([0, lim], [0, lim], ls="--", lw=1.2, color="gray", alpha=0.7)
    ax.set_title("Thermodynamic Mechanism Phase Space", fontweight="bold")
    ax.set_xlabel(r"Burden Contribution")
    ax.set_ylabel(r"Capacity Contribution")
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"Burden Dominance")

    # annotate with repel if possible
    texts = []
    k_anno = min(int(annotate_max), len(f_top_sc))
    for t in range(k_anno):
        texts.append(
            ax.text(IE[t], IS[t], f_top_sc[t],
                    fontsize=10, ha="left", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.70, pad=0.20))
        )

    if use_adjust_text:
        try:
            from adjustText import adjust_text
            adjust_text(
                texts, ax=ax,
                expand_points=(1.3, 1.6), expand_text=(1.2, 1.4),
                arrowprops=dict(arrowstyle="-", lw=0.6, color="gray", alpha=0.8)
            )
        except Exception:
            offsets = [(10, 6), (10, -10), (-10, 8), (-10, -8), (14, 0), (-14, 0), (0, 14), (0, -14)]
            for i in range(k_anno):
                dx, dy = offsets[i % len(offsets)]
                texts[i].remove()
                ax.annotate(
                    f_top_sc[i], (IE[i], IS[i]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=10, ha="left", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.70, pad=0.20),
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="gray", alpha=0.8),
                    annotation_clip=True
                )

    # (C) Spatial Role Heterogeneity
    ax = axes[1, 0]
    ax.axvspan(0.0, 0.2, color="#CFE9CF", alpha=0.35)
    ax.axvspan(0.5, 1.0, color="#F3D7B6", alpha=0.35)
    ax.scatter(rri, IF, s=120, color="gray", edgecolor="black", alpha=0.80)
    ax.set_title("Spatial Role Heterogeneity", fontweight="bold")
    ax.set_xlabel("Role Reversal Index\n(0=Consistent, 1=Spatial Sign Flip)")
    ax.set_ylabel(r"Net Importance ($I_F$)")
    ax.tick_params(axis="both", labelsize=11)

    ytop = ax.get_ylim()[1]
    ax.text(0.02, ytop * 0.95, "Spatially Consistent", color="green", fontweight="bold")
    ax.text(0.62, ytop * 0.95, "Regime-dependent", color="orange", fontweight="bold")

    texts2 = []
    for t in range(len(f_top_sc)):
        texts2.append(
            ax.text(rri[t], IF[t], f_top_sc[t],
                    fontsize=9, ha="left", va="center", alpha=0.95,
                    bbox=dict(fc="white", ec="none", alpha=0.65, pad=0.15))
        )
    if use_adjust_text:
        try:
            from adjustText import adjust_text
            adjust_text(
                texts2, ax=ax,
                expand_points=(1.2, 1.4), expand_text=(1.1, 1.25),
                arrowprops=dict(arrowstyle="-", lw=0.5, color="gray", alpha=0.8)
            )
        except Exception:
            pass

    # (D) Local Sensitivity Distributions (violin)
    ax = axes[1, 1]
    data = [np.asarray(gF[:, j], float) for j in idx]
    pos = np.arange(1, len(data) + 1)

    vp = ax.violinplot(data, positions=pos, vert=False, showmedians=True, showextrema=False)

    for i, body in enumerate(vp["bodies"]):
        body.set_edgecolor("black")
        body.set_linewidth(0.9)
        body.set_alpha(0.78)
        body.set_facecolor(bar_cols[i])

    vp["cmedians"].set_color("white")
    vp["cmedians"].set_linewidth(2.2)

    ax.axvline(0.0, color="gray", lw=1.2)
    ax.set_yticks(pos)
    ax.set_yticklabels(f_top_y)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=11)
    ax.invert_yaxis()
    ax.set_xlim(-xlim, xlim)

    ax.set_title("Local Sensitivity Distributions", fontweight="bold")
    ax.set_xlabel(r"Local Sensitivity Distribution ($\partial F/\partial x$)"
                  "\n(Positive=Increases Risk, Negative=Decreases Risk)")
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")

    out_png = os.path.join(out_fig_dir, f"{fname_prefix}_{case}.png")
    out_pdf = os.path.join(out_fig_dir, f"{fname_prefix}_{case}.pdf")
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


from mpl_toolkits.axes_grid1 import make_axes_locatable

def fig_E_derivative_maps(case, out_fig_dir, coords_xy, feat_cols, grads_F, grads_E, grads_S,
                          features_for_maps=None, top_k=None,
                          fname_prefix="FIG_E_PartialDerivatives_Maps"):
    coords = _maybe_project_for_plot(coords_xy)
    feat_cols = list(feat_cols)
    gF = np.asarray(grads_F, float)
    gE = np.asarray(grads_E, float)
    gS = np.asarray(grads_S, float)

    I_F = np.nanmedian(np.abs(gF), axis=0)

    if features_for_maps is None:
        if top_k is None:
            feats = feat_cols[:]          # ALL
        else:
            idx = np.argsort(I_F)[::-1][:min(int(top_k), len(feat_cols))]
            feats = [feat_cols[i] for i in idx]
    else:
        feats = [f for f in features_for_maps if f in feat_cols]
        if len(feats) == 0:
            feats = feat_cols[:] if top_k is None else [feat_cols[i] for i in np.argsort(I_F)[::-1][:min(int(top_k), len(feat_cols))]]

    nrows = len(feats)
    fig = plt.figure(figsize=(18, 4.6 * nrows))
    gs = fig.add_gridspec(nrows=nrows, ncols=3, wspace=0.22, hspace=0.28)

    div_cmap = plt.cm.RdBu_r

    for r, feat in enumerate(feats):
        j = feat_cols.index(feat)
        vF = gF[:, j]; vE = gE[:, j]; vS = gS[:, j]

        limF = _sym_lim(vF); limE = _sym_lim(vE); limS = _sym_lim(vS)

        # --- dF/dx ---
        ax1 = fig.add_subplot(gs[r, 0])
        sc1 = ax1.scatter(coords[:,0], coords[:,1], c=vF, cmap=div_cmap,
                          vmin=limF[0], vmax=limF[1], s=MAP_S, alpha=MAP_ALP,
                          linewidths=0, rasterized=True)
        ax1.set_title(rf"$\partial F/\partial x$" + f"\n{feat}", fontweight="bold")
        _axis_map_clean(ax1)
        div1 = make_axes_locatable(ax1)
        cax1 = div1.append_axes("right", size="3.5%", pad=0.06)
        fig.colorbar(sc1, cax=cax1)

        # --- dE/dx ---
        ax2 = fig.add_subplot(gs[r, 1])
        sc2 = ax2.scatter(coords[:,0], coords[:,1], c=vE, cmap=div_cmap,
                          vmin=limE[0], vmax=limE[1], s=MAP_S, alpha=MAP_ALP,
                          linewidths=0, rasterized=True)
        ax2.set_title(rf"$\partial E/\partial x$" + f"\n{feat}", fontweight="bold")
        _axis_map_clean(ax2)
        div2 = make_axes_locatable(ax2)
        cax2 = div2.append_axes("right", size="3.5%", pad=0.06)
        fig.colorbar(sc2, cax=cax2)

        # --- dS/dx ---
        ax3 = fig.add_subplot(gs[r, 2])
        sc3 = ax3.scatter(coords[:,0], coords[:,1], c=vS, cmap=div_cmap,
                          vmin=limS[0], vmax=limS[1], s=MAP_S, alpha=MAP_ALP,
                          linewidths=0, rasterized=True)
        ax3.set_title(rf"$\partial S/\partial x$" + f"\n{feat}", fontweight="bold")
        _axis_map_clean(ax3)
        div3 = make_axes_locatable(ax3)
        cax3 = div3.append_axes("right", size="3.5%", pad=0.06)
        fig.colorbar(sc3, cax=cax3)

    out_png = os.path.join(out_fig_dir, f"{fname_prefix}_{case}.png")
    out_pdf = os.path.join(out_fig_dir, f"{fname_prefix}_{case}.pdf")
    fig.savefig(out_png, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def _compute_zentropy_grads_safe(model_zen, X_std, C_std, edges_np, device):
    """
    Compute grads for F/E/S w.r.t X.
    If device fails (MPS/cuda memory), fallback to CPU.
    """
    def _run(dev):
        model_zen_ = model_zen.to(dev)
        model_zen_.eval()
        X_t = torch.tensor(X_std, device=dev, requires_grad=True)
        C_t = torch.tensor(C_std, device=dev)
        edges_t = edges_to_device(edges_np, dev)

        out = model_zen_(X_t, C_t, edges_t)
        gF = torch.autograd.grad(out["F"].sum(), X_t, retain_graph=True, create_graph=False)[0]
        gE = torch.autograd.grad(out["E"].sum(), X_t, retain_graph=True, create_graph=False)[0]
        gS = torch.autograd.grad(out["S"].sum(), X_t, retain_graph=False, create_graph=False)[0]
        return (out, gF.detach().cpu().numpy(), gE.detach().cpu().numpy(), gS.detach().cpu().numpy())

    try:
        out, gF, gE, gS = _run(device)
        out_np = {k: (v.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(v) else v) for k, v in out.items()}
        return out_np, gF, gE, gS
    except RuntimeError as e:
        log(f"[WARN] gradient on {device} failed -> CPU fallback ({type(e).__name__}: {e})")
        out, gF, gE, gS = _run(torch.device("cpu"))
        out_np = {k: (v.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(v) else v) for k, v in out.items()}
        return out_np, gF, gE, gS

# ============================================================
# REPLACE your save_zentropy_thermo_diagnostics with this
# ============================================================
def save_zentropy_thermo_diagnostics(case: str, out_case_dir: str, data: dict, hp_zen: dict,
                                    model_zen: nn.Module, edges_np: np.ndarray, out_np: dict):
    out_csv_dir = os.path.join(out_case_dir, "CSV")
    out_fig_dir = os.path.join(out_case_dir, "FIGURES")
    ensure_dir(out_csv_dir); ensure_dir(out_fig_dir)

    X_std = data["X_std"].astype(np.float32)
    C_std = data["C_std"].astype(np.float32)
    feat_cols = data["feat_cols"]
    coords = data["coords"]
    K = int(hp_zen["K"])

    # ---- recompute out + gradients (F/E/S) safely ----
    out_np2, gF, gE, gS = _compute_zentropy_grads_safe(model_zen, X_std, C_std, edges_np, DEVICE)

    # keep using p/E/S/F from the gradient-consistent forward pass
    p = out_np2["p"].astype(np.float64)
    F_std = out_np2["F"].astype(np.float64)
    E_mix = out_np2["E"].astype(np.float64)
    S_mix = out_np2["S"].astype(np.float64)

    # ---- save global importance CSV (median abs dF/dx, like your example label) ----
    I_F = np.nanmedian(np.abs(gF), axis=0)
    I_E = np.nanmedian(np.abs(gE), axis=0)
    I_S = np.nanmedian(np.abs(gS), axis=0)
    D_E = I_E / (I_E + I_S + 1e-12)

    imp_df = pd.DataFrame({
        "feature": feat_cols,
        "I_F_median_abs_dFdx": I_F,
        "I_E_median_abs_dEdx": I_E,
        "I_S_median_abs_dSdx": I_S,
        "D_E_burden_dominance": D_E
    }).sort_values("I_F_median_abs_dFdx", ascending=False).reset_index(drop=True)

    imp_df.to_csv(os.path.join(out_csv_dir, f"Zentropy_Thermo_GlobalImportance_{case}.csv"), index=False)

    # ---- save local sensitivity summary CSV (top 10 by default) ----
    topn = min(10, len(feat_cols))
    top_feats = imp_df["feature"].iloc[:topn].tolist()
    rows = []
    for f in top_feats:
        j = feat_cols.index(f)
        v = gF[:, j].astype(float)
        rows.append({
            "feature": f,
            "mean": float(np.nanmean(v)),
            "mean_abs": float(np.nanmean(np.abs(v))),
            "median": float(np.nanmedian(v)),
            "p10": float(np.nanpercentile(v, 10)),
            "p50": float(np.nanpercentile(v, 50)),
            "p90": float(np.nanpercentile(v, 90)),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_csv_dir, f"Zentropy_Thermo_LocalSensitivity_Summary_{case}.csv"), index=False)

    # ---- (optional) dump npz so you can replot without retraining ----
    np.savez_compressed(
        os.path.join(out_csv_dir, f"Zentropy_ThermoDump_{case}.npz"),
        coords=_maybe_project_for_plot(coords),
        p=p, F_std=F_std, E_mix=E_mix, S_mix=S_mix,
        grads_F=gF, grads_E=gE, grads_S=gS,
        feat_cols=np.array(feat_cols, dtype=object),
        y_raw=data["y_raw"].astype(np.float64)
    )

    # ---- FIGURES (match your examples) ----
    # A: regimes + entropy + F/E/S maps
    fig_A_regime_entropy_FES(case, out_fig_dir, coords, p, F_std, E_mix, S_mix, K)

    # B: observed vs F/E/S + boxplots
    F_raw = inv_to_original_space(F_std, data["y_scaler"], data["ytf"])
    # Use std-like E/S for scatter labels (consistent w/ your example)
    E_std_like = _zscore(E_mix)
    S_std_like = _zscore(S_mix)
    fig_B_scatter_box(case, out_fig_dir, data["y_raw"].astype(np.float64), F_raw, E_std_like, S_std_like, p, K)

    # C: regime probabilities
    fig_C_prob_maps(case, out_fig_dir, coords, p, K)

    # D: thermo mechanism 2x2
    fig_D_mechanism_2x2(case, out_fig_dir, feat_cols, gF, gE, gS, top_n=10)

    # E: partial derivative maps (top 3 features; you can pass a custom list if you want)
    fig_E_derivative_maps(case, out_fig_dir, coords, feat_cols, gF, gE, gS, features_for_maps=None, top_k=None)





# ============================================================
# Figures: in-sample map panel + residual Moran bar
# ============================================================
def plot_insample_maps(case: str, out_case_dir: str, coords: np.ndarray, y_std: np.ndarray, preds_std: Dict[str, np.ndarray]):
    out_fig_dir = os.path.join(out_case_dir, "FIGURES")
    ensure_dir(out_fig_dir)

    models_to_show = [zkey]
    models_to_show = [m for m in models_to_show if m in preds_std]

    ncols = len(models_to_show) + 2  # y + resid of zentropy
    fig, axes = plt.subplots(1, ncols, figsize=(4.2*ncols, 4.2), constrained_layout=True)

    # Observed y_std
    ax = axes[0]
    sc = ax.scatter(coords[:,0], coords[:,1], c=y_std, s=6)
    ax.set_title("Observed y (std)")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    # predictions
    for i, m in enumerate(models_to_show, 1):
        ax = axes[i]
        sc = ax.scatter(coords[:,0], coords[:,1], c=preds_std[m], s=6)
        ax.set_title(f"{m} pred (std)")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    # Zentropy residual
    ax = axes[-1]
    if zkey in preds_std:
        resid = y_std - preds_std[zkey]
        sc = ax.scatter(coords[:,0], coords[:,1], c=resid, s=6)
        ax.set_title("Zentropy resid (std)")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.axis("off")

    out_png = os.path.join(out_fig_dir, f"InSample_Maps_{case}.png")
    out_pdf = os.path.join(out_fig_dir, f"InSample_Maps_{case}.pdf")
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def plot_residual_moran_bar(case: str, out_case_dir: str, y_std: np.ndarray, preds_std: Dict[str, np.ndarray], coords: np.ndarray):
    out_fig_dir = os.path.join(out_case_dir, "FIGURES")
    ensure_dir(out_fig_dir)

    edges_m = build_knn_edges(coords, k=MORAN_KNN_K)
    rows = []
    for m, pred in preds_std.items():
        pred = np.asarray(pred)
        if pred.ndim != 1:   # skip p-matrices (e.g. zkey+"_p")
            continue
        resid = y_std - pred
        I = global_moran_I_from_edges(resid, edges_m)
        rows.append((m, float(I)))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)

    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar(names, vals)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_title("In-sample residual Moran's I (KNN k=10)")
    ax.set_ylabel("Moran's I")
    ax.set_xticklabels(names, rotation=30, ha="right")

    out_png = os.path.join(out_fig_dir, f"Residual_Moran_Bar_{case}.png")
    out_pdf = os.path.join(out_fig_dir, f"Residual_Moran_Bar_{case}.pdf")
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ============================================================
# Main runner per case
# ============================================================
def run_case(case: str, cfg: dict, hp_zen: dict, out_root: str, zentropy_variant: str = "v2_entropy_complete") -> pd.DataFrame:
    log(f"\n==================== {case} ====================")
    # --- case-specific map point size (kept) ---
    global MAP_S, MAP_ALP
    if case.startswith("PM"):
        MAP_S = 40
        MAP_ALP = 0.85
    elif case == "Housing":
        MAP_S = 14.0
        MAP_ALP = 0.85
    elif case == "MentalHealth":
        MAP_S = 16
        MAP_ALP = 0.85
    else:
        MAP_S = 20
        MAP_ALP = 0.6
    log(f"[PLOT CONFIG] MAP_S={MAP_S}, MAP_ALP={MAP_ALP}")

    log(f"[DEVICE] {DEVICE} | mps_available={torch.backends.mps.is_available()} | cuda_available={torch.cuda.is_available()}")

    df = pd.read_csv(resolve_path(cfg["csv_path"]))
    data = preprocess_case(df, case, cfg)

    coords = data["coords"]
    X_std = data["X_std"].astype(np.float32)
    y_std = data["y_std"].astype(np.float32)
    C_std = data["C_std"].astype(np.float32)

    v = (zentropy_variant or "").strip().lower()
    if v in ("v2_entropy_complete", "v2entropycomplete", "entropy_complete", "entropyinf", "copilot17_19", "copilot"):
        suffix = "_V2EntropyComplete"
    else:
        suffix = ""
    out_case_dir = os.path.join(out_root, f"{case}{suffix}")
    out_csv_dir = os.path.join(out_case_dir, "CSV")
    out_fig_dir = os.path.join(out_case_dir, "FIGURES")
    ensure_dir(out_csv_dir); ensure_dir(out_fig_dir)

    # Save run config (Zentropy only)
    with open(os.path.join(out_csv_dir, f"RunConfig_{case}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "case": case,
            "cfg": cfg,
            "zentropy_hparams": hp_zen,
            "zentropy_variant": zentropy_variant,
            "device": str(DEVICE),
            "seed": SEED,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "lr": LR,
            "wd": WD
        }, f, indent=2, ensure_ascii=False)

    # CV splits
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    random_splits = list(kf.split(X_std))

    groups = make_spatial_groups(coords, grid_n=SPATIAL_GRID_N)
    gkf = GroupKFold(n_splits=N_SPLITS)
    spatial_splits = list(gkf.split(X_std, y_std, groups=groups))

    # ------------------------------------------------------------
    # Step 1) SpatialCV-based HP selection (Zentropy only)
    # ------------------------------------------------------------
    hp_zen_sel, sel_rows, sel_best = select_hp_spatialcv(
        case=case,
        X_std=X_std, C_std=C_std, y_std=y_std,
        coords=coords,
        data=data,
        hp_zen=hp_zen,
        spatial_splits=spatial_splits,
        zentropy_variant=zentropy_variant,
    )
    hp_zen = hp_zen_sel

    # Save selection table
    try:
        if sel_rows:
            pd.DataFrame(sel_rows).to_csv(os.path.join(out_csv_dir, f"HPSelect_SpatialCV_{case}.csv"), index=False)
    except Exception as _e:
        log(f"[WARN] failed to save selection table: {_e}")

    # Build Zentropy KNN graph using selected graph_k
    edges_zen = build_knn_edges(coords, k=int(hp_zen["graph_k"]))

    # ------------------------------------------------------------
    # Reference Zentropy fit for regime label alignment (swap)
    # ------------------------------------------------------------
    p_ref = None
    try:
        idx_all = np.arange(len(y_std))
        _mref, _oref = safe_train_zentropy(
            X_std=X_std, C_std=C_std, y_std=y_std, edges_np=edges_zen,
            E_idx=data["E_idx"], S_idx=data["S_idx"], K=int(hp_zen["K"]),
            tr_idx=idx_all, va_idx=idx_all,
            max_epochs=EPOCHS, patience=PATIENCE, lr=LR, wd=WD,
            lam_sparse=float(hp_zen.get("lam_sparse", 0.01)),
            lam_magnitude=float(hp_zen.get("lam_magnitude", 0.001)),
            device=DEVICE,
            variant=zentropy_variant,
        )
        if isinstance(_oref, dict) and ("p" in _oref):
            p_ref = np.asarray(_oref["p"], dtype=float)
            perm_ref = canonical_perm_from_p(p_ref, mode="mean_p_desc")
            p_ref = apply_perm_out({"p": p_ref}, perm_ref)["p"]
        else:
            p_ref = None
    except Exception as _e:
        log(f"[WARN] reference fit for swap failed -> proceed without swap ({type(_e).__name__}: {_e})")
        p_ref = None

    # ----------------------------
    # Random CV: OOF predictions (Zentropy only)
    # ----------------------------
    log("[CV] Random 5-fold (Zentropy only) ...")
    oof_random: Dict[str, np.ndarray] = {}
    oof_random[zkey], oof_random_p, oof_random_ent = run_cv_zentropy(
        X_std, C_std, y_std, edges_zen,
        data["E_idx"], data["S_idx"],
        hp_zen, random_splits,
        zentropy_variant=zentropy_variant,
        p_ref=p_ref
    )

    df_oof_r = pd.DataFrame({"y_std": y_std.astype(np.float64)})
    df_oof_r["y_raw"] = data["y_raw"].astype(np.float64)
    pred_std = oof_random[zkey]
    df_oof_r[f"{zkey}_pred_std"] = pred_std
    df_oof_r[f"{zkey}_pred_raw"] = inv_to_original_space(pred_std, data["y_scaler"], data["ytf"])
    df_oof_r.to_csv(os.path.join(out_csv_dir, f"OOF_Predictions_{case}_RandomCV.csv"), index=False)

    try:
        df_p = pd.DataFrame(oof_random_p, columns=[f"{zkey}_pR{k+1}" for k in range(oof_random_p.shape[1])])
        df_p[f"{zkey}_entropy"] = oof_random_ent
        df_p.to_csv(os.path.join(out_csv_dir, f"OOF_RegimeProbs_{case}_RandomCV.csv"), index=False)
    except Exception as _e:
        log(f"[WARN] failed to save RandomCV regime probs: {_e}")

    # ----------------------------
    # Spatial CV: OOF predictions (Zentropy only)
    # ----------------------------
    log("[CV] Spatial 5-fold (Zentropy only) ...")
    oof_spatial: Dict[str, np.ndarray] = {}
    reuse_best_spatial = (
        isinstance(sel_best, dict)
        and sel_best.get("oof_F") is not None
        and int(sel_best.get("splits_n", -1)) == int(N_SPLITS)
        and len(np.asarray(sel_best.get("oof_F"))) == len(y_std)
    )
    if reuse_best_spatial:
        log(f"[CV] Reusing best SpatialCV OOF from HP selection (r2_oof={float(sel_best.get('r2_oof', float('nan'))):.4f})")
        oof_spatial[zkey] = np.asarray(sel_best["oof_F"], dtype=float).copy()
        oof_spatial_p = None if sel_best.get("oof_p") is None else np.asarray(sel_best["oof_p"], dtype=float).copy()
        oof_spatial_ent = None if sel_best.get("oof_ent") is None else np.asarray(sel_best["oof_ent"], dtype=float).copy()
    else:
        oof_spatial[zkey], oof_spatial_p, oof_spatial_ent = run_cv_zentropy(
            X_std, C_std, y_std, edges_zen,
            data["E_idx"], data["S_idx"],
            hp_zen, spatial_splits,
            zentropy_variant=zentropy_variant,
            p_ref=p_ref
        )

    df_oof_s = pd.DataFrame({"y_std": y_std.astype(np.float64)})
    df_oof_s["y_raw"] = data["y_raw"].astype(np.float64)
    pred_std2 = oof_spatial[zkey]
    df_oof_s[f"{zkey}_pred_std"] = pred_std2
    df_oof_s[f"{zkey}_pred_raw"] = inv_to_original_space(pred_std2, data["y_scaler"], data["ytf"])
    df_oof_s.to_csv(os.path.join(out_csv_dir, f"OOF_Predictions_{case}_SpatialCV.csv"), index=False)

    try:
        if oof_spatial_p is not None:
            df_p2 = pd.DataFrame(oof_spatial_p, columns=[f"{zkey}_pR{k+1}" for k in range(oof_spatial_p.shape[1])])
            df_p2[f"{zkey}_entropy"] = oof_spatial_ent
            df_p2.to_csv(os.path.join(out_csv_dir, f"OOF_RegimeProbs_{case}_SpatialCV.csv"), index=False)
    except Exception as _e:
        log(f"[WARN] failed to save SpatialCV regime probs: {_e}")

    # ----------------------------
    # In-sample fit and figures (Zentropy only)
    # ----------------------------
    log("[FIT] In-sample (full data) — Zentropy only ...")
    preds_in = fit_predict_insample_all_models(case, cfg, hp_zen, out_case_dir, data, zentropy_variant=zentropy_variant)

    df_in = pd.DataFrame({"y_std": y_std.astype(np.float64)})
    df_in["y_raw"] = data["y_raw"].astype(np.float64)
    pred_in = preds_in[zkey]
    df_in[f"{zkey}_pred_std"] = pred_in
    df_in[f"{zkey}_pred_raw"] = inv_to_original_space(pred_in, data["y_scaler"], data["ytf"])
    df_in.to_csv(os.path.join(out_csv_dir, f"InSample_Predictions_{case}.csv"), index=False)

    plot_insample_maps(case, out_case_dir, coords, y_std, preds_in)
    plot_residual_moran_bar(case, out_case_dir, y_std, preds_in, coords)

    # ----------------------------
    # Summary metrics table (Random/Spatial/Insample) — Zentropy only
    # ----------------------------
    edges_moran = build_knn_edges(coords, k=MORAN_KNN_K)
    summary_rows = []

    metrics_random = regime_usage_metrics(oof_random_p)
    metrics_spatial = regime_usage_metrics(oof_spatial_p)
    metrics_insample = regime_usage_metrics(preds_in.get(zkey + "_p", None))

    log(f"[N_EFF] {case}: InSample n_eff_global={metrics_insample['n_eff_global']:.3f}, "
        f"n_eff_local={metrics_insample['n_eff_local']:.3f}, "
        f"argmax_share_max={metrics_insample['argmax_share_max']:.3f}")

    for cv_name, oof_dict, reg_metrics in [("RandomCV", oof_random, metrics_random), ("SpatialCV", oof_spatial, metrics_spatial)]:
        pred_std_cv = oof_dict[zkey]
        met_std = compute_metrics(y_std.astype(float), pred_std_cv.astype(float))
        y_pred_raw = inv_to_original_space(pred_std_cv, data["y_scaler"], data["ytf"])
        met_raw = compute_metrics(data["y_raw"].astype(float), y_pred_raw)
        summary_rows.append({
            "case": case, "model": zkey, "eval": cv_name,
            "RMSE_y_std": met_std["RMSE"], "R2_y_std": met_std["R2"],
            "RMSE_y_raw": met_raw["RMSE"], "R2_y_raw": met_raw["R2"],
            "MoranI_resid_in": np.nan,
            "K_upper": int(K_UPPER), "alpha_dir": float(ALPHA_DIR),
            "n_eff": reg_metrics["n_eff_global"],
            "n_eff_global": reg_metrics["n_eff_global"],
            "n_eff_local": reg_metrics["n_eff_local"],
            "argmax_share_max": reg_metrics["argmax_share_max"],
        })

    # In-sample metrics + Moran
    pred_std_in = preds_in[zkey]
    met_std = compute_metrics(y_std.astype(float), pred_std_in.astype(float))
    y_pred_raw = inv_to_original_space(pred_std_in, data["y_scaler"], data["ytf"])
    met_raw = compute_metrics(data["y_raw"].astype(float), y_pred_raw)
    resid = (y_std.astype(float) - pred_std_in.astype(float))
    I = global_moran_I_from_edges(resid, edges_moran)
    summary_rows.append({
        "case": case, "model": zkey, "eval": "InSample",
        "RMSE_y_std": met_std["RMSE"], "R2_y_std": met_std["R2"],
        "RMSE_y_raw": met_raw["RMSE"], "R2_y_raw": met_raw["R2"],
        "MoranI_resid_in": float(I),
        "K_upper": int(K_UPPER), "alpha_dir": float(ALPHA_DIR),
        "n_eff": metrics_insample["n_eff_global"],
        "n_eff_global": metrics_insample["n_eff_global"],
        "n_eff_local": metrics_insample["n_eff_local"],
        "argmax_share_max": metrics_insample["argmax_share_max"],
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(out_csv_dir, f"CV_Summary_{case}.csv"), index=False)
    log(f"[SAVE] CV summary -> {os.path.join(out_csv_dir, f'CV_Summary_{case}.csv')}")
    return summary_df

# ============================================================
# Main
# ============================================================
def main(manifest_path: str = DEFAULT_MANIFEST_PATH, out_root: Optional[str] = DEFAULT_OUT_ROOT, zentropy_variant: str = "v2_entropy_complete"):
    manifest_path = resolve_path(manifest_path)
    dfm = read_manifest(manifest_path)

    # auto out root
    if out_root is None:
        base = os.path.dirname(manifest_path)
        out_root = os.path.join(base, "CELL2_ZENTROPY_ONLY")
    ensure_dir(out_root)

    # Force the entropy-adjusted model-only variant
    zentropy_variant = "v2_entropy_complete"

    global zkey
    zkey = zentropy_model_name(zentropy_variant)

    log("============================================================")
    log("ZeGNN empirical evaluation | spatial CV, random CV, and in-sample assessment")
    log("============================================================")
    log(f"[MANIFEST] {manifest_path}")
    log(f"[OUT_ROOT]  {out_root}")
    log(f"[DEVICE]    {DEVICE} | mps_available={torch.backends.mps.is_available()} | cuda_available={torch.cuda.is_available()}")
    log(f"[VARIANT]   {zentropy_variant} -> model_key={zkey}")

    all_rows = []
    for case, cfg in CASES.items():
        hp = get_case_hparams_from_manifest(dfm, case)
        all_rows.append(run_case(case, cfg, hp, out_root, zentropy_variant=zentropy_variant))

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(os.path.join(out_root, "ALL_CASES_CV_SUMMARY.csv"), index=False)
    log(f"[SAVE] ALL_CASES_CV_SUMMARY.csv -> {os.path.join(out_root, 'ALL_CASES_CV_SUMMARY.csv')}")
    log("✅ DONE.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST_PATH, help="Path to ZENTROPY_BEST_HYPERPARAMS_MANIFEST.csv")
    ap.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT, help="Output root folder")
    ap.add_argument("--zentropy_variant", type=str, default="v2_entropy_complete", help="Model configuration identifier (fixed by this implementation).")
    args = ap.parse_known_args()[0]
    main(args.manifest, args.out_root, args.zentropy_variant)


