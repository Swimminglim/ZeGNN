# Source extracted without changing numerical model logic from Global_Linear_simulation.ipynb
# Execute from repository root; see README.md.

import os, json, math, time, warnings, itertools
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

import statsmodels.api as sm
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error, silhouette_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

warnings.filterwarnings("ignore")

# Optional Moran
try:
    from libpysal.weights import KNN as _KNN_W
    from esda.moran import Moran as _Moran
    _HAVE_MORAN = True
except Exception:
    _HAVE_MORAN = False


# ==============================================================================
# 1) CONFIG
# ==============================================================================
@dataclass
class CFG:
    # I/O
    out_dir: str = "./GLOBAL_Linear"
    seed: int = 2024

    # Simulation
    grid_nx: int = 50
    grid_ny: int = 50
    p: int = 5
    k_true: int = 1

    # DGP physics
    diffusion_k: int = 10
    jitter: float = 0.02
    iid_noise: float = 0.12
    unobs_spatial_strength: float = 0.0
    diffusion_rho1: float = 0.0
    fd_eps: float = 0.01

    # Zentropy structure

    E_idx: tuple = (0, 1, 2, 3, 4)
    S_idx: tuple = ()

    # Torch hyperparams
    hidden: int = 64
    lr: float = 0.005
    epochs_final: int = 800
    patience: int = 60

    # Sweep grid (Zentropy ONLY)
    do_sweep: bool = True
    GRAPH_K_CANDIDATES: tuple = (8, 12)
    K_UPPER: int = 3
    z_K: int = 3  # compatibility alias
    LAMBDA_SPARSE_LIST: tuple = (0.0, 0.001, 0.005)
    LAMBDA_MAGNITUDE_LIST: tuple = (0.001, 0.01)
    # entropy-in-F is built into the model; no separate entropy penalty sweep
    # occupancy barrier uses lam_sparse instead of lam_balance
    LAMBDA_MAGNITUDE_LIST: tuple = (0.001,)

    # Defaults
    graph_k: int = 12
    lam_sparse: float = 0.001
    lam_mag: float = 0.001
    lam_entropy: float = 0.0  # compatibility alias after removing entropy sweep
    lam_balance: float = 0.001  # compatibility alias -> mirrors lam_sparse

    # CV
    cv_folds: int = 5
    block_nx: int = 5
    block_ny: int = 5

    # GWR
    gwr_bw_min: int = 40
    gwr_bw_grid: tuple = (40, 60, 80, 100, 140, 180, 220)

    # Moran
    moran_knn: int = 8

    # Plotting
    dpi: int = 240
    PAPER_STYLE: bool = True
    USE_SEABORN: bool = True
    FONT_FAMILY: str = "DejaVu Sans"
    BASE_FONTSIZE: int = 11
    TITLE_FONTSIZE: int = 15
    LABEL_FONTSIZE: int = 12
    TICK_FONTSIZE: int = 11
    LEGEND_FONTSIZE: int = 11
    GRID_ALPHA: float = 0.25
    MAP_S: float = 24.0
    MAP_ALPHA: float = 0.7
    SCATTER_S: float = 30.0
    SCATTER_ALPHA: float = 0.7
    LINE_W: float = 2.2
    FIGB_SHOW_TRENDS: bool = False
    SENS_MAX_VARS: int = 5
    CMAP_SEQ: str = "cividis"
    CMAP_UNC: str = "viridis"
    CMAP_DIVERGE: str = "RdBu_r"
    REGIME_CMAP: dict | None = None
    REGIME_LABEL_OFFSET: int = 1


# ==============================================================================
# 2) UTILITIES / STYLE
# ==============================================================================
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def apply_paper_style(cfg: CFG):
    if not getattr(cfg, "PAPER_STYLE", True):
        return

    # seaborn is optional; if installed, it improves defaults
    if getattr(cfg, "USE_SEABORN", True):
        try:
            import seaborn as sns
            sns.set_theme(style="whitegrid", context="paper", font=cfg.FONT_FAMILY)
            sns.set_style({"axes.grid": True, "grid.alpha": cfg.GRID_ALPHA})
        except Exception:
            pass

    import matplotlib as mpl
    mpl.rcParams["figure.dpi"] = cfg.dpi
    mpl.rcParams["savefig.dpi"] = cfg.dpi
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["savefig.pad_inches"] = 0.03

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [cfg.FONT_FAMILY, "Arial", "Helvetica", "DejaVu Sans"]

    mpl.rcParams["font.size"] = cfg.BASE_FONTSIZE
    mpl.rcParams["axes.titlesize"] = cfg.TITLE_FONTSIZE
    mpl.rcParams["axes.labelsize"] = cfg.LABEL_FONTSIZE
    mpl.rcParams["xtick.labelsize"] = cfg.TICK_FONTSIZE
    mpl.rcParams["ytick.labelsize"] = cfg.TICK_FONTSIZE
    mpl.rcParams["legend.fontsize"] = cfg.LEGEND_FONTSIZE

    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.linewidth"] = 0.8
    mpl.rcParams["grid.alpha"] = cfg.GRID_ALPHA
    mpl.rcParams["grid.linewidth"] = 0.6

    # Make PDFs/PS text selectable
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def std_z(v):
    v = np.asarray(v)
    return (v - np.nanmean(v, axis=0)) / (np.nanstd(v, axis=0) + 1e-12)


def symm_limits(a, q=99.0):
    a = np.asarray(a)
    v = np.nanpercentile(np.abs(a), q)
    if (not np.isfinite(v)) or v <= 0:
        v = 1.0
    return -v, v


def _get_diverging_cmap(name="coolwarm"):
    try:
        import matplotlib as mpl
        try:
            return mpl.colormaps.get_cmap(name)
        except Exception:
            return plt.get_cmap(name)
    except Exception:
        return plt.cm.coolwarm


def _get_sequential_cmap(name="viridis"):
    try:
        import matplotlib as mpl
        try:
            return mpl.colormaps.get_cmap(name)
        except Exception:
            return plt.get_cmap(name)
    except Exception:
        return plt.cm.viridis


def _get_regime_cmap(cfg, uniq_labels):
    """Return dict {label: color}."""
    try:
        rc = getattr(cfg, "REGIME_CMAP")
        if isinstance(rc, dict) and len(rc) > 0:
            return rc
    except Exception:
        pass
    colors = list(plt.cm.tab10.colors)
    return {int(lab): colors[i % len(colors)] for i, lab in enumerate(sorted(list(uniq_labels)))}


def _get_regime_colors(K: int):
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
        return [cmap(i % 10) for i in range(int(K))]
    except Exception:
        return list(plt.cm.tab10.colors)[:int(K)]


def _add_cbar_right(ax, mappable, label, size="3.5%", pad=0.04):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cb = plt.colorbar(mappable, cax=cax)
    if label:
        cb.set_label(label)
    return cb


def _clean_ax(ax):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _paper_ax(ax, grid=True, alpha=0.25):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(True, linewidth=0.7, alpha=alpha)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _make_blocks(coords, nx, ny):
    coords = np.asarray(coords)
    x, y = coords[:, 0], coords[:, 1]
    xb = np.digitize(x, np.linspace(x.min(), x.max(), nx + 1)) - 1
    yb = np.digitize(y, np.linspace(y.min(), y.max(), ny + 1)) - 1
    return np.clip(yb * nx + xb, 0, nx * ny - 1)


def smooth_knn_field(coords, val, k=10, steps=2):
    coords = np.asarray(coords)
    val = np.asarray(val).astype(float)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    v = val.copy()
    for _ in range(steps):
        v_new = np.zeros_like(v)
        for i in range(len(v)):
            v_new[i] = 0.5 * v[i] + 0.5 * v[idx[i, 1:]].mean()
        v = v_new
    return std_z(v)


def make_knn_edges(coords, k):
    coords = np.asarray(coords)
    nbrs = NearestNeighbors(n_neighbors=int(k) + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    src, dst = [], []
    for i in range(len(coords)):
        for j in idx[i, 1:]:
            src.append(i); dst.append(j)
            src.append(j); dst.append(i)
    return np.array([src, dst], dtype=np.int64)


def disconnect_train_test_edges(edges, train_mask):
    u = edges[0]
    v = edges[1]
    tm_u = train_mask[u]
    tm_v = train_mask[v]
    keep = (tm_u & tm_v) | ((~tm_u) & (~tm_v))
    return edges[:, keep]


def moran_residuals(resid, coords, k=8):
    if not _HAVE_MORAN:
        return np.nan, np.nan
    w = _KNN_W.from_array(np.asarray(coords), k=int(k))
    mi = _Moran(np.asarray(resid), w)
    return float(mi.I), float(mi.p_sim)


def _safe_save_error_figure(out_path, title, err):
    fig = plt.figure(figsize=(8, 3), dpi=200)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.01, 0.80, title, fontsize=12, fontweight="bold")
    ax.text(0.01, 0.50, "Figure failed with exception:", fontsize=10)
    ax.text(0.01, 0.25, str(err), fontsize=9, family="monospace")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def safe_run_figure(fn, out_path, *args, **kwargs):
    """Run a paper figure strictly; never save an error placeholder as a successful figure.

    Research models, CV and figure calculations are unchanged.
    """
    try:
        fn(*args, out_path=out_path, **kwargs)
    except TypeError as exc:
        # Historical plotting functions use either keyword or positional out_path.
        # Preserve the original fallback but report failures instead of an error-PNG.
        try:
            fn(*args, out_path, **kwargs)
        except Exception as second:
            raise RuntimeError(f"Figure generation failed: {fn.__name__} -> {out_path}: {second}") from second
    except Exception as exc:
        raise RuntimeError(f"Figure generation failed: {fn.__name__} -> {out_path}: {exc}") from exc
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"Figure was not created: {fn.__name__} -> {out_path}")


def _paper_map_axes(ax):
    """Paper-style map axes: no ticks, equal aspect, no frame."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _map_symm_vlim(values, q=98, floor=1.0):
    """Symmetric vmin/vmax for diverging maps."""
    v = np.nanpercentile(np.abs(np.asarray(values).ravel()), q)
    if (not np.isfinite(v)) or v <= 0:
        v = float(floor)
    v = max(float(v), float(floor))
    return -v, v

# ==============================================================================
# 3) SIMULATION ENGINE
# ==============================================================================
def make_regimes_boundaries(coords, K, rng=None):
    """
    [Global Linear]
    """
    return np.zeros(len(coords), dtype=int)


def true_mechanism_FES_linear(X, regimes, cfg=None):
    """
    [Global Linear Mechanism - Mixed Signs]
    - Predictors may have positive or negative effects.
    - E = X @ w + b
    - The Capacity component S is fixed to zero.
    """
    X = np.asarray(X)
    n, p = X.shape




    w_global = np.zeros(p)
    if p >= 5:
        w_global[0] = 1.5
        w_global[1] = 1.2
        w_global[2] = 0.9
        w_global[3] = -0.9
        w_global[4] = -1.2

    else:

        w_global = np.linspace(1.5, -1.5, p)


    b_global = 0.0



    E = (X @ w_global) + b_global
    

    S = np.zeros(n, dtype=float)

    Fv = E - S
    return Fv, E, S

def fd_local_grads_FES_linear(X, regimes, cfg: CFG, eps=0.01):
    """
    Finite-difference gradients for Global Linear Mechanism.
    """
    n, p = X.shape
    dF = np.zeros((n, p))
    dE = np.zeros((n, p))
    dS = np.zeros((n, p))

    for j in range(p):
        Xp = X.copy()
        Xm = X.copy()
        Xp[:, j] += eps
        Xm[:, j] -= eps
        
        Fp, Ep, Sp = true_mechanism_FES_linear(Xp, regimes, cfg=cfg)
        Fm, Em, Sm = true_mechanism_FES_linear(Xm, regimes, cfg=cfg)
        
        dF[:, j] = (Fp - Fm) / (2 * eps)
        dE[:, j] = (Ep - Em) / (2 * eps)
        dS[:, j] = (Sp - Sm) / (2 * eps)

    return dF, dE, dS


def generate_simulation_linear(cfg: CFG):
    rng = np.random.default_rng(cfg.seed)

    # 1. Grid & Coordinates
    xs = np.linspace(0, 1, cfg.grid_nx)
    ys = np.linspace(0, 1, cfg.grid_ny)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    coords = coords + rng.normal(0, cfg.jitter, size=coords.shape)
    coords_std = std_z(coords)

    # 2. Regimes (Global = Single)
    regimes = make_regimes_boundaries(coords, cfg.k_true, rng=rng)

    # 3. Covariates (Spatially smooth X)
    X = np.zeros((len(coords), cfg.p))
    for j in range(cfg.p):
        base = rng.normal(0, 1, size=len(coords))
        X[:, j] = smooth_knn_field(coords_std, base, k=15, steps=3)
    X_std = std_z(X)

    # 4. Mechanism (Global Linear + entropy-in-F compatible DGP)
    F_base, E_true, S_true = true_mechanism_FES_linear(X_std, regimes, cfg=cfg)
    K_true = max(int(np.max(regimes)) + 1, 1)
    p_true = smooth_regime_memberships(coords_std, regimes, K_true, k=12, steps=2, tau=1.15)
    H_true = -np.sum(p_true * np.log(p_true + 1e-12), axis=1)
    H_true_norm = H_true / max(np.log(max(2, K_true)), 1e-12) if K_true > 1 else np.zeros(len(coords))
    T_true = np.linspace(0.8, 1.2, K_true)
    T_eff_true = (p_true * T_true[None, :]).sum(axis=1)
    F_true = F_base + T_eff_true * H_true_norm

    # 5. Outcome y
    y = F_true + rng.normal(0, cfg.iid_noise, len(coords))
    y_std = std_z(y)

    # 6. Gradients
    dF_true, dE_true, dS_true = fd_local_grads_FES_linear(X_std, regimes, cfg, eps=cfg.fd_eps)

    F_true_std = std_z(F_true)
    E_true_std = std_z(E_true)
    

    if np.nanstd(S_true) < 1e-9:
        S_true_std = np.zeros_like(S_true)
    else:
        S_true_std = std_z(S_true)

    return {
        "coords": coords,
        "coords_std": coords_std,
        "regimes_true": regimes,
        "X_raw": X,
        "y_raw": y,
        "X_std": X_std,
        "y_std": y_std,
        "F_true": F_true,
        "E_true": E_true,
        "S_true": S_true,
        "F_true_std": F_true_std,
        "E_true_std": E_true_std,
        "S_true_std": S_true_std,
        "dF_true": dF_true,
        "dE_true": dE_true,
        "dS_true": dS_true,
    }



def regime_utilization_metrics(p):
    p = np.asarray(p, dtype=float)
    eps = 1e-12
    if p.ndim != 2 or p.shape[0] == 0:
        return np.nan, np.nan, np.nan
    mean_p = p.mean(axis=0)
    H_global = -np.sum(mean_p * np.log(mean_p + eps))
    n_eff_global = float(np.exp(H_global))
    H_local = -np.sum(p * np.log(p + eps), axis=1)
    n_eff_local = float(np.mean(np.exp(H_local)))
    argmax_share_max = float(np.max(np.bincount(np.argmax(p, axis=1), minlength=p.shape[1]) / p.shape[0]))
    return n_eff_global, n_eff_local, argmax_share_max


def smooth_regime_memberships(coords_std, regimes, K, k=12, steps=2, tau=1.25):
    """Create soft memberships that preserve the original regime shape but allow transition ambiguity."""
    regimes = np.asarray(regimes).astype(int)
    p = np.eye(K)[regimes]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(coords_std))).fit(coords_std)
    nn = nbrs.kneighbors(coords_std, return_distance=False)[:, 1:]
    for _ in range(int(steps)):
        p = 0.5 * p + 0.5 * p[nn].mean(axis=1)
        p = np.clip(p, 1e-9, None)
        p /= p.sum(axis=1, keepdims=True)
    logits = np.log(p + 1e-12) / max(1e-6, float(tau))
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    return p


# ==============================================================================
# 4) MODELS
# ==============================================================================
class FastGWR:
    """Lightweight GWR: local weighted least squares with kNN bandwidth."""
    def __init__(self, coords, X, y, bw):
        self.coords = np.asarray(coords)
        self.X = np.asarray(X)
        self.y = np.asarray(y).ravel()
        self.bw = int(bw)
        self.nbrs = NearestNeighbors(n_neighbors=self.bw).fit(self.coords)

    def _get_beta(self, q_coords):
        dists, idxs = self.nbrs.kneighbors(q_coords.reshape(1, -1))
        d = dists[0]
        denom = np.max(d) + 1e-12
        w = (1 - (d / denom) ** 2) ** 2  # bisquare

        X_sub = self.X[idxs[0]]
        y_sub = self.y[idxs[0]]
        X_des = np.column_stack([np.ones(len(X_sub)), X_sub])

        W = np.diag(w)
        XtW = X_des.T @ W
        try:
            beta = np.linalg.solve(XtW @ X_des, XtW @ y_sub)
        except Exception:
            beta = np.zeros(X_des.shape[1])
        return beta

    def predict(self, coords_eval, X_eval, return_coeffs=False):
        coords_eval = np.asarray(coords_eval)
        X_eval = np.asarray(X_eval)
        n = len(coords_eval)
        preds = np.zeros(n)
        coeffs = np.zeros((n, X_eval.shape[1]))
        for i in range(n):
            beta = self._get_beta(coords_eval[i])
            preds[i] = beta[0] + np.dot(beta[1:], X_eval[i])
            coeffs[i] = beta[1:]
        if return_coeffs:
            return preds, coeffs
        return preds


def select_gwr_bandwidth_once(coords, X, y, cfg: CFG):
    n = len(y)
    grid = []
    for bw in cfg.gwr_bw_grid:
        bw2 = max(cfg.gwr_bw_min, int(bw))
        bw2 = min(bw2, n - 1)
        grid.append(bw2)
    grid = sorted(list(set(grid)))

    best_bw = grid[0]
    best_obj = np.inf
    k_par = X.shape[1] + 1

    for bw in grid:
        gwr = FastGWR(coords, X, y, bw)
        yhat = gwr.predict(coords, X)
        resid = y - yhat
        rss = float(np.sum(resid ** 2))
        obj = rss + 2.0 * k_par
        if obj < best_obj:
            best_obj = obj
            best_bw = bw
    return int(best_bw)


# Torch models
class MLPReg(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)


class StandardGNN(nn.Module):
    def __init__(self, d_in, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x, edges):
        h = F.relu(self.fc1(x))
        row, col = edges
        aggr = torch.zeros_like(h).index_add_(0, row, h[col])
        deg = torch.zeros(h.size(0), 1, device=x.device).index_add_(0, row, torch.ones(row.size(0), 1, device=x.device))
        h = (h + aggr) / (deg + 1.0)
        h = F.relu(self.fc2(h))
        return self.out(h).squeeze(1)


class ZentropyGNN(nn.Module):
    """
    Updated Zentropy model aligned to the final (cell 3) logic.
    - graph-smoothed gating only
    - local E/S encoders
    - entropy-in-F: F = sum_k p_k F_k + T_eff * Hbar
    """
    def __init__(self, d_in_E, d_in_S, hidden, K, entropy_in_F=True):
        super().__init__()
        self.K = int(K)
        self.entropy_in_F = bool(entropy_in_F)
        d_gate = int(d_in_E + d_in_S + 2)
        self.gate = nn.Sequential(
            nn.Linear(d_gate, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.K)
        )
        self.enc_E = nn.Sequential(nn.Linear(d_in_E, hidden), nn.ReLU())
        self.head_E = nn.Linear(hidden, self.K)
        self.enc_S = nn.Sequential(nn.Linear(d_in_S, hidden), nn.ReLU())
        self.head_S = nn.Linear(hidden, self.K)
        self.T_raw = nn.Parameter(torch.zeros(self.K))

    def _mean_aggregate(self, Z, edges):
        row, col = edges
        aggr = torch.zeros_like(Z).index_add_(0, row, Z[col])
        deg = torch.zeros(Z.size(0), 1, device=Z.device).index_add_(0, row, torch.ones(row.size(0), 1, device=Z.device))
        return (Z + aggr) / (deg + 1.0)

    def forward(self, x_E, x_S, coords2, edges):
        gate_in = torch.cat([x_E, x_S, coords2], dim=1)
        logits = self.gate(gate_in)
        logits = self._mean_aggregate(logits, edges)
        p = torch.softmax(logits, dim=1)

        hE = self.enc_E(x_E)
        E_k = self.head_E(hE)

        if x_S.shape[1] > 0:
            hS = self.enc_S(x_S)
            S_k = self.head_S(hS)
        else:
            S_k = torch.zeros(x_E.size(0), self.K, device=x_E.device, dtype=x_E.dtype)

        T = F.softplus(self.T_raw) + 1e-6
        F_k = E_k - S_k * T.view(1, -1)
        F_mix = (p * F_k).sum(1)

        eps = 1e-9
        H_raw = -torch.sum(p * torch.log(p + eps), dim=1)
        if self.K > 1:
            H_bar = H_raw / (torch.log(torch.tensor(float(self.K), device=H_raw.device, dtype=H_raw.dtype)) + 1e-12)
        else:
            H_bar = torch.zeros_like(H_raw)
        T_eff = torch.sum(p * T.view(1, -1), dim=1)
        if self.entropy_in_F:
            F_mix = F_mix + T_eff * H_bar

        E_mix = (p * E_k).sum(1)
        S_mix = (p * S_k).sum(1)
        return {"F": F_mix, "E": E_mix, "S": S_mix, "p": p, "F_k": F_k, "E_k": E_k, "S_k": S_k,
                "T": T, "T_eff": T_eff, "H_node": H_raw, "H_node_norm": H_bar}




def train_torch(model, data_dict, train_mask, edges, cfg: CFG, model_name="DNN", epochs=None):
    device = get_device()
    X = torch.tensor(data_dict["X"], dtype=torch.float32, device=device)
    y = torch.tensor(data_dict["y"], dtype=torch.float32, device=device)
    c = torch.tensor(data_dict["coords"], dtype=torch.float32, device=device)
    e = torch.tensor(edges, dtype=torch.long, device=device)

    if model_name == "DNN":
        model = MLPReg(X.shape[1], cfg.hidden).to(device)
        ep = epochs or int(cfg.epochs_final * 0.7)
    elif model_name == "GNN":
        model = StandardGNN(X.shape[1], cfg.hidden).to(device)
        ep = epochs or int(cfg.epochs_final * 0.7)
    elif model_name == "Zentropy":
        model = ZentropyGNN(len(cfg.E_idx), len(cfg.S_idx), cfg.hidden, cfg.K_UPPER, entropy_in_F=True).to(device)
        ep = epochs or cfg.epochs_final
    else:
        raise ValueError(model_name)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    best = np.inf
    best_state = None
    bad = 0

    for _t in range(int(ep)):
        model.train()
        opt.zero_grad()
        if model_name == "DNN":
            yhat = model(X)
            loss = F.mse_loss(yhat[train_mask], y[train_mask])
        elif model_name == "GNN":
            yhat = model(X, e)
            loss = F.mse_loss(yhat[train_mask], y[train_mask])
        else:
            out = model(X[:, cfg.E_idx], X[:, cfg.S_idx], c, e)
            mse = F.mse_loss(out["F"][train_mask], y[train_mask])
            p_tr = out["p"][train_mask]
            mean_p = p_tr.mean(0)
            # retain the original sign choice discussed with the user: anti-collapse occupancy barrier
            sparse_penalty = -torch.sum(torch.log(mean_p + 1e-9))
            mag = (out["E_k"][train_mask] ** 2).mean() + (out["S_k"][train_mask] ** 2).mean()
            loss = mse + cfg.lam_sparse * sparse_penalty + cfg.lam_mag * mag

        loss.backward()
        opt.step()

        val = float(loss.detach().cpu().item())
        if val < best - 1e-6:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if model_name == "Zentropy" and best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        if model_name == "DNN":
            yhat = model(X).detach().cpu().numpy()
            return yhat, model
        elif model_name == "GNN":
            yhat = model(X, e).detach().cpu().numpy()
            return yhat, model
        else:
            out = model(X[:, cfg.E_idx], X[:, cfg.S_idx], c, e)
            return out["F"].detach().cpu().numpy(), model




def zentropy_internals(model: ZentropyGNN, data_dict, edges, cfg: CFG):
    device = next(model.parameters()).device
    X = torch.tensor(data_dict["X"], dtype=torch.float32, device=device)
    c = torch.tensor(data_dict["coords"], dtype=torch.float32, device=device)
    e = torch.tensor(edges, dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(X[:, cfg.E_idx], X[:, cfg.S_idx], c, e)
    return (
        out["F"].detach().cpu().numpy(),
        out["p"].detach().cpu().numpy(),
        out["F_k"].detach().cpu().numpy(),
        out["E_k"].detach().cpu().numpy(),
        out["S_k"].detach().cpu().numpy(),
        out["E"].detach().cpu().numpy(),
        out["S"].detach().cpu().numpy(),
        out["H_node_norm"].detach().cpu().numpy(),
    )




def zentropy_grads(model: ZentropyGNN, data_dict, edges, cfg: CFG):
    device = next(model.parameters()).device
    X = torch.tensor(data_dict["X"], dtype=torch.float32, device=device, requires_grad=True)
    c = torch.tensor(data_dict["coords"], dtype=torch.float32, device=device)
    e = torch.tensor(edges, dtype=torch.long, device=device)

    out = model(X[:, cfg.E_idx], X[:, cfg.S_idx], c, e)
    gF = torch.autograd.grad(out["F"].sum(), X, retain_graph=True)[0]
    gE = torch.autograd.grad(out["E"].sum(), X, retain_graph=True)[0]
    gS = torch.autograd.grad(out["S"].sum(), X)[0]
    return gF.detach().cpu().numpy(), gE.detach().cpu().numpy(), gS.detach().cpu().numpy()


# ==============================================================================
# 5) FIGURES

# ==============================================================================
# 5) FIGURES
# ==============================================================================
def fig_B_true_overview_paper(coords_std, true_reg, y_std, cfg: CFG, out_path):
    coords_std = np.asarray(coords_std)
    true_reg = np.asarray(true_reg).astype(int)
    y_std = np.asarray(y_std, float)

    fig = plt.figure(figsize=(10.8, 4.2), dpi=cfg.dpi)
    gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.0, 1.15], wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    uniq = sorted(np.unique(true_reg).tolist())
    cmap = _get_regime_cmap(cfg, uniq)

    for k in uniq:
        m = (true_reg == k)
        ax0.scatter(coords_std[m, 0], coords_std[m, 1], s=cfg.MAP_S, alpha=cfg.MAP_ALPHA,
                    rasterized=True, color=cmap.get(int(k), "#333333"), linewidths=0)

    ax0.set_title("True regimes")
    ax0.set_xticks([]); ax0.set_yticks([])
    ax0.set_aspect("equal", adjustable="box")


    if len(uniq) > 1:
        handles, labels = [], []
        for k in uniq:
            handles.append(plt.Line2D([0], [0], marker="s", linestyle="None", markersize=7,
                                      markerfacecolor=cmap.get(int(k), "#333333"), markeredgewidth=0))
            labels.append(str(int(k + cfg.REGIME_LABEL_OFFSET)))
        ax0.legend(handles, labels, title="Regime", frameon=False, ncol=len(uniq),
                   bbox_to_anchor=(0.0, -0.08), loc="upper left")

    sc = ax1.scatter(coords_std[:, 0], coords_std[:, 1], c=y_std, s=cfg.MAP_S, alpha=cfg.MAP_ALPHA,
                     rasterized=True, cmap=_get_diverging_cmap(cfg.CMAP_DIVERGE),
                     vmin=np.nanpercentile(y_std, 2), vmax=np.nanpercentile(y_std, 98), linewidths=0)
    ax1.set_title("Observed outcome y")
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(sc, ax=ax1, fraction=0.046, pad=0.02)
    cbar.set_label("y (std)")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def fig_B_dashboard_paper(coords_std, true_reg, pred_reg, unc01, mean_p,
                          y_std, F_pred_std, E_pred_std, S_pred_std,
                          cfg, out_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # -----------------------------
    # local helpers (self-contained)
    # -----------------------------
    def _paper_map_axes(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        for sp in ax.spines.values():
            sp.set_visible(False)

    def _add_cbar_right(ax, mappable, label):
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="3.8%", pad=0.04)
        cb = plt.colorbar(mappable, cax=cax)
        cb.set_label(label)
        return cb

    def _map_symm_vlim(vals, q=98, floor=1.0):
        vals = np.asarray(vals).reshape(-1)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return -floor, floor
        vmax = np.nanpercentile(np.abs(vals), q)
        if (not np.isfinite(vmax)) or (vmax <= 0):
            vmax = floor
        vmax = max(float(vmax), float(floor))
        return -vmax, vmax

    def _get_regime_colors(K):
        base = list(plt.cm.tab10.colors) + list(plt.cm.tab20.colors)
        if K <= len(base):
            return base[:K]
        cmap = plt.get_cmap("tab20", K)
        return [cmap(i) for i in range(K)]

    def _get_seq_cmap(name):
        try:
            return plt.get_cmap(name)
        except Exception:
            return plt.get_cmap("viridis")

    def _get_div_cmap(name):
        try:
            return plt.get_cmap(name)
        except Exception:
            return plt.get_cmap("RdBu_r")

    # -----------------------------
    # inputs
    # -----------------------------
    coords_std = np.asarray(coords_std, float)
    true_reg = np.asarray(true_reg).astype(int)
    pred_reg = np.asarray(pred_reg).astype(int)
    unc01 = np.asarray(unc01).astype(float)

    y_std = np.asarray(y_std).astype(float)
    F_pred_std = np.asarray(F_pred_std).astype(float)
    E_pred_std = np.asarray(E_pred_std).astype(float)
    S_pred_std = np.asarray(S_pred_std).astype(float)

    s_map = float(max(getattr(cfg, "MAP_S", 10.0), 14.0))
    a_map = float(getattr(cfg, "MAP_ALPHA", 0.62))
    dpi = int(getattr(cfg, "dpi", 300))
    lab_off = int(getattr(cfg, "REGIME_LABEL_OFFSET", 1))

    K = int(np.max(true_reg) + 1)
    colors = _get_regime_colors(K)

    cmap_unc = _get_seq_cmap(getattr(cfg, "CMAP_UNC", "viridis"))
    cmap_div = _get_div_cmap(getattr(cfg, "CMAP_DIVERGE", "RdBu_r"))

    # shared diverging limits for bottom row
    allv = np.concatenate([
        y_std.ravel(),
        F_pred_std.ravel(),
        E_pred_std.ravel(),
        S_pred_std.ravel()
    ])
    vmin_all, vmax_all = _map_symm_vlim(allv, q=98, floor=1.0)

    # -----------------------------
    # figure layout
    # top row: 3 equal panels
    # middle row: centered regime legend
    # bottom row: 4 equal panels
    # -----------------------------
    fig = plt.figure(figsize=(18.8, 9.2), dpi=dpi)
    outer = fig.add_gridspec(
        3, 1,
        height_ratios=[1.0, 0.13, 1.0],
        hspace=0.12
    )

    gs_top = outer[0].subgridspec(1, 3, wspace=0.25)
    gs_bot = outer[2].subgridspec(1, 4, wspace=0.25)

    # -----------------------------
    # top row
    # -----------------------------
    ax_true = fig.add_subplot(gs_top[0, 0])
    for k in range(K):
        m = (true_reg == k)
        if np.any(m):
            ax_true.scatter(
                coords_std[m, 0], coords_std[m, 1],
                s=s_map, alpha=a_map, color=colors[k],
                linewidths=0, rasterized=True
            )
    ax_true.set_title("True regimes", fontweight="bold", pad=10)
    _paper_map_axes(ax_true)

    ax_pred = fig.add_subplot(gs_top[0, 1])
    for k in range(K):
        m = (pred_reg == k)
        if np.any(m):
            ax_pred.scatter(
                coords_std[m, 0], coords_std[m, 1],
                s=s_map, alpha=a_map, color=colors[k],
                linewidths=0, rasterized=True
            )
    ax_pred.set_title("Predicted regimes", fontweight="bold", pad=10)
    _paper_map_axes(ax_pred)

    ax_unc = fig.add_subplot(gs_top[0, 2])
    ord_unc = np.argsort(np.clip(unc01, 0, 1))
    sc_unc = ax_unc.scatter(
        coords_std[ord_unc, 0], coords_std[ord_unc, 1],
        c=np.clip(unc01[ord_unc], 0, 1),
        s=s_map, alpha=a_map, rasterized=True,
        cmap=cmap_unc, vmin=0, vmax=1,
        linewidths=0
    )
    ax_unc.set_title("Gating uncertainty (normalized entropy)", fontweight="bold", pad=10)
    _paper_map_axes(ax_unc)
    _add_cbar_right(ax_unc, sc_unc, "H(p) (0–1)")

    # -----------------------------
    # legend row
    # -----------------------------
    ax_leg = fig.add_subplot(outer[1])
    ax_leg.axis("off")
    handles = [Patch(facecolor=colors[k], edgecolor="none", label=str(k + lab_off)) for k in range(K)]
    ax_leg.legend(
        handles=handles,
        title="Regime",
        loc="center",
        ncol=min(K, 8),
        frameon=False,
        handlelength=1.0,
        columnspacing=1.2
    )

    # -----------------------------
    # bottom row
    # -----------------------------
    def _panel(ax, vals, title, cblab):
        vals = np.asarray(vals).ravel()
        ord_ = np.argsort(vals)
        sc = ax.scatter(
            coords_std[ord_, 0], coords_std[ord_, 1],
            c=vals[ord_],
            s=s_map, alpha=a_map, rasterized=True,
            cmap=cmap_div, vmin=vmin_all, vmax=vmax_all,
            linewidths=0
        )
        ax.set_title(title, fontweight="bold", pad=10)
        _paper_map_axes(ax)
        _add_cbar_right(ax, sc, cblab)

    ax_y = fig.add_subplot(gs_bot[0, 0])
    _panel(ax_y, y_std, "Observed outcome y", "y")

    ax_F = fig.add_subplot(gs_bot[0, 1])
    _panel(ax_F, F_pred_std, "Predicted free energy F", "F")

    ax_E = fig.add_subplot(gs_bot[0, 2])
    _panel(ax_E, E_pred_std, "Predicted burden E", "E")

    ax_S = fig.add_subplot(gs_bot[0, 3])
    _panel(ax_S, S_pred_std, "Predicted capacity S", "S")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)




import matplotlib.patches as mpatches

def fig_B_yx_scatter_panel(y_std, X_std, regimes, cfg, out_path, var_names=None, show_trends=False):
    y_std = np.asarray(y_std).ravel()
    X_std = np.asarray(X_std)
    regimes = np.asarray(regimes).astype(int)

    n, d = X_std.shape
    if var_names is None:
        var_names = [f"x{j+1}" for j in range(d)]


    s = float(max(getattr(cfg, "SCATTER_S", 12.0), 18.0))
    a = float(getattr(cfg, "SCATTER_ALPHA", 0.60))
    lab_off = int(getattr(cfg, "REGIME_LABEL_OFFSET", 1))

    K = int(np.max(regimes) + 1)
    colors = _get_regime_colors(K)

    cols = min(d, 5)
    fig, axes = plt.subplots(1, cols, figsize=(3.9 * cols, 4.2), dpi=cfg.dpi)
    if cols == 1:
        axes = [axes]

    for j, ax in enumerate(axes):
        xj = X_std[:, j]
        for k in range(K):
            m = (regimes == k)
            if np.any(m):
                ax.scatter(xj[m], y_std[m], s=s, alpha=a, color=colors[k],
                           linewidths=0, rasterized=True)

        if show_trends:
            centers, means = _binned_mean(xj, y_std, nbins=25)
            ax.plot(centers, means, lw=2.0, color="k", alpha=0.85)

        ax.set_title(f"y vs {var_names[j]}", fontweight="bold")
        ax.set_xlabel(var_names[j])
        if j == 0:
            ax.set_ylabel("y")
        ax.set_axisbelow(True)
        ax.grid(True, linewidth=0.7, alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color=colors[k], label=str(k + lab_off)) for k in range(K)]
    fig.legend(handles=handles, title="Regime", loc="lower center", ncol=min(K, 8),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def _binned_mean(x, y, nbins=25):
    x = np.asarray(x); y = np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if len(x) < max(60, nbins * 3):
        nbins = max(10, len(x) // 12)

    qs = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(x, qs)
    edges = np.unique(edges)
    if len(edges) < 6:
        edges = np.linspace(np.min(x), np.max(x), nbins + 1)

    centers, means = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x <= b)
        if m.sum() < 10:
            continue
        centers.append(0.5 * (a + b))
        means.append(float(np.mean(y[m])))
    return np.array(centers), np.array(means)


def fig_C_true_vs_pred_FES(coords_std, F_true_std, E_true_std, S_true_std, F_pred_std, E_pred_std, S_pred_std, cfg: CFG, out_path):
    fig = plt.figure(figsize=(18, 9), dpi=cfg.dpi)
    gs = fig.add_gridspec(2, 3, wspace=0.22, hspace=0.22)

    cols = [
        ("Free energy F", F_true_std, F_pred_std),
        ("Burden E", E_true_std, E_pred_std),
        ("Capacity S", S_true_std, S_pred_std),
    ]

    for c, (name, t, p) in enumerate(cols):
        vmin, vmax = symm_limits(np.concatenate([t, p]), 99)

        # True Plot
        axT = fig.add_subplot(gs[0, c])
        scT = axT.scatter(coords_std[:, 0], coords_std[:, 1], c=t, s=cfg.MAP_S,
                          cmap=_get_diverging_cmap(cfg.CMAP_DIVERGE), vmin=vmin, vmax=vmax, linewidth=0)
        axT.set_title(f"True {name} (std)", fontweight="bold")
        axT.set_xticks([]); axT.set_yticks([])
        

        axT.set_aspect("equal", adjustable="box")
        
        _add_cbar_right(axT, scT, "")

        # Pred Plot
        axP = fig.add_subplot(gs[1, c])
        scP = axP.scatter(coords_std[:, 0], coords_std[:, 1], c=p, s=cfg.MAP_S,
                          cmap=_get_diverging_cmap(cfg.CMAP_DIVERGE), vmin=vmin, vmax=vmax, linewidth=0)
        axP.set_title(f"Predicted {name} (std)", fontweight="bold")
        axP.set_xticks([]); axP.set_yticks([])
        

        axP.set_aspect("equal", adjustable="box")
        
        _add_cbar_right(axP, scP, "")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_D_thermo_diagnostics(dF_true, dF_zen, dE_zen, dS_zen, var_names, cfg, out_path):
    from matplotlib.colors import Normalize
    import numpy as np
    import matplotlib.pyplot as plt

    dF_true = np.asarray(dF_true, float)
    dF_zen  = np.asarray(dF_zen,  float)
    dE_zen  = np.asarray(dE_zen,  float)
    dS_zen  = np.asarray(dS_zen,  float)

    # Global importance (median |dF/dx|)
    I_true = np.nanmedian(np.abs(dF_true), axis=0)
    I_zen  = np.nanmedian(np.abs(dF_zen),  axis=0)

    # Burden dominance color (same as before; used in other panels)
    I_E = np.nanmedian(np.abs(dE_zen), axis=0)
    I_S = np.nanmedian(np.abs(dS_zen), axis=0)
    D_E = (I_E / (I_E + I_S + 1e-12)).clip(0, 1)

    # role reversal index (use Zentropy dF/dx)
    rr = []
    for j in range(dF_zen.shape[1]):
        g = np.nanmedian(dF_zen[:, j])
        sgn_g = 1.0 if g >= 0 else -1.0
        sgn_loc = np.where(np.isnan(dF_zen[:, j]), np.nan, np.sign(dF_zen[:, j]))
        valid = np.isfinite(sgn_loc)
        rr.append(np.mean(sgn_loc[valid] != sgn_g) if valid.sum() else np.nan)
    rr = np.array(rr)

    cmap_div = _get_diverging_cmap(cfg.CMAP_DIVERGE)
    norm01 = Normalize(0, 1)

    fig = plt.figure(figsize=(16, 10), dpi=cfg.dpi)
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.30)

    # ============================================================
    # (1) Global importance: True vs Zentropy  [UPDATED to match Fig_F]
    #   - scale 0~1 separately (True/max(True), Zentropy/max(Zentropy))
    #   - sort by True (descending)
    #   - use 2 colors (no D_E colormap here)
    # ============================================================
    axA = fig.add_subplot(gs[0, 0])

    def _norm01_vec(a):
        a = np.asarray(a, float)
        mx = np.nanmax(a)
        return a / mx if (np.isfinite(mx) and mx > 0) else a

    I_true_n = _norm01_vec(I_true)
    I_zen_n  = _norm01_vec(I_zen)

    order = np.argsort(I_true_n)[::-1]
    y = np.arange(len(order))
    h = 0.38

    axA.barh(y - h/2, I_true_n[order], height=h, label="True")
    axA.barh(y + h/2, I_zen_n[order],  height=h, label="Zentropy")

    axA.set_yticks(y)
    axA.set_yticklabels([var_names[i] for i in order])
    axA.invert_yaxis()
    axA.set_xlabel("Global importance (scaled)")
    axA.set_title("Global importance comparison")
    axA.legend(frameon=False, loc="lower right")
    _paper_ax(axA, grid=False)

    # (2) Mechanism phase space (unchanged)
    axB = fig.add_subplot(gs[0, 1])
    sizes = 200 + 1800 * (I_zen / (np.nanmax(I_zen) + 1e-12))
    sc = axB.scatter(I_E, I_S, s=sizes, c=D_E, cmap=cmap_div, norm=norm01,
                     alpha=0.9, edgecolor="k", linewidth=0.4)
    mx = max(np.nanmax(I_E), np.nanmax(I_S))
    axB.plot([0, mx], [0, mx], "--", color="gray", alpha=0.6)
    axB.set_xlabel(r"I_E (burden)")
    axB.set_ylabel(r"I_S (capacity)")
    axB.set_title("Mechanism phase space")
    _paper_ax(axB, grid=True, alpha=0.22)
    cbar = fig.colorbar(sc, ax=axB, fraction=0.046, pad=0.04)
    cbar.set_label(r"Burden dominance $D_E$")

    # (3) Spatial role heterogeneity (unchanged)
    axC = fig.add_subplot(gs[1, 0])
    axC.axvspan(0.0, 0.25, alpha=0.08)
    axC.axvspan(0.50, 1.00, alpha=0.08)
    axC.scatter(rr, I_zen, s=90, color="gray", edgecolor="k", linewidth=0.3, alpha=0.9)
    for j, name in enumerate(var_names):
        axC.text(rr[j] + 0.01, I_zen[j] + 0.01 * (np.nanmax(I_zen) + 1e-12), name, fontsize=10)
    axC.set_xlim(0, 1)
    axC.set_xlabel("Role reversal index (sign flip rate)")
    axC.set_ylabel(r"I_F (importance, Zentropy)")
    axC.set_title("Spatial role heterogeneity")
    _paper_ax(axC, grid=True, alpha=0.22)

    # (4) Local sensitivity distributions (Zentropy) (unchanged)
    axD = fig.add_subplot(gs[1, 1])
    data = [dF_zen[:, j][np.isfinite(dF_zen[:, j])] for j in range(dF_zen.shape[1])]
    parts = axD.violinplot(data, vert=False, showmeans=False, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("k")
    axD.axvline(0, color="k", linewidth=1, alpha=0.7)
    axD.set_yticks(np.arange(1, len(var_names) + 1))
    axD.set_yticklabels(var_names)
    axD.set_xlabel(r"dF/dx")
    axD.set_title("Local sensitivity distributions (Zentropy)")
    _paper_ax(axD, grid=True, alpha=0.22)

    fig.suptitle("Thermodynamic diagnostics", fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def fig_E_sweep_k_curve(df_any, cfg, out_path):
    df = df_any.copy()
    if "k" in df.columns and "graph_k" not in df.columns:
        df = df.rename(columns={"k": "graph_k"})
    if "moran_mean" in df.columns and "moranI_mean" not in df.columns:
        df = df.rename(columns={"moran_mean": "moranI_mean"})
    if "r2" in df.columns and "r2_mean" not in df.columns:
        df["r2_mean"] = df["r2"]

    if "lam_entropy" in df.columns or "z_K" in df.columns:
        df = (df.sort_values(["graph_k", "r2_mean", "moranI_mean"], ascending=[True, False, True])
                .groupby("graph_k", as_index=False)
                .first())

    df = df.sort_values("graph_k")
    ks = df["graph_k"].astype(int).values
    r2m = df["r2_mean"].astype(float).values
    r2se = df["r2_se"].astype(float).values if "r2_se" in df.columns else np.zeros_like(r2m)
    mi = df["moranI_mean"].astype(float).values if "moranI_mean" in df.columns else np.full_like(r2m, np.nan)

    fig, ax1 = plt.subplots(figsize=(7.2, 3.8), dpi=cfg.dpi)
    ax1.errorbar(ks, r2m, yerr=r2se, marker="o")
    ax1.set_xlabel("k (graph neighbors)")
    ax1.set_ylabel("CV R² (mean ± SE)")
    ax1.set_title("Zentropy: k sweep (Spatial Block CV)")
    _paper_ax(ax1, grid=True, alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(ks, mi, marker="s", linestyle="--")
    ax2.set_ylabel("Held-out residual Moran's I")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_E_sweep_k_curve(df_any, cfg, out_path):
    df = df_any.copy()
    if "k" in df.columns and "graph_k" not in df.columns:
        df = df.rename(columns={"k": "graph_k"})
    if "moran_mean" in df.columns and "moranI_mean" not in df.columns:
        df = df.rename(columns={"moran_mean": "moranI_mean"})
    if "r2" in df.columns and "r2_mean" not in df.columns:
        df["r2_mean"] = df["r2"]

    if "lam_entropy" in df.columns or "z_K" in df.columns:
        df = (df.sort_values(["graph_k", "r2_mean", "moranI_mean"], ascending=[True, False, True])
                .groupby("graph_k", as_index=False)
                .first())

    df = df.sort_values("graph_k")
    ks = df["graph_k"].astype(int).values
    r2m = df["r2_mean"].astype(float).values
    r2se = df["r2_se"].astype(float).values if "r2_se" in df.columns else np.zeros_like(r2m)
    mi = df["moranI_mean"].astype(float).values if "moranI_mean" in df.columns else np.full_like(r2m, np.nan)

    fig, ax1 = plt.subplots(figsize=(7.2, 3.8), dpi=cfg.dpi)
    ax1.errorbar(ks, r2m, yerr=r2se, marker="o")
    ax1.set_xlabel("k (graph neighbors)")
    ax1.set_ylabel("CV R² (mean ± SE)")
    ax1.set_title("Zentropy: k sweep (Spatial Block CV)")
    _paper_ax(ax1, grid=True, alpha=0.22)

    ax2 = ax1.twinx()
    ax2.plot(ks, mi, marker="s", linestyle="--")
    ax2.set_ylabel("Held-out residual Moran's I")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_E_sweep_lambda_heatmaps(df_lam, cfg, out_path):
    """Visualize exactly the current full-grid search parameters.

    Input is the CURRENT selection table with columns:
    graph_k, K_upper, lam_sparse, lam_mag, r2_mean.
    This visualizes the empirical CV grid; it does not change training.
    """
    required = {"graph_k", "K_upper", "lam_sparse", "lam_mag", "r2_mean"}
    missing = required.difference(df_lam.columns)
    if missing:
        raise ValueError(f"Lambda plot input missing current-grid columns: {sorted(missing)}")
    df = df_lam.loc[(df_lam["graph_k"] == cfg.graph_k) &
                    (df_lam["K_upper"] == cfg.K_UPPER)].copy()
    if df.empty:
        raise ValueError(f"No lambda-sweep rows for graph_k={cfg.graph_k}, K_upper={cfg.K_UPPER}")

    lams_sparse = sorted(df["lam_sparse"].unique())
    lams_mag = sorted(df["lam_mag"].unique())
    mat = np.full((len(lams_sparse), len(lams_mag)), np.nan, dtype=float)
    for ri, ls in enumerate(lams_sparse):
        for ci, lm in enumerate(lams_mag):
            candidates = df.loc[(df["lam_sparse"] == ls) & (df["lam_mag"] == lm), "r2_mean"]
            if len(candidates):
                mat[ri, ci] = float(candidates.max())
    if not np.isfinite(mat).any():
        raise ValueError("Lambda sweep contained no finite CV R² values")
    low, high = float(np.nanmin(mat)), float(np.nanmax(mat))
    if high - low < 1e-12:
        high = low + 1e-6

    fig, ax = plt.subplots(figsize=(max(5.4, 1.4 + 1.2 * len(lams_mag)),
                                    max(4.0, 1.8 + 0.7 * len(lams_sparse))), dpi=cfg.dpi)
    im = ax.imshow(mat, cmap=_get_sequential_cmap(getattr(cfg, "CMAP_SEQ", "cividis")),
                   vmin=low, vmax=high, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(lams_mag)), [f"{v:g}" for v in lams_mag])
    ax.set_yticks(np.arange(len(lams_sparse)), [f"{v:g}" for v in lams_sparse])
    ax.set_xlabel(r"$\lambda_{mag}$")
    ax.set_ylabel(r"$\lambda_{sparse}$")
    ax.set_title(f"Spatial block CV at k={cfg.graph_k}, K={cfg.K_UPPER}")
    for ri in range(mat.shape[0]):
        for ci in range(mat.shape[1]):
            if np.isfinite(mat[ri, ci]):
                mid = (low + high) / 2
                ax.text(ci, ri, f"{mat[ri,ci]:.3f}", ha="center", va="center",
                        color="white" if mat[ri,ci] > mid else "black")
    fig.colorbar(im, ax=ax, fraction=0.048, pad=0.04, label=r"Block-CV mean $R^2$")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



def fig_F_cv_benchmark(cv_long, cfg: CFG, out_path):
    df0 = cv_long.copy()
    cols = set(df0.columns)

    # allow cv_summary input
    if "r2_mean" in cols and "rmse_mean" in cols:
        df = df0.copy()
        if "r2_std" not in df.columns: df["r2_std"] = 0.0
        if "rmse_std" not in df.columns: df["rmse_std"] = 0.0
    else:
        # fold-level long -> aggregate
        need = {"cv_type","model","r2","rmse"}
        if not need.issubset(cols):
            _safe_save_error_figure(out_path, "fig_F_cv_benchmark", f"Missing columns: {sorted(list(need-cols))}")
            return
        df = df0.groupby(["cv_type","model"], as_index=False).agg(
            r2_mean=("r2","mean"), r2_std=("r2","std"),
            rmse_mean=("rmse","mean"), rmse_std=("rmse","std")
        )
        df["r2_std"] = df["r2_std"].fillna(0.0)
        df["rmse_std"] = df["rmse_std"].fillna(0.0)

    model_order = ["OLS", "GWR", "DT", "RF", "GBR", "DNN", "GNN", "Zentropy"]
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values(["cv_type", "model"])

    cv_types = list(df["cv_type"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), dpi=cfg.dpi)

    ax = axes[0]
    for cvt in cv_types:
        sub = df[df["cv_type"] == cvt].copy()
        x = np.arange(len(sub))
        ax.errorbar(x, sub["r2_mean"], yerr=sub["r2_std"], marker="o", linestyle="-", label=str(cvt), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sub["model"]], rotation=25, ha="right")
    ax.set_ylabel("R²")
    ax.set_title("Cross-validation R² (all models)")
    _paper_ax(ax, grid=True, alpha=0.22)
    ax.legend(frameon=False)

    ax = axes[1]
    for cvt in cv_types:
        sub = df[df["cv_type"] == cvt].copy()
        x = np.arange(len(sub))
        ax.errorbar(x, sub["rmse_mean"], yerr=sub["rmse_std"], marker="o", linestyle="-", label=str(cvt), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in sub["model"]], rotation=25, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Cross-validation RMSE (all models)")
    _paper_ax(ax, grid=True, alpha=0.22)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_F_validation_importance_2x2(df_cv, dF_true, dF_zen, var_names, cfg, out_path):
    from matplotlib.lines import Line2D
    df = df_cv.copy()
    need = {"cv_type","model","r2","rmse"}
    if not need.issubset(df.columns):
        _safe_save_error_figure(out_path, "fig_F_validation_importance_2x2", f"Missing columns: {sorted(list(need-set(df.columns)))}")
        return

    agg = df.groupby(["cv_type", "model"], as_index=False).agg(
        r2_mean=("r2", "mean"), r2_sem=("r2", "sem"),
        rmse_mean=("rmse", "mean"), rmse_sem=("rmse", "sem")
    )

    models = ["OLS", "GWR", "DT", "RF", "GBR", "DNN", "GNN", "Zentropy"]
    models = [m for m in models if m in agg["model"].unique()]
    x = np.arange(len(models))

    fig = plt.figure(figsize=(16, 10), dpi=cfg.dpi)
    gs = fig.add_gridspec(2, 2, wspace=0.25, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    for i, cvt in enumerate(["Random", "Block"]):
        sub = agg[agg["cv_type"] == cvt].set_index("model").reindex(models)
        off = (i - 0.5) * 0.15
        ax1.errorbar(x + off, sub["r2_mean"], yerr=sub["r2_sem"], fmt="o", label=cvt, capsize=3)
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=25)
    ax1.set_ylabel("R²"); ax1.set_title("CV R² Score")
    ax1.legend(frameon=False)
    _paper_ax(ax1, grid=True, alpha=0.22)

    ax2 = fig.add_subplot(gs[0, 1])
    for i, cvt in enumerate(["Random", "Block"]):
        sub = agg[agg["cv_type"] == cvt].set_index("model").reindex(models)
        off = (i - 0.5) * 0.15
        ax2.errorbar(x + off, sub["rmse_mean"], yerr=sub["rmse_sem"], fmt="s", label=cvt, capsize=3)
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=25)
    ax2.set_ylabel("RMSE"); ax2.set_title("CV RMSE")
    ax2.legend(frameon=False)
    _paper_ax(ax2, grid=True, alpha=0.22)

    ax3 = fig.add_subplot(gs[1, 0])
    I_true = np.nanmedian(np.abs(dF_true), axis=0)
    I_zen = np.nanmedian(np.abs(dF_zen), axis=0)
    def _norm01(a):
        a = np.asarray(a, float)
        mx = np.nanmax(a)
        return a / mx if mx > 0 else a
    I_true_n = _norm01(I_true)
    I_zen_n = _norm01(I_zen)
    y = np.arange(len(var_names))
    h = 0.35
    ax3.barh(y - h/2, I_true_n, height=h, label="True")
    ax3.barh(y + h/2, I_zen_n, height=h, label="Zentropy")
    ax3.set_yticks(y)
    ax3.set_yticklabels(var_names)
    ax3.invert_yaxis()
    ax3.set_xlabel("importance (scaled)")
    ax3.set_title("Global importance comparison")
    ax3.legend(frameon=False, loc="lower right")
    _paper_ax(ax3, grid=False)

    ax4 = fig.add_subplot(gs[1, 1])
    yy = np.arange(1, len(var_names) + 1)
    parts_true = ax4.violinplot([dF_true[:, j] for j in range(dF_true.shape[1])], positions=yy-0.15,
                                vert=False, showmeans=False, showmedians=True, showextrema=False, widths=0.32)
    parts_zen = ax4.violinplot([dF_zen[:, j] for j in range(dF_zen.shape[1])], positions=yy+0.15,
                               vert=False, showmeans=False, showmedians=True, showextrema=False, widths=0.32)

    for pc in parts_true["bodies"]:
        pc.set_facecolor("C0"); pc.set_alpha(0.55); pc.set_edgecolor("none")
    for pc in parts_zen["bodies"]:
        pc.set_facecolor("C1"); pc.set_alpha(0.55); pc.set_edgecolor("none")
    if "cmedians" in parts_true: parts_true["cmedians"].set_color("C0")
    if "cmedians" in parts_zen: parts_zen["cmedians"].set_color("C1")

    ax4.axvline(0, color="k", lw=1, alpha=0.55)
    ax4.set_yticks(yy)
    ax4.set_yticklabels(var_names)
    ax4.invert_yaxis()
    ax4.set_xlabel("dF/dx")
    ax4.set_title("Local sensitivity distributions (dF)")

    legend_elements = [Line2D([0], [0], color="C0", lw=8, label="True"),
                       Line2D([0], [0], color="C1", lw=8, label="Pred")]
    ax4.legend(handles=legend_elements, frameon=False, loc="lower right")
    _paper_ax(ax4, grid=True, alpha=0.22)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_G_residual_diagnostics(coords_std, y_std, preds_dict, out_path, cfg):
    coords_std = np.asarray(coords_std)
    y_std = np.asarray(y_std, float)

    models = list(preds_dict.keys())
    resid_by_model = {}
    I_vals, p_vals = [], []

    for m in models:
        resid = y_std - np.asarray(preds_dict[m], float)
        resid_by_model[m] = resid
        I, p = moran_residuals(resid, coords_std, k=cfg.moran_knn)
        I_vals.append(I); p_vals.append(p)

    all_r = np.concatenate([r.ravel() for r in resid_by_model.values()])
    v = np.nanpercentile(np.abs(all_r), 98)
    v = float(v) if np.isfinite(v) and v > 0 else 1.0

    fig = plt.figure(figsize=(22, 9.5), dpi=cfg.dpi)
    gs = fig.add_gridspec(2, 5, width_ratios=[1.55, 1, 1, 1, 1], wspace=0.30, hspace=0.30)

    ax0 = fig.add_subplot(gs[:, 0])
    y_pos = np.arange(len(models))
    ax0.barh(y_pos, I_vals, color="0.45")
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(models)
    ax0.invert_yaxis()
    ax0.set_xlabel("Moran's I")
    ax0.set_title("Residual Moran's I (in-sample)", fontweight="bold")
    _paper_ax(ax0, grid=True, alpha=0.22)

    cmap = _get_diverging_cmap(cfg.CMAP_DIVERGE)

    models_plot = models[:8]  # 2x4
    last_idx = len(models_plot) - 1

    last_sc = None
    last_ax = None

    for idx, m in enumerate(models_plot):
        r = idx // 4
        c = idx % 4
        ax = fig.add_subplot(gs[r, c + 1])

        sc = ax.scatter(coords_std[:, 0], coords_std[:, 1], c=resid_by_model[m], s=cfg.MAP_S,
                        cmap=cmap, vmin=-v, vmax=v, linewidths=0, rasterized=True, alpha=cfg.MAP_ALPHA)
        ax.set_title(f"{m}\n(I={I_vals[idx]:.2f}, p={p_vals[idx]:.3f})", fontsize=10, fontweight="bold")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        last_sc = sc
        last_ax = ax


    if (last_ax is not None) and (last_sc is not None):
        _add_cbar_right(last_ax, last_sc, "Residuals")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame({"model": models, "I": I_vals, "p": p_vals})



def fig_H2_sensitivity_maps_combined(
    coords_std, var_names,
    dF_true, dE_true, dS_true,
    dF_zen, dE_zen, dS_zen,
    gwr_coeffs,
    cfg, out_path, max_vars=5
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    coords_std = np.asarray(coords_std)
    dF_true = np.asarray(dF_true)
    dE_true = np.asarray(dE_true)
    dS_true = np.asarray(dS_true)
    dF_zen  = np.asarray(dF_zen)
    dE_zen  = np.asarray(dE_zen)
    dS_zen  = np.asarray(dS_zen)
    gwr_coeffs = np.asarray(gwr_coeffs)

    d = int(dF_true.shape[1])
    m = min(int(max_vars), int(d))
    var_names = list(var_names)[:m]

    # rows (keep structure)
    rows = [
        ("True dF/dx",        dF_true[:, :m]),
        ("Zentropy dF/dx",    dF_zen[:,  :m]),
        ("Zentropy dE/dx",    dE_zen[:,  :m]),
        ("Zentropy dS/dx",    dS_zen[:,  :m]),
        ("GWR coeff",         gwr_coeffs[:, :m]),
    ]

    cmap = _get_diverging_cmap(getattr(cfg, "CMAP_DIVERGE", "RdBu_r"))
    s_map = float(max(getattr(cfg, "MAP_S", 10.0), 14.0))
    a_map = float(getattr(cfg, "MAP_ALPHA", 0.75))

    # robust symmetric limit helper
    def _robust_sym_lim(vals, q=98.0, min_v=1e-3):
        vals = np.asarray(vals, float).ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 1.0
        v = np.percentile(np.abs(vals), q)
        if (not np.isfinite(v)) or (v < min_v):
            v2 = np.nanstd(vals) * 2.5
            v = v2 if (np.isfinite(v2) and v2 >= min_v) else min_v
        return float(v)


    vmax_by_var = np.zeros(m, dtype=float)
    for j in range(m):
        cat = np.concatenate([
            rows[0][1][:, j],  # true dF
            rows[1][1][:, j],  # z dF
            rows[2][1][:, j],  # z dE
            rows[3][1][:, j],  # z dS
            rows[4][1][:, j],  # gwr
        ])
        vmax_by_var[j] = _robust_sym_lim(cat, q=98.0, min_v=1e-3)

    # optional fixed vlim override (if user wants -1~1 etc)
    fixed = getattr(cfg, "DF_GWR_FIXED_VLIM", None)
    if fixed is not None:
        try:
            fv = float(abs(fixed))
            if np.isfinite(fv) and fv > 0:
                vmax_by_var[:] = fv
        except Exception:
            pass

    # layout: add one extra bottom row for per-column colorbars
    nrows = len(rows)
    fig = plt.figure(figsize=(2.55 * m + 1.6, 2.05 * (nrows + 1)), dpi=cfg.dpi)
    gs = fig.add_gridspec(
        nrows=nrows + 1,
        ncols=m,
        height_ratios=[1] * nrows + [0.10],
        wspace=0.10,
        hspace=0.18
    )

    # draw panels
    for r, (row_title, arr) in enumerate(rows):
        for j in range(m):
            ax = fig.add_subplot(gs[r, j])
            norm = Normalize(vmin=-vmax_by_var[j], vmax=vmax_by_var[j])

            ax.scatter(
                coords_std[:, 0], coords_std[:, 1],
                c=arr[:, j],
                s=s_map,
                cmap=cmap,
                norm=norm,
                linewidth=0,
                alpha=a_map,
                rasterized=True
            )

            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.8)
                sp.set_color("#222222")

            if r == 0:
                ax.set_title(var_names[j], fontweight="bold")
            if j == 0:
                ax.set_ylabel(row_title, fontweight="bold", rotation=90, labelpad=16)

    # per-column colorbar (accurate because each column has one fixed scale)
    for j in range(m):
        cax = fig.add_subplot(gs[-1, j])
        norm = Normalize(vmin=-vmax_by_var[j], vmax=vmax_by_var[j])
        smap = cm.ScalarMappable(norm=norm, cmap=cmap)
        smap.set_array([])
        cb = fig.colorbar(smap, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=9)
        # fewer ticks for cleanliness
        cb.set_ticks([-vmax_by_var[j], 0.0, vmax_by_var[j]])
        cax.set_xlabel("value", fontsize=9)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)



# ==============================================================================
# 6) EVALUATION / CV / SWEEP
# ==============================================================================
def fit_baselines_in_sample(data, edges_full, cfg: CFG, gwr_bw):
    # in-sample uses globally standardized data for clean figures
    X = data["X_std"]; y = data["y_std"]; coords = data["coords_std"]
    preds = {}

    preds["OLS"] = sm.OLS(y, sm.add_constant(X)).fit().predict(sm.add_constant(X))
    preds["DT"] = DecisionTreeRegressor(max_depth=8, random_state=cfg.seed).fit(X, y).predict(X)
    preds["RF"] = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=cfg.seed).fit(X, y).predict(X)
    preds["GBR"] = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=cfg.seed).fit(X, y).predict(X)

    gwr = FastGWR(coords, X, y, gwr_bw)
    preds["GWR"], gwr_coeffs = gwr.predict(coords, X, return_coeffs=True)

    # torch
    mask_all = np.ones(len(y), dtype=bool)
    data_dict = {"X": X, "y": y, "coords": coords}
    yhat_dnn, model_dnn = train_torch(None, data_dict, mask_all, edges_full, cfg, model_name="DNN")
    preds["DNN"] = yhat_dnn

    yhat_gnn, model_gnn = train_torch(None, data_dict, mask_all, edges_full, cfg, model_name="GNN")
    preds["GNN"] = yhat_gnn

    yhat_z, model_z = train_torch(None, data_dict, mask_all, edges_full, cfg, model_name="Zentropy")
    preds["Zentropy"] = yhat_z

    return preds, gwr, gwr_coeffs, model_dnn, model_gnn, model_z


def _fold_scaled_data(X_raw, y_raw, coords_std, tr_idx, te_idx):
    # fold-wise scaling to avoid leakage + stabilize baselines
    xsc = StandardScaler().fit(X_raw[tr_idx])
    ysc = StandardScaler().fit(y_raw[tr_idx].reshape(-1, 1))

    X_tr = xsc.transform(X_raw[tr_idx])
    X_te = xsc.transform(X_raw[te_idx])

    y_tr = ysc.transform(y_raw[tr_idx].reshape(-1, 1)).ravel()
    y_te = ysc.transform(y_raw[te_idx].reshape(-1, 1)).ravel()

    # full arrays on standardized scale of the fold (for torch convenience)
    X_all = np.empty_like(X_raw, dtype=float)
    y_all = np.empty_like(y_raw, dtype=float)
    X_all[tr_idx] = X_tr
    X_all[te_idx] = X_te
    y_all[tr_idx] = y_tr
    y_all[te_idx] = y_te
    return X_tr, y_tr, X_te, y_te, X_all, y_all


def run_cv_all_models(data, cfg: CFG, edges_full, gwr_bw):
    n = len(data["y_raw"])
    X_raw = data["X_raw"]
    y_raw = data["y_raw"]
    coords_std = data["coords_std"]
    blocks = _make_blocks(data["coords"], cfg.block_nx, cfg.block_ny)

    kf = KFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    gkf = GroupKFold(n_splits=cfg.cv_folds)

    cv_long = []
    oof_preds_by_cv = {}
    model_list = ["OLS", "GWR", "DT", "RF", "GBR", "DNN", "GNN", "Zentropy"]

    for cv_type, splits in [
        ("Random", kf.split(X_raw)),
        ("Block", gkf.split(X_raw, groups=blocks)),
    ]:
        oof = {m: np.full(n, np.nan, dtype=float) for m in model_list}
        fold_id = 0

        for tr, te in splits:
            fold_id += 1
            train_mask = np.zeros(n, dtype=bool); train_mask[tr] = True
            edges_fold = disconnect_train_test_edges(edges_full, train_mask)

            X_tr, y_tr, X_te, y_te, X_all, y_all = _fold_scaled_data(X_raw, y_raw, coords_std, tr, te)

            # OLS
            ols = sm.OLS(y_tr, sm.add_constant(X_tr)).fit()
            oof["OLS"][te] = ols.predict(sm.add_constant(X_te))

            # Trees / ensembles
            dt = DecisionTreeRegressor(max_depth=8, random_state=cfg.seed).fit(X_tr, y_tr)
            oof["DT"][te] = dt.predict(X_te)

            rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=cfg.seed).fit(X_tr, y_tr)
            oof["RF"][te] = rf.predict(X_te)

            gbr = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=cfg.seed).fit(X_tr, y_tr)
            oof["GBR"][te] = gbr.predict(X_te)

            # GWR (fit on train only)
            bw = int(np.clip(gwr_bw, cfg.gwr_bw_min, len(tr) - 1))
            gwr = FastGWR(coords_std[tr], X_tr, y_tr, bw)
            oof["GWR"][te] = gwr.predict(coords_std[te], X_te)

            # torch models trained on fold-standardized arrays
            fold_data = {"X": X_all, "y": y_all, "coords": coords_std}

            set_seed(cfg.seed + fold_id)
            yhat_dnn, _ = train_torch(None, fold_data, train_mask, edges_full, cfg, model_name="DNN", epochs=int(cfg.epochs_final * 0.55))
            oof["DNN"][te] = yhat_dnn[te]

            set_seed(cfg.seed + 1000 + fold_id)
            yhat_gnn, _ = train_torch(None, fold_data, train_mask, edges_fold, cfg, model_name="GNN", epochs=int(cfg.epochs_final * 0.55))
            oof["GNN"][te] = yhat_gnn[te]

            set_seed(cfg.seed + 2000 + fold_id)
            yhat_z, _ = train_torch(None, fold_data, train_mask, edges_fold, cfg, model_name="Zentropy", epochs=int(cfg.epochs_final * 0.55))
            oof["Zentropy"][te] = yhat_z[te]

            for m in model_list:
                r2 = float(r2_score(y_te, oof[m][te]))
                rmse = float(math.sqrt(mean_squared_error(y_te, oof[m][te])))
                cv_long.append({"cv_type": cv_type, "fold": fold_id, "model": m, "r2": r2, "rmse": rmse})

        for m in model_list:
            if np.any(~np.isfinite(oof[m])):
                raise RuntimeError(f"[{cv_type}] OOF has NaNs for model={m}.")
        oof_preds_by_cv[cv_type] = oof

    return pd.DataFrame(cv_long), oof_preds_by_cv


def _cv_stats_from_folds(rows, folds):
    r2 = np.array([r["r2"] for r in rows], dtype=float)
    rmse = np.array([r["rmse"] for r in rows], dtype=float)
    mi = np.array([r.get("moranI", np.nan) for r in rows], dtype=float)
    bic = np.array([r.get("bic", np.nan) for r in rows], dtype=float)
    sil = np.array([r.get("silhouette", np.nan) for r in rows], dtype=float)

    out = {
        "r2_mean": float(np.nanmean(r2)),
        "r2_std": float(np.nanstd(r2, ddof=1)) if len(r2) > 1 else 0.0,
        "rmse_mean": float(np.nanmean(rmse)),
        "rmse_std": float(np.nanstd(rmse, ddof=1)) if len(rmse) > 1 else 0.0,
        "moranI_mean": float(np.nanmean(mi)),
        "moranI_std": float(np.nanstd(mi, ddof=1)) if np.sum(np.isfinite(mi)) > 1 else 0.0,
        "bic_mean": float(np.nanmean(bic)),
        "bic_std": float(np.nanstd(bic, ddof=1)) if np.sum(np.isfinite(bic)) > 1 else 0.0,
        "silhouette_mean": float(np.nanmean(sil)),
        "silhouette_std": float(np.nanstd(sil, ddof=1)) if np.sum(np.isfinite(sil)) > 1 else 0.0,
    }
    out["r2_se"] = out["r2_std"] / math.sqrt(max(1, folds))
    out["rmse_se"] = out["rmse_std"] / math.sqrt(max(1, folds))
    out["moranI_se"] = out["moranI_std"] / math.sqrt(max(1, folds))
    out["bic_se"] = out["bic_std"] / math.sqrt(max(1, folds))
    out["silhouette_se"] = out["silhouette_std"] / math.sqrt(max(1, folds))
    return out


def eval_zentropy_block_cv(data, cfg_eval: CFG, return_fold_details=False):

    X_raw = data.get("X_raw", data["X_std"])
    y_raw = data.get("y_raw", data["y_std"])
    coords_std = data.get("coords_std", std_z(data["coords"]))

    n = len(y_raw)
    blocks = _make_blocks(data["coords"], cfg_eval.block_nx, cfg_eval.block_ny)
    gkf = GroupKFold(n_splits=cfg_eval.cv_folds)

    edges_full = make_knn_edges(coords_std, cfg_eval.graph_k)

    fold_rows = []
    fold_id = 0
    for tr, te in gkf.split(X_raw, groups=blocks):
        fold_id += 1
        train_mask = np.zeros(n, dtype=bool); train_mask[tr] = True
        edges_fold = disconnect_train_test_edges(edges_full, train_mask)


        X_tr, y_tr, X_te, y_te, X_all, y_all = _fold_scaled_data(
            X_raw, y_raw, coords_std, tr, te
        )
        fold_data = {"X": X_all, "y": y_all, "coords": coords_std}

        set_seed(cfg_eval.seed + 70000 + fold_id + 31 * int(cfg_eval.graph_k) + 101 * int(cfg_eval.z_K))
        yhat, model = train_torch(None, fold_data, train_mask, edges_fold, cfg_eval,
                                  model_name="Zentropy", epochs=160)

        yhat_te = yhat[te]
        resid = (y_te - yhat_te)

        r2 = float(r2_score(y_te, yhat_te))
        rmse = float(math.sqrt(mean_squared_error(y_te, yhat_te)))

        mi, mip = moran_residuals(resid, coords_std[te], k=cfg_eval.moran_knn)

        rss = float(np.sum(resid ** 2))
        n_te = int(len(te))
        k_params = int(sum(p.numel() for p in model.parameters()))
        bic = float(n_te * np.log((rss / max(1, n_te)) + 1e-12) + k_params * np.log(max(2, n_te)))

        sil = np.nan
        try:
            _yhat_all, p_all, *_ = zentropy_internals(model, fold_data, edges_fold, cfg_eval)
            p_te = p_all[te]
            lab = np.argmax(p_te, axis=1)
            if (len(np.unique(lab)) >= 2) and (len(lab) > len(np.unique(lab))):
                sil = float(silhouette_score(p_te, lab, metric="euclidean"))
        except Exception:
            sil = np.nan

        fold_rows.append({
            "fold": fold_id, "r2": r2, "rmse": rmse,
            "moranI": float(mi), "moran_p": float(mip),
            "bic": bic, "silhouette": sil,
        })

    return fold_rows if return_fold_details else fold_rows


def zentropy_full_grid_search_methodology(data, cfg: CFG):
    grid = list(itertools.product(
        list(cfg.GRAPH_K_CANDIDATES),
        list(cfg.LAMBDA_SPARSE_LIST),
        list(cfg.LAMBDA_MAGNITUDE_LIST),
    ))

    rows = []
    best_r2 = -np.inf
    best_se = np.inf

    for (gk, ls, lm) in grid:
        tmp = deepcopy(cfg)
        tmp.graph_k = int(gk)
        tmp.z_K = int(cfg.K_UPPER)
        tmp.lam_sparse = float(ls)
        tmp.lam_mag = float(lm)

        fold_rows = eval_zentropy_block_cv(data, tmp)
        stats = _cv_stats_from_folds(fold_rows, cfg.cv_folds)

        row = {
            "graph_k": int(gk), "K_upper": int(cfg.K_UPPER),
            "lam_sparse": float(ls), "lam_mag": float(lm),
            "r2_mean": stats["r2_mean"], "r2_se": stats["r2_se"],
            "rmse_mean": stats["rmse_mean"], "rmse_se": stats["rmse_se"],
            "moranI_mean": stats.get("moranI_mean", np.nan),
            "bic_mean": stats.get("bic_mean", np.nan),
            "silhouette_mean": stats.get("silhouette_mean", np.nan),
        }
        rows.append(row)
        if row["r2_mean"] > best_r2:
            best_r2 = row["r2_mean"]
            best_se = row["r2_se"]
        print(f"[GRID] k={int(gk):2d} K_upper={int(cfg.K_UPPER)} ls={ls} lm={lm} | "
              f"R2={row['r2_mean']:.4f}±{row['r2_se']:.4f} | RMSE={row['rmse_mean']:.4f} | "
              f"MoranI={row['moranI_mean'] if np.isfinite(row['moranI_mean']) else np.nan:.4f}")

    sweep_df = pd.DataFrame(rows)
    thr = best_r2 - best_se
    cand = sweep_df[sweep_df["r2_mean"] >= thr].copy()
    if cand.empty:
        cand = sweep_df.copy()

    cand["reg_total"] = cand["lam_sparse"] + cand["lam_mag"]
    cand["moranI_sort"] = cand["moranI_mean"].fillna(np.inf)
    cand["bic_sort"] = cand["bic_mean"].fillna(np.inf)
    cand["sil_sort"] = cand["silhouette_mean"].fillna(-np.inf)
    cand = cand.sort_values(by=["graph_k", "reg_total", "moranI_sort", "bic_sort", "sil_sort"],
                            ascending=[True, True, True, True, False])

    best_row = cand.iloc[0].to_dict()
    best_cfg = deepcopy(cfg)
    best_cfg.graph_k = int(best_row["graph_k"])
    best_cfg.K_UPPER = int(cfg.K_UPPER)
    best_cfg.lam_sparse = float(best_row["lam_sparse"])
    best_cfg.lam_mag = float(best_row["lam_mag"])
    print(f"[GRID DONE] best_mean={best_r2:.4f}, 1SE={best_se:.4f}, threshold={thr:.4f} -> "
          f"selected: k={best_cfg.graph_k}, K_upper={best_cfg.K_UPPER}, ls={best_cfg.lam_sparse}, lm={best_cfg.lam_mag}")

    sweep_df = sweep_df.sort_values(["r2_mean", "moranI_mean"], ascending=[False, True]).reset_index(drop=True)
    df_k = (sweep_df.sort_values(["graph_k", "r2_mean", "moranI_mean"], ascending=[True, False, True])
                    .groupby("graph_k", as_index=False).first().rename(columns={"graph_k":"k"}))
    df_lam = sweep_df[sweep_df["graph_k"] == best_cfg.graph_k].copy()
    return best_cfg, sweep_df, df_k, df_lam


# ==============================================================================
# 7) MAIN

# ==============================================================================
# 7) MAIN
# ==============================================================================
def main():
    cfg = CFG()
    apply_paper_style(cfg)
    set_seed(cfg.seed)

    OUT = cfg.out_dir
    DIR_DATA = os.path.join(OUT, "DATA")
    DIR_TABLES = os.path.join(OUT, "TABLES")
    DIR_FIG = os.path.join(OUT, "FIGURES")
    DIR_MODELS = os.path.join(OUT, "MODELS")
    for d in [OUT, DIR_DATA, DIR_TABLES, DIR_FIG, DIR_MODELS]:
        ensure_dir(d)

    print("============================================================")
    print("ZENTROPY FINAL PIPELINE (SINGLE FILE, ALL FIGURES)")
    print("============================================================")

    print("\n[1/6] Generating simulation dataset...")
    data = generate_simulation_linear(cfg)

    var_names = [f"x{i+1}" for i in range(cfg.p)]

    # Save dataset snapshot
    df_data = pd.DataFrame(data["X_std"], columns=var_names)
    df_data["y_std"] = data["y_std"]
    df_data["true_regime"] = data["regimes_true"]
    df_data["x"] = data["coords"][:, 0]
    df_data["y"] = data["coords"][:, 1]
    df_data.to_csv(os.path.join(DIR_DATA, "SimData.csv"), index=False)

    np.save(os.path.join(DIR_DATA, "True_dF.npy"), data["dF_true"])
    np.save(os.path.join(DIR_DATA, "True_dE.npy"), data["dE_true"])
    np.save(os.path.join(DIR_DATA, "True_dS.npy"), data["dS_true"])

    best_cfg = None
    sweep_df = None
    df_k = None
    df_lam = None

    if cfg.do_sweep:
        print("\n[2/6] Zentropy hyperparameter selection (Spatial Block CV; 1-SE)...")
        best_cfg, sweep_df, df_k, df_lam = zentropy_full_grid_search_methodology(data, cfg)
        sweep_df.to_csv(os.path.join(DIR_TABLES, "Zentropy_Sweep_FullGrid_BlockCV.csv"), index=False)
        df_k.to_csv(os.path.join(DIR_TABLES, "Zentropy_Select_graph_k_BlockCV.csv"), index=False)
        df_lam.to_csv(os.path.join(DIR_TABLES, "Zentropy_Select_Lambdas_BlockCV.csv"), index=False)

        cfg.graph_k = best_cfg.graph_k
        cfg.K_UPPER = int(best_cfg.K_UPPER)
        cfg.z_K = int(cfg.K_UPPER)
        cfg.lam_sparse = float(best_cfg.lam_sparse)
        cfg.lam_entropy = 0.0
        cfg.lam_balance = float(best_cfg.lam_sparse)
        cfg.lam_mag = float(best_cfg.lam_mag)

        with open(os.path.join(DIR_TABLES, "Best_Zentropy_Config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "graph_k": cfg.graph_k,
                "z_K": cfg.K_UPPER,
                "lam_entropy": cfg.lam_entropy,
                "lam_sparse": cfg.lam_sparse,
                "lam_balance": cfg.lam_balance,
                "lam_mag": cfg.lam_mag,
                "selection_rule": "Block-CV + 1-SE on mean R²; smallest k within 1-SE; smaller total regularization among eligible lambdas",
            }, f, indent=2)
        print(f"[SELECT DONE] k={cfg.graph_k}, K_upper={cfg.K_UPPER}, ls={cfg.lam_sparse}, lm={cfg.lam_mag}")
    else:
        print("\n[2/6] Sweep disabled. Using defaults.")

    edges_full = make_knn_edges(data["coords_std"], cfg.graph_k)

    print("\n[3/6] Selecting GWR bandwidth ONCE...")
    gwr_bw = select_gwr_bandwidth_once(data["coords_std"], data["X_std"], data["y_std"], cfg)
    with open(os.path.join(DIR_TABLES, "GWR_Bandwidth.json"), "w", encoding="utf-8") as f:
        json.dump({"gwr_bw": gwr_bw}, f, indent=2)
    print(f"[GWR] selected bw = {gwr_bw}")

    print("\n[4/6] Running CV benchmarking (Random + Spatial Block; leakage-free graphs; fold-wise scaling)...")
    cv_long, _ = run_cv_all_models(data, cfg, edges_full, gwr_bw)
    cv_long.to_csv(os.path.join(DIR_TABLES, "CV_FoldLevel_AllModels.csv"), index=False)

    cv_summary = cv_long.groupby(["cv_type", "model"], as_index=False).agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std")
    ).fillna(0.0)
    cv_summary.to_csv(os.path.join(DIR_TABLES, "CV_Summary_AllModels.csv"), index=False)

    print("\n[5/6] Fitting in-sample models (for figures + diagnostics)...")
    preds_in, gwr_full, gwr_coeffs, model_dnn, model_gnn, model_z = fit_baselines_in_sample(data, edges_full, cfg, gwr_bw)

    # save models
    torch.save(model_dnn.state_dict(), os.path.join(DIR_MODELS, "DNN_in_sample.pt"))
    torch.save(model_gnn.state_dict(), os.path.join(DIR_MODELS, "GNN_in_sample.pt"))
    torch.save(model_z.state_dict(), os.path.join(DIR_MODELS, "Zentropy_in_sample.pt"))

    # Zentropy internals (in-sample)
    data_dict_in = {"X": data["X_std"], "y": data["y_std"], "coords": data["coords_std"]}
    yhat_z, p, F_k, E_k, S_k, E_pred, S_pred, unc = zentropy_internals(model_z, data_dict_in, edges_full, cfg)
    pred_reg = np.argmax(p, axis=1)
    unc = np.clip(unc, 0, 1)
    mean_p = p.mean(0)

    F_pred = yhat_z

    F_pred_std = std_z(F_pred)
    E_pred_std = std_z(E_pred)
    S_pred_std = std_z(S_pred)

    # ===========================
    # Zentropy gradients (these are already dy_std/dx_std because model is trained on y_std, X_std)
    # ===========================
    dF_zen, dE_zen, dS_zen = zentropy_grads(model_z, data_dict_in, edges_full, cfg)

    # ===========================




    # ===========================
    y_scale = float(np.nanstd(data["y_raw"]) + 1e-12)

    dY_true = data["dF_true"] / y_scale
    dE_true = data["dE_true"] / y_scale
    dS_true = data["dS_true"] / y_scale

    dY_zen = dF_zen  # rename for clarity (same array)


    np.save(os.path.join(DIR_DATA, "True_dY_std.npy"), dY_true)

    # save arrays
    np.save(os.path.join(DIR_DATA, "Zentropy_p.npy"), p)
    np.save(os.path.join(DIR_DATA, "Zentropy_dF.npy"), dF_zen)
    np.save(os.path.join(DIR_DATA, "Zentropy_dE.npy"), dE_zen)
    np.save(os.path.join(DIR_DATA, "Zentropy_dS.npy"), dS_zen)
    np.save(os.path.join(DIR_DATA, "GWR_coeffs.npy"), gwr_coeffs)

    print("\n[6/6] Generating ALL figures (no silent skipping; errors become error-PNGs)...")

    safe_run_figure(fig_B_true_overview_paper, os.path.join(DIR_FIG, "FIG_B_TrueOverview.png"),
                    data["coords_std"], data["regimes_true"], data["y_std"], cfg)
    safe_run_figure(fig_B_yx_scatter_panel, os.path.join(DIR_FIG, "FIG_B_y_vs_x_scatter_TRUE.png"),
                    data["y_std"], data["X_std"], data["regimes_true"], cfg, var_names=var_names, show_trends=False)

    safe_run_figure(fig_B_dashboard_paper, os.path.join(DIR_FIG, "FIG_B2_ZentropyDashboard.png"),
                    data["coords_std"], data["regimes_true"], pred_reg, unc, mean_p,
                    data["y_std"], F_pred_std, E_pred_std, S_pred_std, cfg)

    if cfg.do_sweep and df_k is not None:
        safe_run_figure(fig_E_sweep_k_curve, os.path.join(DIR_FIG, "FIG_E_Select_graph_k_curve.png"), df_k, cfg)
    if cfg.do_sweep and df_lam is not None:
        safe_run_figure(fig_E_sweep_lambda_heatmaps, os.path.join(DIR_FIG, "FIG_E_Select_lambda_heatmaps.png"), df_lam, cfg)

    safe_run_figure(fig_C_true_vs_pred_FES, os.path.join(DIR_FIG, "FIG_C_TrueVsPred_FES.png"),
                    data["coords_std"], data["F_true_std"], data["E_true_std"], data["S_true_std"],
                    F_pred_std, E_pred_std, S_pred_std, cfg)

    safe_run_figure(fig_D_thermo_diagnostics, os.path.join(DIR_FIG, "FIG_D_Thermodynamic_Diagnostics.png"),
                    data["dF_true"], dF_zen, dE_zen, dS_zen, var_names, cfg)

    safe_run_figure(fig_F_cv_benchmark, os.path.join(DIR_FIG, "FIG_F_CV_Benchmark.png"), cv_summary, cfg)
    safe_run_figure(fig_F_validation_importance_2x2, os.path.join(DIR_FIG, "FIG_F_validation_importance_2x2.png"),
                    cv_long, data["dF_true"], dF_zen, var_names, cfg)

    # Residual diagnostics (returns df)
    try:
        df_moran = fig_G_residual_diagnostics(data["coords_std"], data["y_std"], preds_in,
                                             os.path.join(DIR_FIG, "FIG_G_residual_diagnostics_all_models.png"), cfg)
        df_moran.to_csv(os.path.join(DIR_TABLES, "MoranResiduals_InSample_AllModels.csv"), index=False)
    except Exception as e:
        _safe_save_error_figure(os.path.join(DIR_FIG, "FIG_G_residual_diagnostics_all_models.png"), "fig_G_residual_diagnostics", e)

    # ===========================

    # ===========================
    safe_run_figure(
        fig_H2_sensitivity_maps_combined,
        os.path.join(DIR_FIG, "FIG_H2_SensitivityMaps_True_Zentropy_FES_vs_GWR.png"),
        data["coords_std"], var_names,
        dY_true, dE_true, dS_true,
        dY_zen,  dE_zen,  dS_zen,
        gwr_coeffs, cfg, max_vars=cfg.SENS_MAX_VARS
    )

    print("\n[ALL DONE] Outputs saved under:")
    print(f" - {OUT}")
    print("Key outputs:")
    print(" - DATA/SimData.csv")
    print(" - TABLES/CV_Summary_AllModels.csv")
    print(" - FIGURES/ (all figures)")
    if not _HAVE_MORAN:
        print("\n[NOTE] libpysal/esda not installed -> Moran's I will be NA (fig still produced).")


if __name__ == "__main__":
    main()
