"""
caspect_simulation.py
=====================
Simulation study for CaSPECT (CSC-PC pipeline).

Reproduces all results in Section 6 of the paper:
  - Setting S1: Clean linear DAG, non-Gaussian errors
  - Setting S2: Mixed linearity (three nonlinear edges, RESET routing)
  - Setting S3: Mild causal sufficiency violation (latent confounder)
  - Ablations A1–A3 (within S1 at n=1000)

Dependencies
------------
  numpy, scipy, pandas, scikit-learn, statsmodels, networkx,
  causal-learn, lingam, pygam, matplotlib, tabulate

Run
---
  python caspect_simulation.py

Results are printed as LaTeX-style tables and saved to
  simulation_results.csv
"""
import itertools
import warnings
import time
from copy import deepcopy
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from scipy import stats
from scipy.linalg import eig

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera as jb_test

warnings.filterwarnings("ignore")

# ═════════════════════════════════════════════════════════════════════════════
#  OPTIONAL IMPORTS  (graceful fallback if not installed)
# ═════════════════════════════════════════════════════════════════════════════
try:
    from pygam import LinearGAM, s as gam_s
    HAS_GAM = True
except ImportError:
    HAS_GAM = False

try:
    from lingam import DirectLiNGAM
    HAS_LINGAM = True
except ImportError:
    HAS_LINGAM = False

try:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    HAS_CAUSALLEARN = True
except ImportError:
    HAS_CAUSALLEARN = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ─ DATA GENERATING PROCESS
# ═════════════════════════════════════════════════════════════════════════════

def sample_dag(q: int, p_edge: float = 0.35, seed: int = 0) -> np.ndarray:
    """
    Random DAG adjacency matrix.
    Draw a uniformly random topological ordering; include each upper-triangle
    edge with probability p_edge.  Returns binary adjacency A where A[u,v]=1
    means u → v.
    """
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(q)          # random topological ordering
    A    = np.zeros((q, q), dtype=int)
    for i in range(q):
        for j in range(i + 1, q):
            if rng.random() < p_edge:
                u, v      = perm[i], perm[j]
                A[u, v]   = 1
    return A


def sample_coefficients(A: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    Draw structural coefficients β_uv ~ Uniform(0.3, 0.8) × Rademacher.
    Returns a float matrix of the same shape as A.
    """
    rng  = np.random.default_rng(seed)
    q    = A.shape[0]
    B    = np.zeros((q, q))
    mask = A > 0
    n_e  = mask.sum()
    mags = rng.uniform(0.3, 0.8, size=n_e)
    signs = rng.choice([-1, 1],   size=n_e)
    B[mask] = mags * signs
    return B


def topological_order(A: np.ndarray) -> list[int]:
    """Kahn's algorithm for topological sort of DAG adjacency matrix."""
    G   = nx.DiGraph(A)
    return list(nx.topological_sort(G))


def generate_data(
    n:          int,
    A:          np.ndarray,
    B:          np.ndarray,
    treat_idx:  int,
    outcome_idx:int,
    error_dist: str  = "t5",          # "t5" | "gaussian"
    nonlinear_edges: Optional[list[tuple[int,int]]] = None,
    latent_gamma: float = 0.0,        # γ for latent confounder (S3)
    latent_targets: Optional[tuple[int,int]] = None,
    alpha_clusters: tuple = (-1.0, 0.0, 1.0),
    seed:       int  = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate data from the SEM described in Section 6.1.1.

    Returns
    -------
    X        : (n, q) data matrix
    clusters : (n,)  ground-truth cluster labels in {0,1,2}
    true_ace  : (3,)  true cluster-level ACE  τ_c^0
    """
    rng  = np.random.default_rng(seed)
    q    = A.shape[0]
    order = topological_order(A)

    # ── Cluster assignment via treatment intercept shift ───────────────────────
    # Assign each unit to a cluster first; intercept is applied during SEM.
    base_cluster = np.repeat([0, 1, 2], [n // 3, n // 3, n - 2 * (n // 3)])
    rng.shuffle(base_cluster)

    # ── Latent confounder (S3) ────────────────────────────────────────────────
    H = rng.standard_normal(n) if latent_gamma > 0 else np.zeros(n)

    # ── Error draws ───────────────────────────────────────────────────────────
    if error_dist == "t5":
        E_raw = rng.standard_t(5, size=(n, q))
        # Normalise variance to 1
        E = E_raw / np.sqrt(5 / 3)
    else:
        E = rng.standard_normal(size=(n, q))

    X = np.zeros((n, q))

    # ── Recursive SEM ─────────────────────────────────────────────────────────
    for v in order:
        parents = np.where(A[:, v] > 0)[0]
        Xv = E[:, v].copy()

        # Latent confounder contribution
        if latent_gamma > 0 and latent_targets and v in latent_targets:
            Xv += latent_gamma * H

        for u in parents:
            if nonlinear_edges and (u, v) in nonlinear_edges:
                Xv += B[u, v] * np.sin(np.pi * X[:, u])
            else:
                Xv += B[u, v] * X[:, u]

        # Treatment node gets cluster-specific intercept
        if v == treat_idx:
            treat_parents = np.where(A[:, treat_idx] > 0)[0]
            parent_contrib = sum((B[p, treat_idx] * X[:, p] for p in treat_parents),
                                 np.zeros(n))   # np.zeros start: safe when Z has no parents
            Xv = np.zeros(n)
            for c_idx, alpha in enumerate(alpha_clusters):
                mask = base_cluster == c_idx
                Xv[mask] = alpha + parent_contrib[mask] + E[mask, v]
            if latent_gamma > 0 and latent_targets and v in latent_targets:
                Xv += latent_gamma * H

        X[:, v] = Xv

    # ── True cluster-level ACE ─────────────────────────────────────────────────
    # τ_c^0 = β_{Z→Y} + sum of indirect path products
    # For linear SEM: this is the total causal effect of Z on Y.
    # We estimate it analytically as the (treat→outcome) entry of
    # the path-coefficient matrix (I - B)^{-1}.
    try:
        I_minus_B    = np.eye(q) - B
        total_effects = np.linalg.inv(I_minus_B)
        base_ace     = total_effects[treat_idx, outcome_idx]
    except np.linalg.LinAlgError:
        base_ace = B[treat_idx, outcome_idx]

    # Cluster-level ACE differs because covariate distributions shift
    true_ace = np.array([base_ace * (1 + 0.1 * (c - 1)) for c in range(3)])

    return X, base_cluster, true_ace


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ─ MINI PC ALGORITHM  (fast fallback)
# ═════════════════════════════════════════════════════════════════════════════

def _partial_corr(X: np.ndarray, i: int, j: int, S: list[int]) -> float:
    """Partial correlation of X[:,i] and X[:,j] given X[:,S]."""
    if not S:
        r  = np.corrcoef(X[:, i], X[:, j])[0, 1]
        return r
    idx = [i, j] + list(S)
    C   = np.corrcoef(X[:, idx].T)
    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return 0.0
    r   = -C_inv[0, 1] / np.sqrt(C_inv[0, 0] * C_inv[1, 1] + 1e-12)
    return float(np.clip(r, -1 + 1e-9, 1 - 1e-9))


def _fisher_z(r: float, n: int) -> float:
    z = 0.5 * np.log((1 + r) / (1 - r))
    return abs(z) * np.sqrt(n - 3)


def pc_algorithm_simple(
    X:     np.ndarray,
    alpha: float = 0.05,
    max_cond: int = 3,
) -> np.ndarray:
    """
    Simplified PC algorithm using Fisher-Z partial correlation tests.
    Returns binary adjacency matrix (may contain 1s in both directions for
    undirected edges).
    """
    n, q    = X.shape
    adj     = np.ones((q, q), dtype=int) - np.eye(q, dtype=int)
    sep_set = {(i, j): [] for i in range(q) for j in range(q)}

    # ── Skeleton phase ────────────────────────────────────────────────────────
    for size in range(max_cond + 1):
        for i in range(q):
            for j in range(i + 1, q):
                if adj[i, j] == 0:
                    continue
                neighbours_i = [k for k in range(q)
                                 if k != j and adj[i, k]]
                if len(neighbours_i) < size:
                    continue
                for S in itertools.combinations(neighbours_i, size):
                    S = list(S)
                    r = _partial_corr(X, i, j, S)
                    z = _fisher_z(r, n)
                    p = 2 * (1 - stats.norm.cdf(z))
                    if p > alpha:
                        adj[i, j] = adj[j, i] = 0
                        sep_set[(i, j)] = sep_set[(j, i)] = S
                        break

    # ── V-structure orientation ───────────────────────────────────────────────
    directed = adj.copy()
    for i in range(q):
        for j in range(i + 1, q):
            if adj[i, j] == 0:
                continue
            for k in range(q):
                if k == i or k == j:
                    continue
                if adj[i, k] and adj[j, k] and not adj[i, j]:
                    if i not in sep_set[(i, k)] and j not in sep_set[(j, k)]:
                        directed[k, i] = directed[k, j] = 0

    return directed


def run_pc(X: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Run PC algorithm, using causal-learn if available, else fallback."""
    if HAS_CAUSALLEARN:
        try:
            result = pc(X, indep_test=fisherz, alpha=alpha,
                        uc_rule=0, uc_priority=2,
                        mvpc=False, verbose=False, show_progress=False)
            G   = result.G.graph
            d   = G.shape[0]
            adj = np.zeros((d, d), dtype=int)
            for i in range(d):
                for j in range(d):
                    if G[i, j] == -1 and G[j, i] == -1:
                        adj[i, j] = adj[j, i] = 1
                    elif G[i, j] == 1 and G[j, i] == -1:
                        adj[i, j] = 1
                    elif G[i, j] == -1 and G[j, i] == 1:
                        adj[j, i] = 1
            return adj
        except Exception:
            pass
    return pc_algorithm_simple(X, alpha=alpha)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ─ BOOTSTRAP STABILITY
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_pc(
    X:     np.ndarray,
    B:     int   = 100,
    alpha: float = 0.05,
    theta: float = 0.50,
    seed:  int   = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap PC: returns f_uv, g_uv, rho."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    f_count = np.zeros((d, d))
    g_count = np.zeros((d, d))

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            G = run_pc(X[idx], alpha=alpha)
        except Exception:
            continue
        for i in range(d):
            for j in range(i + 1, d):
                if G[i, j] or G[j, i]:
                    f_count[i, j] += 1
                    f_count[j, i] += 1
                if G[i, j] == 1 and G[j, i] == 0:
                    g_count[i, j] += 1
                elif G[j, i] == 1 and G[i, j] == 0:
                    g_count[j, i] += 1

    f_uv = f_count / B
    g_uv = g_count / B
    with np.errstate(invalid="ignore"):
        rho = np.where(f_uv > 0, g_uv / f_uv, 0.0)
    return f_uv, g_uv, rho


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ─ OVS
# ═════════════════════════════════════════════════════════════════════════════

def run_lingam(X: np.ndarray) -> np.ndarray:
    if HAS_LINGAM:
        try:
            model = DirectLiNGAM()
            model.fit(X)
            return model.adjacency_matrix_
        except Exception:
            pass
    # Fallback: pairwise linear regression sign heuristic
    q = X.shape[1]
    B = np.zeros((q, q))
    for i in range(q):
        for j in range(q):
            if i != j:
                b = np.linalg.lstsq(X[:, [i]], X[:, j], rcond=None)[0][0]
                B[i, j] = b
    return B


def _jb_fraction(X: np.ndarray, alpha: float = 0.10) -> float:
    n, q = X.shape
    count = 0
    for j in range(q):
        others = [k for k in range(q) if k != j]
        Xo = sm.add_constant(X[:, others])
        try:
            resid = sm.OLS(X[:, j], Xo).fit().resid
            _, p, _, _ = jb_test(resid)
            if p < alpha:
                count += 1
        except Exception:
            pass
    return count / q


def compute_ovs(
    f_uv:  np.ndarray,
    g_uv:  np.ndarray,
    rho:   np.ndarray,
    B_hat: np.ndarray,
    w_L:   float = 0.20,
    tau:   float = 0.15,
    theta: float = 0.50,
) -> tuple[dict, np.ndarray]:
    w_pc = 1.0 - w_L
    d    = f_uv.shape[0]
    ovs_matrix   = np.zeros((d, d))
    orientations = {}

    for i in range(d):
        for j in range(i + 1, d):
            if f_uv[i, j] < theta:
                continue
            delta_pc = rho[i, j] - rho[j, i]
            delta_L  = float(np.sign(B_hat[i, j] - B_hat[j, i]))
            ovs      = w_pc * f_uv[i, j] * delta_pc + w_L * delta_L
            ovs_matrix[i, j] =  ovs
            ovs_matrix[j, i] = -ovs
            if ovs > tau:
                orientations[(i, j)] = (i, j)
            elif ovs < -tau:
                orientations[(i, j)] = (j, i)
            else:
                orientations[(i, j)] = None

    return orientations, ovs_matrix


def ovs_accuracy(
    orientations: dict,
    A_true:       np.ndarray,
    f_uv:         np.ndarray,
    theta:        float = 0.50,
) -> float:
    """Fraction of stable-skeleton edges correctly oriented by OVS."""
    correct = total = 0
    for (i, j), direction in orientations.items():
        if direction is None:
            continue
        total += 1
        u, v = direction
        if A_true[u, v] == 1 and A_true[v, u] == 0:
            correct += 1
    return correct / total if total > 0 else float("nan")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 ─ DAG RESOLUTION
# ═════════════════════════════════════════════════════════════════════════════

def resolve_dag(
    orientations: dict,
    d:            int,
    ovs_matrix:   np.ndarray,
    symmetrise:   bool = False,   # Ablation A3
) -> tuple[np.ndarray, list]:
    """
    Resolve orientation ambiguities via hierarchy.
    Returns adjacency matrix of final DAG and list of contracted pairs.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(d))
    resolved       = {}
    ambiguous      = []
    contracted     = []

    for (i, j), direction in orientations.items():
        if direction is None:
            ambiguous.append((i, j))
            continue
        u, v = direction
        G.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(u, v)
            G.add_edge(v, u)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(v, u)
                ambiguous.append((i, j))
            else:
                resolved[(i, j)] = (v, u)
        else:
            resolved[(i, j)] = direction

    for (i, j) in ambiguous:
        if symmetrise:
            # Ablation A3: symmetrisation
            G.add_edge(i, j)
            G.add_edge(j, i)
            resolved[(i, j)] = (i, j)
        else:
            # Contraction
            u_keep = min(i, j)
            u_drop = max(i, j)
            if G.has_node(u_drop):
                for pred in list(G.predecessors(u_drop)):
                    if pred != u_keep:
                        G.add_edge(pred, u_keep)
                for succ in list(G.successors(u_drop)):
                    if succ != u_keep:
                        G.add_edge(u_keep, succ)
                G.remove_node(u_drop)
            contracted.append((i, j))
            resolved[(i, j)] = (u_keep, -1)

    A_est = np.zeros((d, d), dtype=int)
    for u, v in G.edges():
        if u < d and v < d:
            A_est[u, v] = 1
    return A_est, contracted


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 6 ─ EDGE WEIGHT ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

def _reset_test(y, X_reg, alpha=0.05):
    Xc = sm.add_constant(X_reg)
    try:
        res0  = sm.OLS(y, Xc).fit()
        yhat  = res0.fittedvalues
        X_aug = np.column_stack([Xc, yhat**2, yhat**3])
        res1  = sm.OLS(y, X_aug).fit()
        F_num = (res0.ssr - res1.ssr) / 2
        F_den = res1.ssr / max(1, len(y) - X_aug.shape[1])
        p     = 1 - stats.f.cdf(F_num / (F_den + 1e-12), 2,
                                  max(1, len(y) - X_aug.shape[1]))
        return p > alpha
    except Exception:
        return True


def _ols_ate(y, t, Z):
    X_r = sm.add_constant(np.column_stack([t, Z])) if Z.shape[1] > 0 \
          else sm.add_constant(t)
    try:
        return float(sm.OLS(y, X_r).fit().params[1])
    except Exception:
        return float(LinearRegression().fit(t.reshape(-1, 1), y).coef_[0])


def _dml_ate(y, t, Z, K=5, seed=42):
    n = len(y)
    V_res = np.zeros(n)
    U_res = np.zeros(n)
    kf    = KFold(n_splits=K, shuffle=True, random_state=seed)

    for tr, te in kf.split(np.arange(n)):
        if Z.shape[1] == 0:
            V_res[te] = y[te] - y[tr].mean()
            U_res[te] = t[te] - t[tr].mean()
            continue
        if HAS_GAM:
            try:
                terms  = sum(gam_s(j, n_splines=min(20, len(tr)//10))
                             for j in range(Z.shape[1]))
                m_y = LinearGAM(terms).fit(Z[tr], y[tr])
                m_t = LinearGAM(terms).fit(Z[tr], t[tr])
                V_res[te] = y[te] - m_y.predict(Z[te])
                U_res[te] = t[te] - m_t.predict(Z[te])
                continue
            except Exception:
                pass
        rf_y = RandomForestRegressor(100, random_state=seed).fit(Z[tr], y[tr])
        rf_t = RandomForestRegressor(100, random_state=seed).fit(Z[tr], t[tr])
        V_res[te] = y[te] - rf_y.predict(Z[te])
        U_res[te] = t[te] - rf_t.predict(Z[te])

    denom = np.sum(U_res**2)
    return float(np.sum(U_res * V_res) / denom) if denom > 1e-12 else 0.0


def estimate_weights(
    X:          np.ndarray,
    A_est:      np.ndarray,
    f_uv:       np.ndarray,
    alpha_reset:float = 0.05,
    dml_k:      int   = 5,
    no_stability:bool = False,   # Ablation A2
    track_b_calls: Optional[list] = None,
) -> np.ndarray:
    n, d = X.shape
    A    = np.zeros((d, d))

    for u in range(d):
        for v in range(d):
            if A_est[u, v] == 0:
                continue
            t  = X[:, u]
            y  = X[:, v]
            pa = [w for w in range(d) if A_est[w, v] and w != u]
            Z  = X[:, pa] if pa else np.empty((n, 0))
            X_r = np.column_stack([t, Z]) if Z.shape[1] > 0 else t.reshape(-1, 1)
            linear = _reset_test(y, X_r, alpha=alpha_reset)
            if not linear and track_b_calls is not None:
                track_b_calls.append((u, v))
            ate = _ols_ate(y, t, Z) if linear else _dml_ate(y, t, Z, K=dml_k)
            fij = 1.0 if no_stability else (f_uv[u, v] if u < d and v < d else 1.0)
            A[u, v] = fij * abs(ate)
    return A


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 7 ─ CHUNG LAPLACIAN + SPECTRAL CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════

def compute_laplacian(A: np.ndarray, alpha_pr: float = 0.15) -> np.ndarray:
    d        = A.shape[0]
    row_sums = A.sum(axis=1, keepdims=True)
    safe     = np.where(row_sums == 0, 1.0, row_sums)
    P0       = A / safe
    P0[row_sums.flatten() == 0] = 1.0 / d
    P        = (1 - alpha_pr) * P0 + (alpha_pr / d) * np.ones((d, d))

    eigvals, eigvecs = eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi  = np.abs(np.real(eigvecs[:, idx]))
    pi /= pi.sum()
    pi_safe = np.where(pi > 1e-12, pi, 1e-12)

    P_star = (pi[np.newaxis, :] / pi_safe[:, np.newaxis]) * P.T
    L      = np.eye(d) - 0.5 * (P + P_star)
    L      = 0.5 * (L + L.T)
    return L


def spectral_embed_and_cluster(
    X:    np.ndarray,
    L:    np.ndarray,
    seed: int = 42,
) -> tuple[np.ndarray, int, float, int]:
    """
    Embed X via Chung Laplacian eigenvectors, select K* and k*.
    Returns cluster labels, k*, spectral gap, K*.
    """
    eigvals, eigvecs = np.linalg.eigh(L)
    order   = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    gaps    = np.diff(eigvals)

    if len(gaps) > 1:
        K_star = int(np.argmax(gaps[1:])) + 2
        K_star = max(2, min(K_star, L.shape[0] - 2))
    else:
        K_star = 1

    gap_val = gaps[K_star - 1] if K_star - 1 < len(gaps) else 0.0
    V_K     = eigvecs[:, 1:K_star + 1]
    X_embed = X @ V_K

    best_k, best_sil = 2, -1
    for k in range(2, min(K_star + 3, X.shape[0])):
        try:
            labels = KMeans(n_clusters=k, n_init=20,
                            random_state=seed).fit_predict(X_embed)
            sil = silhouette_score(X_embed, labels) if len(set(labels)) > 1 else -1
            if sil > best_sil:
                best_sil, best_k = sil, k
        except Exception:
            pass

    clusters = KMeans(n_clusters=best_k, n_init=50,
                      random_state=seed).fit_predict(X_embed)
    return clusters, best_k, float(gap_val), K_star


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 8 ─ ACE ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

def cluster_ace(
    X:           np.ndarray,
    clusters:    np.ndarray,
    treat_idx:   int,
    outcome_idx: int,
    A_est:       np.ndarray,
    k_star:      int,
) -> np.ndarray:
    """DML-based cluster-level ACE estimates."""
    n, d  = X.shape
    aces  = np.full(k_star, np.nan)

    for c in range(k_star):
        mask = clusters == c
        if mask.sum() < 20:
            continue
        Xc   = X[mask]
        t    = Xc[:, treat_idx]
        y    = Xc[:, outcome_idx]
        pa   = [w for w in range(d)
                if A_est[w, outcome_idx] and w != treat_idx]
        Z    = Xc[:, pa] if pa else np.empty((mask.sum(), 0))
        aces[c] = _dml_ate(y, t, Z)

    return aces


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 9 ─ ONE REPLICATION
# ═════════════════════════════════════════════════════════════════════════════

def one_replication(
    n:           int,
    A_true:      np.ndarray,
    B_true:      np.ndarray,
    treat_idx:   int,
    outcome_idx: int,
    true_ace:    np.ndarray,
    k_true:      int        = 3,
    error_dist:  str        = "t5",
    nonlinear_edges: Optional[list] = None,
    latent_gamma:    float  = 0.0,
    latent_targets:  Optional[tuple] = None,
    B_boot:          int    = 100,
    alpha_ci:        float  = 0.05,
    theta:           float  = 0.50,
    tau:             float  = 0.15,
    w_L_max:         float  = 0.30,
    force_pc_only:   bool   = False,    # Ablation A1
    no_stability:    bool   = False,    # Ablation A2
    symmetrise:      bool   = False,    # Ablation A3
    seed:            int    = 0,
    true_nonlinear_edges: Optional[list] = None,
) -> dict:
    rng = np.random.default_rng(seed)
    q   = A_true.shape[0]

    # ── Generate data ─────────────────────────────────────────────────────────
    X, gt_clusters, _ = generate_data(
        n, A_true, B_true, treat_idx, outcome_idx,
        error_dist=error_dist,
        nonlinear_edges=nonlinear_edges,
        latent_gamma=latent_gamma,
        latent_targets=latent_targets,
        seed=seed,
    )
    Xs = StandardScaler().fit_transform(X)

    # ── Bootstrap PC ──────────────────────────────────────────────────────────
    f_uv, g_uv, rho = bootstrap_pc(Xs, B=B_boot, alpha=alpha_ci,
                                    theta=theta, seed=seed)

    # ── LiNGAM + OVS ─────────────────────────────────────────────────────────
    if force_pc_only:
        w_L = 0.0
        B_hat = np.zeros((q, q))
    else:
        ng_frac = _jb_fraction(Xs)
        w_L     = min(ng_frac, 1.0) * w_L_max
        B_hat   = run_lingam(Xs)

    orientations, ovs_matrix = compute_ovs(
        f_uv, g_uv, rho, B_hat, w_L=w_L, tau=tau, theta=theta,
    )

    # ── OVS accuracy ─────────────────────────────────────────────────────────
    ovs_acc = ovs_accuracy(orientations, A_true, f_uv, theta=theta)

    # ── DAG resolution ───────────────────────────────────────────────────────
    A_est, contracted = resolve_dag(orientations, q, ovs_matrix,
                                    symmetrise=symmetrise)

    # ── Edge weights ─────────────────────────────────────────────────────────
    track_b_calls: list = []
    A_weighted = estimate_weights(
        Xs, A_est, f_uv,
        no_stability=no_stability,
        track_b_calls=track_b_calls,
    )

    # RESET routing metrics (for S2)
    true_track_b_rate = false_track_b_rate = float("nan")
    if true_nonlinear_edges:
        nl_set   = set(true_nonlinear_edges)
        all_edges = [(u, v) for u in range(q) for v in range(q)
                     if A_est[u, v]]
        lin_edges = [e for e in all_edges if e not in nl_set]
        true_called  = sum(1 for e in track_b_calls if e in nl_set)
        false_called = sum(1 for e in track_b_calls if e in set(lin_edges))
        true_track_b_rate  = (true_called  / max(1, len(nl_set)))
        false_track_b_rate = (false_called / max(1, len(lin_edges)))

    # ── Laplacian + Clustering ────────────────────────────────────────────────
    L = compute_laplacian(A_weighted)
    clusters, k_star, gap_val, K_star = spectral_embed_and_cluster(Xs, L, seed=seed)

    ari = adjusted_rand_score(gt_clusters, clusters)

    # ── Cluster-level ACE ─────────────────────────────────────────────────────
    aces_est = cluster_ace(Xs, clusters, treat_idx, outcome_idx, A_est, k_star)

    # Map estimated clusters to true clusters by majority vote
    ace_rmse = np.full(k_true, np.nan)
    for c_true in range(k_true):
        mask_true = gt_clusters == c_true
        if mask_true.sum() == 0:
            continue
        # Find estimated cluster with highest overlap
        est_labels_in_true = clusters[mask_true]
        if len(est_labels_in_true) == 0:
            continue
        c_est = int(stats.mode(est_labels_in_true, keepdims=True).mode[0])
        if c_est < len(aces_est) and not np.isnan(aces_est[c_est]):
            ace_rmse[c_true] = (aces_est[c_est] - true_ace[c_true]) ** 2

    ace_rmse = np.sqrt(ace_rmse)

    return {
        "ari":              ari,
        "ovs_acc":          ovs_acc,
        "gap_val":          gap_val,
        "K_star":           K_star,
        "k_star":           k_star,
        "ace_rmse":         ace_rmse,
        "true_track_b":     true_track_b_rate,
        "false_track_b":    false_track_b_rate,
        "n_contracted":     len(contracted),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 10 ─ MONTE CARLO RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_mc(
    setting:     str,
    n_list:      list[int],
    M:           int   = 200,
    q:           int   = 8,
    treat_idx:   int   = 6,
    outcome_idx: int   = 7,
    p_edge:      float = 0.35,
    B_boot:      int   = 100,
    alpha_ci:    float = 0.05,
    theta:       float = 0.50,
    tau:         float = 0.15,
    w_L_max:     float = 0.30,
    force_pc_only:bool = False,
    no_stability: bool = False,
    symmetrise:   bool = False,
    base_seed:   int   = 0,
    verbose:     bool  = True,
) -> pd.DataFrame:
    """Run M replications for each n in n_list under the specified setting."""
    records = []

    for n in n_list:
        ari_list    = []
        ovs_list    = []
        ace0_list   = []
        ace1_list   = []
        ace2_list   = []
        tb_true_list = []
        tb_false_list = []

        for m in range(M):
            seed_dag  = base_seed + m * 1000
            seed_data = base_seed + m * 1000 + 1

            A_true = sample_dag(q, p_edge=p_edge, seed=seed_dag)
            B_true = sample_coefficients(A_true, seed=seed_dag)

            # True ACE (path-tracing)
            try:
                I_B = np.eye(q) - B_true
                te  = np.linalg.inv(I_B)
                base_ace = te[treat_idx, outcome_idx]
            except Exception:
                base_ace = B_true[treat_idx, outcome_idx]
            true_ace = np.array([base_ace * (1 + 0.1*(c-1)) for c in range(3)])

            # Setting-specific arguments
            nl_edges = None
            if setting in ("S2",):
                # Three nonlinear edges = top-3 |β| edges
                edge_list = [(u, v, abs(B_true[u, v]))
                             for u in range(q) for v in range(q)
                             if A_true[u, v]]
                edge_list.sort(key=lambda x: -x[2])
                nl_edges = [(e[0], e[1]) for e in edge_list[:3]]

            lat_gamma   = 0.4 if setting == "S3" else 0.0
            lat_targets = None
            if setting == "S3":
                non_tz = [i for i in range(q)
                           if i not in (treat_idx, outcome_idx)]
                rng_lt = np.random.default_rng(seed_dag + 7)
                lt     = rng_lt.choice(non_tz, size=2, replace=False)
                lat_targets = (int(lt[0]), int(lt[1]))

            try:
                res = one_replication(
                    n=n, A_true=A_true, B_true=B_true,
                    treat_idx=treat_idx, outcome_idx=outcome_idx,
                    true_ace=true_ace,
                    error_dist="gaussian" if setting == "S0" else "t5",
                    nonlinear_edges=nl_edges,
                    latent_gamma=lat_gamma,
                    latent_targets=lat_targets,
                    B_boot=B_boot, alpha_ci=alpha_ci,
                    theta=theta, tau=tau, w_L_max=w_L_max,
                    force_pc_only=force_pc_only,
                    no_stability=no_stability,
                    symmetrise=symmetrise,
                    seed=seed_data,
                    true_nonlinear_edges=nl_edges,
                )
            except Exception as e:
                if verbose:
                    print(f"  Replication {m} failed: {e}")
                continue

            ari_list.append(res["ari"])
            ovs_list.append(res["ovs_acc"])
            rmse = res["ace_rmse"]
            ace0_list.append(rmse[0] if not np.isnan(rmse[0]) else np.nan)
            ace1_list.append(rmse[1] if not np.isnan(rmse[1]) else np.nan)
            ace2_list.append(rmse[2] if not np.isnan(rmse[2]) else np.nan)
            if not np.isnan(res["true_track_b"]):
                tb_true_list.append(res["true_track_b"])
                tb_false_list.append(res["false_track_b"])

        def ms(lst):
            a = np.array(lst)
            a = a[~np.isnan(a)]
            return np.nanmean(a), np.nanstd(a)

        row = {
            "setting":   setting,
            "n":         n,
            "M":         len(ari_list),
            "ARI_mean":  ms(ari_list)[0],  "ARI_sd":  ms(ari_list)[1],
            "OVS_mean":  ms(ovs_list)[0],  "OVS_sd":  ms(ovs_list)[1],
            "ACE0_mean": ms(ace0_list)[0], "ACE0_sd": ms(ace0_list)[1],
            "ACE1_mean": ms(ace1_list)[0], "ACE1_sd": ms(ace1_list)[1],
            "ACE2_mean": ms(ace2_list)[0], "ACE2_sd": ms(ace2_list)[1],
            "TB_true":   ms(tb_true_list)[0] if tb_true_list else np.nan,
            "TB_false":  ms(tb_false_list)[0] if tb_false_list else np.nan,
        }
        records.append(row)

        if verbose:
            print(f"  {setting} | n={n:5d} | "
                  f"ARI={row['ARI_mean']:.3f}({row['ARI_sd']:.3f}) | "
                  f"OVS={row['OVS_mean']:.3f} | "
                  f"RMSE=[{row['ACE0_mean']:.3f},{row['ACE1_mean']:.3f},{row['ACE2_mean']:.3f}]")

    return pd.DataFrame(records)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 11 ─ FORMATTING
# ═════════════════════════════════════════════════════════════════════════════

def fmt(mean, sd, decimals=2):
    if np.isnan(mean):
        return "—"
    return f"{mean:.{decimals}f} ({sd:.{decimals}f})"


def print_table(df: pd.DataFrame, title: str, setting: str):
    sub = df[df["setting"] == setting].copy()
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    header = ["n", "ARI", "OVS Acc",
              "ACE RMSE C1", "ACE RMSE C2", "ACE RMSE C3"]
    rows = []
    for _, r in sub.iterrows():
        rows.append([
            int(r["n"]),
            fmt(r["ARI_mean"], r["ARI_sd"]),
            fmt(r["OVS_mean"], r["OVS_sd"]),
            fmt(r["ACE0_mean"], r["ACE0_sd"]),
            fmt(r["ACE1_mean"], r["ACE1_sd"]),
            fmt(r["ACE2_mean"], r["ACE2_sd"]),
        ])
    if HAS_TABULATE:
        print(tabulate(rows, headers=header, tablefmt="github"))
    else:
        print("\t".join(header))
        for row in rows:
            print("\t".join(str(x) for x in row))


def print_s2_table(df: pd.DataFrame):
    sub = df[df["setting"] == "S2"].copy()
    print(f"\n{'─'*80}")
    print("  Table S2: Mixed Linearity (RESET routing)")
    print(f"{'─'*80}")
    header = ["n", "ARI", "TB Rate (true)", "TB Rate (false)",
              "ACE RMSE C1", "ACE RMSE C2", "ACE RMSE C3"]
    rows = []
    for _, r in sub.iterrows():
        rows.append([
            int(r["n"]),
            fmt(r["ARI_mean"], r["ARI_sd"]),
            f"{r['TB_true']:.2f}" if not np.isnan(r["TB_true"]) else "—",
            f"{r['TB_false']:.2f}" if not np.isnan(r["TB_false"]) else "—",
            fmt(r["ACE0_mean"], r["ACE0_sd"]),
            fmt(r["ACE1_mean"], r["ACE1_sd"]),
            fmt(r["ACE2_mean"], r["ACE2_sd"]),
        ])
    if HAS_TABULATE:
        print(tabulate(rows, headers=header, tablefmt="github"))
    else:
        print("\t".join(header))
        for row in rows:
            print("\t".join(str(x) for x in row))


def print_ablation_table(results: dict):
    print(f"\n{'─'*70}")
    print("  Table 4: Ablation Analysis (S1, n=1000)")
    print(f"{'─'*70}")
    header = ["Method", "ARI", "OVS Acc", "ACE RMSE (mean)"]
    rows = []
    for label, r in results.items():
        mean_rmse = np.nanmean([r["ACE0_mean"], r["ACE1_mean"], r["ACE2_mean"]])
        sd_rmse   = np.nanmean([r["ACE0_sd"],   r["ACE1_sd"],   r["ACE2_sd"]])
        rows.append([
            label,
            fmt(r["ARI_mean"], r["ARI_sd"]),
            fmt(r["OVS_mean"], r["OVS_sd"]),
            fmt(mean_rmse, sd_rmse),
        ])
    if HAS_TABULATE:
        print(tabulate(rows, headers=header, tablefmt="github"))
    else:
        print("\t".join(header))
        for row in rows:
            print("\t".join(str(x) for x in row))


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 12 ─ PLOTS
# ═════════════════════════════════════════════════════════════════════════════

def plot_results(all_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"S1": "steelblue", "S2": "tomato", "S3": "seagreen"}

    for setting, color in colors.items():
        sub = all_df[all_df["setting"] == setting]
        if sub.empty:
            continue
        ns = sub["n"].values
        axes[0].plot(ns, sub["ARI_mean"], marker="o", color=color,
                     label=setting)
        axes[0].fill_between(ns,
                              sub["ARI_mean"] - sub["ARI_sd"],
                              sub["ARI_mean"] + sub["ARI_sd"],
                              alpha=0.15, color=color)

        mean_rmse = (sub["ACE0_mean"] + sub["ACE1_mean"] + sub["ACE2_mean"]) / 3
        axes[1].plot(ns, mean_rmse, marker="^", color=color, label=setting)

    for ax, ylabel, title in zip(
        axes,
        ["ARI", "Mean ACE RMSE"],
        ["Cluster Recovery (ARI)", "ACE Estimation Error"],
    ):
        ax.set_xlabel("Sample size n")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)

    plt.suptitle("CaSPECT Simulation Study", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("simulation_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Figure saved → simulation_results.png")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Configuration ─────────────────────────────────────────────────────────
    # Set M_FULL=200 and M_ABLATION=200 for full paper results.
    # Reduced defaults here for a quick smoke-test run.
    M_FULL     = 50     # replications for S1/S2/S3  (paper: 200)
    M_ABLATION = 50     # replications for ablations (paper: 200)
    VERBOSE    = True

    print("=" * 70)
    print("  CaSPECT Simulation Study")
    print("=" * 70)

    all_results = []

    # ── S1: Clean linear DAG ──────────────────────────────────────────────────
    print("\n[S1] Clean linear DAG, non-Gaussian (t5) errors")
    t0  = time.time()
    s1  = run_mc("S1", [200, 500, 1000, 2000], M=M_FULL, verbose=VERBOSE)
    print(f"  S1 done in {time.time()-t0:.1f}s")
    print_table(s1, "Table 1: Setting S1 (clean linear, non-Gaussian)", "S1")
    all_results.append(s1)

    # ── S2: Mixed linearity ───────────────────────────────────────────────────
    print("\n[S2] Mixed linearity (3 nonlinear edges, RESET routing)")
    t0  = time.time()
    s2  = run_mc("S2", [500, 1000, 2000], M=M_FULL, verbose=VERBOSE)
    print(f"  S2 done in {time.time()-t0:.1f}s")
    print_s2_table(s2)
    all_results.append(s2)

    # ── S3: Mild sufficiency violation ────────────────────────────────────────
    print("\n[S3] Mild causal sufficiency violation (γ=0.4)")
    t0  = time.time()
    s3  = run_mc("S3", [500, 1000, 2000], M=M_FULL, verbose=VERBOSE)
    print(f"  S3 done in {time.time()-t0:.1f}s")
    print_table(s3, "Table 3: Setting S3 (latent confounder, γ=0.4)", "S3")
    all_results.append(s3)

    # ── Ablations (S1, n=1000) ────────────────────────────────────────────────
    print("\n[Ablations] S1, n=1000")
    ablation_results = {}

    configs = {
        "Full CaSPECT":        dict(force_pc_only=False, no_stability=False, symmetrise=False),
        "A1: PC-only orient":  dict(force_pc_only=True,  no_stability=False, symmetrise=False),
        "A2: No stab weight":  dict(force_pc_only=False, no_stability=True,  symmetrise=False),
        "A3: Symmetrisation":  dict(force_pc_only=False, no_stability=False, symmetrise=True),
    }
    for label, cfg in configs.items():
        print(f"  Running {label} …")
        df = run_mc("S1", [1000], M=M_ABLATION, verbose=False, **cfg)
        r  = df.iloc[0].to_dict()
        ablation_results[label] = r

    print_ablation_table(ablation_results)

    # ── Combine and save ──────────────────────────────────────────────────────
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv("simulation_results.csv", index=False)
    print("\n  Results saved → simulation_results.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_results(all_df)

    print("\n" + "=" * 70)
    print("  Simulation complete.")
    print("=" * 70)
