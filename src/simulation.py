import warnings
import itertools
from copy import deepcopy
from typing import Optional

import numpy as np
import scipy.linalg as sla
import scipy.stats as sst
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera as jb_test
from statsmodels.stats.multitest import multipletests

from pygam import LinearGAM, s
from lingam import DirectLiNGAM
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from scipy.optimize import linear_sum_assignment 

warnings.filterwarnings("ignore")

MASTER_SEED = 42
np.random.seed(MASTER_SEED)


def generate_random_dag(q: int, edge_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a random DAG adjacency matrix (q×q)."""
    B = np.zeros((q, q))
    for i in range(q):
        for j in range(i):
            if rng.random() < edge_prob:
                coeff = rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
                B[i, j] = coeff          # j → i
    return B


def compute_true_ace(B: np.ndarray, treat_idx: int, outcome_idx: int) -> float:
    """Total causal effect of treat on outcome via all directed paths."""
    q = B.shape[0]
    IB_inv = np.linalg.inv(np.eye(q) - B)
    return float(IB_inv[outcome_idx, treat_idx])


def generate_data_s1(
    n: int,
    B: np.ndarray,
    cluster_intercepts: np.ndarray,
    treat_idx: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Setting S1: linear SEM, t5 errors."""
    q = B.shape[0]
    k = len(cluster_intercepts)
    n_per = n // k
    sizes = [n_per] * k
    sizes[-1] += n - sum(sizes)

    X_list, labels_list = [], []
    true_aces = []

    for c, (alpha_c, nc) in enumerate(zip(cluster_intercepts, sizes)):
        E = rng.standard_t(df=5, size=(nc, q)) / np.sqrt(5 / 3)
        X = np.zeros((nc, q))
        for j in range(q):
            parents = np.where(B[j] != 0)[0]
            X[:, j] = X[:, parents] @ B[j, parents] + E[:, j]
            if j == treat_idx:
                X[:, j] += alpha_c
        X_list.append(X)
        labels_list.append(np.full(nc, c))
        true_aces.append(compute_true_ace(B, treat_idx, q - 1))

    X_all = np.vstack(X_list)
    labels_all = np.concatenate(labels_list)
    return X_all, labels_all, np.array(true_aces)


def generate_data_s2(
    n: int,
    B: np.ndarray,
    nonlinear_edges: list[tuple[int, int]],
    cluster_intercepts: np.ndarray,
    treat_idx: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Setting S2: three nonlinear edges v = beta*sin(pi*u) + e."""
    q = B.shape[0]
    k = len(cluster_intercepts)
    n_per = n // k
    sizes = [n_per] * k
    sizes[-1] += n - sum(sizes)

    nonlinear_set = set(nonlinear_edges)
    X_list, labels_list = [], []
    true_aces = []

    for c, (alpha_c, nc) in enumerate(zip(cluster_intercepts, sizes)):
        E = rng.standard_t(df=5, size=(nc, q)) / np.sqrt(5 / 3)
        X = np.zeros((nc, q))
        for j in range(q):
            parents = np.where(B[j] != 0)[0]
            contrib = np.zeros(nc)
            for p in parents:
                if (p, j) in nonlinear_set:
                    contrib += B[j, p] * np.sin(np.pi * X[:, p])
                else:
                    contrib += B[j, p] * X[:, p]
            X[:, j] = contrib + E[:, j]
            if j == treat_idx:
                X[:, j] += alpha_c
        X_list.append(X)
        labels_list.append(np.full(nc, c))
        true_aces.append(compute_true_ace(B, treat_idx, q - 1))

    X_all = np.vstack(X_list)
    labels_all = np.concatenate(labels_list)
    return X_all, labels_all, np.array(true_aces)


def generate_data_s3(
    n: int,
    B: np.ndarray,
    latent_targets: tuple[int, int],
    gamma: float,
    cluster_intercepts: np.ndarray,
    treat_idx: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Setting S3: mild latent confounder H affecting two observed variables."""
    q = B.shape[0]
    k = len(cluster_intercepts)
    n_per = n // k
    sizes = [n_per] * k
    sizes[-1] += n - sum(sizes)

    u_star, v_star = latent_targets
    X_list, labels_list = [], []
    true_aces = []

    for c, (alpha_c, nc) in enumerate(zip(cluster_intercepts, sizes)):
        H = rng.standard_normal(nc)
        E = rng.standard_t(df=5, size=(nc, q)) / np.sqrt(5 / 3)
        X = np.zeros((nc, q))
        for j in range(q):
            parents = np.where(B[j] != 0)[0]
            X[:, j] = X[:, parents] @ B[j, parents] + E[:, j]
            if j == u_star:
                X[:, j] += gamma * H
            if j == v_star:
                X[:, j] += gamma * H
            if j == treat_idx:
                X[:, j] += alpha_c
        X_list.append(X)
        labels_list.append(np.full(nc, c))
        true_aces.append(compute_true_ace(B, treat_idx, q - 1))

    X_all = np.vstack(X_list)
    labels_all = np.concatenate(labels_list)
    return X_all, labels_all, np.array(true_aces)


def run_pc_algo(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Run PC algorithm, return binary directed adjacency."""
    cg = pc(data, indep_test=fisherz, alpha=alpha,
            uc_rule=0, uc_priority=2,
            mvpc=False, verbose=False, show_progress=False)
    G = cg.G.graph
    d = G.shape[0]
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


def bootstrap_pc(
    data: np.ndarray,
    B: int = 100,
    alpha: float = 0.05,
    theta: float = 0.50,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = data.shape
    f_count = np.zeros((d, d))
    g_count = np.zeros((d, d))
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            adj = run_pc_algo(data[idx], alpha=alpha)
        except Exception:
            continue
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                if adj[i, j] != 0 or adj[j, i] != 0:
                    f_count[i, j] += 1
                if adj[i, j] == 1 and adj[j, i] == 0:
                    g_count[i, j] += 1
    f = f_count / B
    g = g_count / B
    with np.errstate(invalid="ignore"):
        rho = np.where(f > 0, g / f, 0.0)
    return f, g, rho


def jb_fraction(data: np.ndarray, alpha: float = 0.10) -> float:
    n, d = data.shape
    count = 0
    for j in range(d):
        others = [k for k in range(d) if k != j]
        Xo = sm.add_constant(data[:, others])
        try:
            resid = sm.OLS(data[:, j], Xo).fit().resid
            _, p, _, _ = jb_test(resid)
            if p < alpha:
                count += 1
        except Exception:
            pass
    return count / d


def compute_ovs(
    f: np.ndarray, rho: np.ndarray, delta_L: np.ndarray,
    w_L_max: float = 0.20, tau: float = 0.15, theta: float = 0.50, ng_frac: float = 0.0
) -> tuple[dict, np.ndarray]:
    w_L = min(ng_frac, 1.0) * w_L_max
    w_pc = 1.0 - w_L
    d = f.shape[0]
    ovs_mat = np.zeros((d, d))
    orientations = {}
    for i in range(d):
        for j in range(i + 1, d):
            if f[i, j] < theta:
                continue
            delta_pc = rho[i, j] - rho[j, i]
            dl = float(np.sign(delta_L[i, j] - delta_L[j, i]))
            ovs = w_pc * f[i, j] * delta_pc + w_L * dl
            ovs_mat[i, j] = ovs
            ovs_mat[j, i] = -ovs
            if ovs > tau:
                orientations[(i, j)] = (i, j)
            elif ovs < -tau:
                orientations[(i, j)] = (j, i)
            else:
                orientations[(i, j)] = None
    return orientations, ovs_mat


def resolve_dag(
    orientations: dict,
    d: int,
    ovs_mat: np.ndarray,
    symmetrise: bool = False,
) -> tuple[np.ndarray, nx.DiGraph]:
    G_nx = nx.DiGraph()
    G_nx.add_nodes_from(range(d))
    resolved = {}
    ambiguous = []

    for edge, direction in orientations.items():
        if direction is None:
            ambiguous.append(edge)
            continue
        u, v = direction
        G_nx.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(G_nx):
            G_nx.remove_edge(u, v)
            G_nx.add_edge(v, u)
            if nx.is_directed_acyclic_graph(G_nx):
                resolved[edge] = (v, u)
            else:
                G_nx.remove_edge(v, u)
                ambiguous.append(edge)
        else:
            resolved[edge] = direction

    changed = True
    while changed:
        changed = False
        for a, b in list(G_nx.edges()):
            for c in list(G_nx.successors(b)):
                if c == a:
                    continue
                if G_nx.has_edge(a, c) and G_nx.has_edge(c, a):
                    G_nx.remove_edge(c, a)
                    changed = True

    ambiguous.sort(key=lambda e: abs(ovs_mat[e[0], e[1]]))
    for (i, j) in ambiguous:
        if not G_nx.has_node(i) or not G_nx.has_node(j):
            continue
        if symmetrise:
            resolved[(i, j)] = (i, j)
            G_nx.add_edge(i, j)
            G_nx.add_edge(j, i)
        else:
            u_keep = min(i, j)
            u_drop = max(i, j)
            if not G_nx.has_node(u_drop):
                continue
            for pred in list(G_nx.predecessors(u_drop)):
                if pred != u_keep and G_nx.has_node(pred) and not G_nx.has_edge(pred, u_keep):
                    G_nx.add_edge(pred, u_keep)
            for succ in list(G_nx.successors(u_drop)):
                if succ != u_keep and G_nx.has_node(succ) and not G_nx.has_edge(u_keep, succ):
                    G_nx.add_edge(u_keep, succ)
            G_nx.remove_node(u_drop)
            resolved[(i, j)] = (u_keep, -1)

    return resolved, G_nx


def reset_test(y: np.ndarray, X_reg: np.ndarray, alpha: float = 0.05) -> bool:
    """Returns True if linearity NOT rejected."""
    Xc = sm.add_constant(X_reg)
    try:
        res0 = sm.OLS(y, Xc).fit()
        yhat = res0.fittedvalues
        X_aug = np.column_stack([Xc, yhat ** 2, yhat ** 3])
        res1 = sm.OLS(y, X_aug).fit()
        ss0, ss1 = res0.ssr, res1.ssr
        df_den = len(y) - X_aug.shape[1]
        if df_den <= 0 or ss1 < 1e-12:
            return True
        F = ((ss0 - ss1) / 2) / (ss1 / df_den)
        p = 1 - sst.f.cdf(F, 2, df_den)
        return p > alpha
    except Exception:
        return True


def ols_ate(y, t, Z):
    if Z.shape[1] > 0:
        Xr = sm.add_constant(np.column_stack([t, Z]))
        idx = 1
    else:
        Xr = sm.add_constant(t)
        idx = 1
    try:
        return float(sm.OLS(y, Xr).fit().params[idx])
    except Exception:
        return 0.0


def dml_ate(y, t, Z, K=5, seed=42):
    n = len(y)
    V_res = np.zeros(n)
    U_res = np.zeros(n)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    for tr, te in kf.split(np.arange(n)):
        Xtr, Xte = Z[tr], Z[te]
        ytr, yte = y[tr], y[te]
        ttr, tte = t[tr], t[te]
        if Z.shape[1] == 0:
            V_res[te] = yte - ytr.mean()
            U_res[te] = tte - ttr.mean()
            continue
        try:
            n_sp = min(20, max(4, len(tr) // 10))
            terms = sum(s(j, n_splines=n_sp) for j in range(Z.shape[1]))
            m_y = LinearGAM(terms).fit(Xtr, ytr)
            m_t = LinearGAM(terms).fit(Xtr, ttr)
            V_res[te] = yte - m_y.predict(Xte)
            U_res[te] = tte - m_t.predict(Xte)
        except Exception:
            rf_y = RandomForestRegressor(50, random_state=seed).fit(Xtr, ytr)
            rf_t = RandomForestRegressor(50, random_state=seed).fit(Xtr, ttr)
            V_res[te] = yte - rf_y.predict(Xte)
            U_res[te] = tte - rf_t.predict(Xte)
    denom = np.sum(U_res ** 2)
    return float(np.sum(U_res * V_res) / denom) if denom > 1e-12 else 0.0


def backdoor_set(G_nx, u, v, d):
    if not G_nx.has_node(u) or not G_nx.has_node(v):
        return []
    try:
        desc_u = nx.descendants(G_nx, u)
    except Exception:
        desc_u = set()
    try:
        return [w for w in G_nx.predecessors(u)
                if w not in desc_u and w != v and w < d and G_nx.has_node(w)]
    except Exception:
        return []


def build_adjacency(
    data: np.ndarray,
    resolved: dict,
    G_nx: nx.DiGraph,
    f: np.ndarray,
    alpha_reset: float = 0.05,
    dml_k: int = 5,
    use_stability: bool = True,
) -> np.ndarray:
    d = data.shape[1]
    A = np.zeros((d, d))

    edges_to_process = []
    for (i, j), direction in resolved.items():
        u, v = direction
        if v == -1 or u >= d or v >= d:
            continue
        if not G_nx.has_node(u) or not G_nx.has_node(v):
            continue
        edges_to_process.append((u, v))

    for (u, v) in edges_to_process:
        bd = backdoor_set(G_nx, u, v, d)
        bd = [b for b in bd if b < d]
        t_col = data[:, u]
        y_col = data[:, v]
        Z = data[:, bd] if bd else np.empty((len(y_col), 0))
        X_reg = np.column_stack([t_col, Z]) if Z.shape[1] > 0 else t_col.reshape(-1, 1)
        linear = reset_test(y_col, X_reg, alpha=alpha_reset)
        ate = ols_ate(y_col, t_col, Z) if linear else dml_ate(y_col, t_col, Z, K=dml_k)
        fi = f[u, v] if use_stability and u < f.shape[0] and v < f.shape[1] else 1.0
        A[u, v] = fi * abs(ate)
    return A


def chung_laplacian(A: np.ndarray, alpha: float = 0.15):
    d = A.shape[0]
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    P0 = A / row_sums_safe
    P0[row_sums.flatten() == 0] = 1.0 / d
    P = (1 - alpha) * P0 + (alpha / d) * np.ones((d, d))
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.abs(np.real(eigvecs[:, idx]))
    pi /= pi.sum()
    pi_safe = np.where(pi > 1e-12, pi, 1e-12)
    Pstar = (pi[np.newaxis, :] / pi_safe[:, np.newaxis]) * P.T
    L = np.eye(d) - 0.5 * (P + Pstar)
    L = 0.5 * (L + L.T)
    return L, P, pi


def select_k_star(L: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(L)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    gaps = np.diff(eigvals)
    K_star = max(1, int(np.argmax(gaps)))
    return K_star, eigvals, eigvecs


def cluster_embedding(
    data: np.ndarray,
    eigvecs: np.ndarray,
    K_star: int,
    k_max: int = 9,
    seed: int = 42,
) -> tuple[np.ndarray, int, float]:
    Vk = eigvecs[:, 1:K_star + 1]
    X_emb = data @ Vk
    best_k, best_sil = 2, -1.0
    for k in range(2, min(k_max + 1, X_emb.shape[0])):
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X_emb)
        if len(set(labels)) > 1:
            sc = silhouette_score(X_emb, labels)
            if sc > best_sil:
                best_sil, best_k = sc, k
    labels_final = KMeans(n_clusters=best_k, n_init=50, random_state=seed).fit_predict(X_emb)
    return labels_final, best_k, best_sil

def compute_shd(B_true: np.ndarray, resolved: dict, d: int) -> int:
    true_adj = (B_true != 0).astype(int)
    true_dir = true_adj.T.copy()
    est_dir = np.zeros((d, d), dtype=int)
    for (i, j), direction in resolved.items():
        u, v = direction
        if v == -1 or u >= d or v >= d:
            continue
        est_dir[u, v] = 1

    shd = 0
    for i in range(d):
        for j in range(i + 1, d):
            t_ij, t_ji = true_dir[i, j], true_dir[j, i]
            e_ij, e_ji = est_dir[i, j], est_dir[j, i]
            true_edge = t_ij or t_ji
            est_edge = e_ij or e_ji

            if true_edge and not est_edge:
                shd += 1
            elif not true_edge and est_edge:
                shd += 1
            elif true_edge and est_edge:
                if (t_ij != e_ij) or (t_ji != e_ji):
                    shd += 1
    return shd


def ovs_orientation_accuracy(
    orientations: dict,
    resolved: dict,
    B_true: np.ndarray,
    d: int,
) -> float:
    true_dir = (B_true.T != 0).astype(int)
    correct, total = 0, 0
    for (i, j), direction in resolved.items():
        u, v = direction
        if v == -1 or u >= d or v >= d:
            continue
        total += 1
        if true_dir[u, v] == 1:
            correct += 1
    return correct / total if total > 0 else 0.0

def cluster_ace(
    data_orig: np.ndarray,
    labels: np.ndarray,
    treat_idx: int,
    outcome_idx: int,
    k_clusters: int,
    seed: int = 42,
) -> np.ndarray:
    aces = np.full(k_clusters, np.nan)
    for c in range(k_clusters):
        mask = labels == c
        if mask.sum() < 20:
            continue
        Xc = data_orig[mask]
        t = Xc[:, treat_idx]
        y = Xc[:, outcome_idx]
        cov_idx = [j for j in range(data_orig.shape[1]) if j not in (treat_idx, outcome_idx)]
        Z = Xc[:, cov_idx] if cov_idx else np.empty((len(y), 0))
        try:
            aces[c] = dml_ate(y, t, Z, K=5, seed=seed)
        except Exception:
            aces[c] = np.nan
    return aces


def run_pipeline(
    data: np.ndarray,
    alpha_ci: float = 0.05,
    B: int = 100,
    theta: float = 0.50,
    tau: float = 0.15,
    w_L_max: float = 0.20,
    alpha_pr: float = 0.15,
    use_stability: bool = True,
    symmetrise: bool = False,
    seed: int = 42,
) -> dict:
    data_std = StandardScaler().fit_transform(data)
    d = data_std.shape[1]

    f, g, rho = bootstrap_pc(data_std, B=B, alpha=alpha_ci, theta=theta, seed=seed)
    ng_frac = jb_fraction(data_std)

    try:
        model = DirectLiNGAM()
        model.fit(data_std)
        B_hat = model.adjacency_matrix_
    except Exception:
        B_hat = np.zeros((d, d))

    orientations, ovs_mat = compute_ovs(f, rho, B_hat, w_L_max=w_L_max, tau=tau, theta=theta, ng_frac=ng_frac)
    resolved, G_nx = resolve_dag(orientations, d, ovs_mat, symmetrise=symmetrise)
    A = build_adjacency(data_std, resolved, G_nx, f, use_stability=use_stability, dml_k=5)
    L, P, pi = chung_laplacian(A, alpha=alpha_pr)
    K_star, eigvals, eigvecs = select_k_star(L)
    gaps = np.diff(eigvals)
    spectral_gap = float(gaps[K_star - 1]) if K_star - 1 < len(gaps) else 0.0
    labels_pred, k_star, sil = cluster_embedding(data_std, eigvecs, K_star, seed=seed)

    return {
        "labels"       : labels_pred,
        "k_star"       : k_star,
        "K_star"       : K_star,
        "spectral_gap" : spectral_gap,
        "resolved"     : resolved,
        "G_nx"         : G_nx,
        "f"            : f,
        "ovs_mat"      : ovs_mat,
        "orientations" : orientations,
        "A"            : A,
        "L"            : L,
        "eigvals"      : eigvals,
    }


def evaluate_replication(
    data: np.ndarray,
    true_labels: np.ndarray,
    true_aces: np.ndarray,
    B_true: np.ndarray,
    treat_idx: int,
    outcome_idx: int,
    pipeline_kwargs: dict,
    seed: int,
) -> dict:
    VALID_KEYS = {"alpha_ci","B","theta","tau","w_L_max","alpha_pr","use_stability","symmetrise"}
    kw = {k: v for k, v in pipeline_kwargs.items() if k in VALID_KEYS}
    try:
        result = run_pipeline(data, seed=seed, **kw)
    except Exception:
        return None

    labels_pred = result["labels"]
    d = B_true.shape[0]

    ari = adjusted_rand_score(true_labels, labels_pred)
    shd = compute_shd(B_true, result["resolved"], d)
    ova = ovs_orientation_accuracy(result["orientations"], result["resolved"], B_true, d)
    k_pred = result["k_star"]
    pred_aces = cluster_ace(data, labels_pred, treat_idx, outcome_idx, k_pred, seed=seed)

    true_k = len(np.unique(true_labels))
    ace_rmse = np.full(true_k, np.nan)

    # RECTIFIED: Map available valid clusters without throwing away the whole replication on a single NaN
    if k_pred == true_k:
        cost_matrix = np.zeros((true_k, true_k))
        valid_mask = ~np.isnan(pred_aces)

        # Fill cost matrix; assign high penalty if the predicted cluster was an empty NaN
        for i in range(true_k):
            for j in range(true_k):
                if valid_mask[j]:
                    cost_matrix[i, j] = (pred_aces[j] - true_aces[i]) ** 2
                else:
                    cost_matrix[i, j] = 999.0

        true_ind, pred_ind = linear_sum_assignment(cost_matrix)
        for t_idx, p_idx in zip(true_ind, pred_ind):
            if valid_mask[p_idx]:
                ace_rmse[t_idx] = cost_matrix[t_idx, p_idx]

    return {
        "ari"           : ari,
        "shd"           : shd,
        "ova"           : ova,
        "ace_rmse"      : ace_rmse,
        "spectral_gap"  : result["spectral_gap"],
        "K_star"        : result["K_star"],
        "k_star"        : k_pred,
    }


def run_mc(
    setting: str,
    n: int,
    M: int,
    q: int,
    treat_idx: int,
    outcome_idx: int,
    cluster_intercepts: np.ndarray,
    pipeline_kwargs: dict,
    master_seed: int = 2024,
    nonlinear_edges: Optional[list] = None,
    latent_targets: Optional[tuple] = None,
    gamma: float = 0.4,
) -> dict:
    rng_meta = np.random.default_rng(master_seed + hash(setting) % 10000 + n)
    ari_list, shd_list, ova_list, gap_list = [], [], [], []
    ace_rmse_list = [[] for _ in range(3)]

    B_true = generate_random_dag(q, 0.35, rng_meta)

    if setting == "S2" and nonlinear_edges is None:
        edges = [(i, j) for i in range(q) for j in range(i) if B_true[i, j] != 0]
        edges_sorted = sorted(edges, key=lambda e: abs(B_true[e[0], e[1]]), reverse=True)
        nonlinear_edges = [(j, i) for (i, j) in edges_sorted[:3]]

    if setting == "S3" and latent_targets is None:
        non_special = [v for v in range(q) if v not in (treat_idx, outcome_idx)]
        latent_targets = tuple(rng_meta.choice(non_special, 2, replace=False))

    for m in range(M):
        seed_m = master_seed + m * 997 + n * 13
        rng_m = np.random.default_rng(seed_m)

        if setting == "S1":
            data, tlabels, taces = generate_data_s1(n, B_true, cluster_intercepts, treat_idx, rng_m)
        elif setting == "S2":
            data, tlabels, taces = generate_data_s2(n, B_true, nonlinear_edges, cluster_intercepts, treat_idx, rng_m)
        elif setting == "S3":
            data, tlabels, taces = generate_data_s3(n, B_true, latent_targets, gamma, cluster_intercepts, treat_idx, rng_m)

        res = evaluate_replication(data, tlabels, taces, B_true, treat_idx, outcome_idx, pipeline_kwargs, seed=seed_m)
        if res is None:
            continue

        ari_list.append(res["ari"])
        shd_list.append(res["shd"])
        ova_list.append(res["ova"])
        gap_list.append(res["spectral_gap"])
        for c in range(3):
            if not np.isnan(res["ace_rmse"][c]):
                ace_rmse_list[c].append(res["ace_rmse"][c])

    def ms(lst):
        a = np.array(lst)
        return float(np.mean(a)) if len(a) > 0 else 0.0, float(np.std(a)) if len(a) > 0 else 0.0

    return {
        "ari_mean"  : ms(ari_list)[0], "ari_std"  : ms(ari_list)[1],
        "shd_mean"  : ms(shd_list)[0], "shd_std"  : ms(shd_list)[1],
        "ova_mean"  : ms(ova_list)[0], "ova_std"  : ms(ova_list)[1],
        "gap_mean"  : ms(gap_list)[0], "gap_std"  : ms(gap_list)[1],
        "rmse_c1"   : ms(ace_rmse_list[0])[0] ** 0.5 if ace_rmse_list[0] else np.nan,
        "rmse_c2"   : ms(ace_rmse_list[1])[0] ** 0.5 if ace_rmse_list[1] else np.nan,
        "rmse_c3"   : ms(ace_rmse_list[2])[0] ** 0.5 if ace_rmse_list[2] else np.nan,
        "n_reps"    : len(ari_list),
    }


def print_table(title: str, rows: list[dict], cols: list[str], headers: list[str]) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    col_w = 12
    header_line = "".join(f"{h:>{col_w}}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        line = ""
        for c in cols:
            val = row.get(c, np.nan)
            if isinstance(val, float) and not np.isnan(val):
                line += f"{val:>{col_w}.3f}"
            elif isinstance(val, (int, np.integer)):
                line += f"{val:>{col_w}}"
            else:
                line += f"{'N/A':>{col_w}}"
        print(line)
    print("=" * 70)


def main():
    Q          = 8
    TREAT_IDX  = 6
    OUT_IDX    = 7
    INTERCEPTS = np.array([-3.0, 0.0, 3.0])

    M_REPS     = 100  

    DEFAULT_KW = dict(
        alpha_ci=0.05,
        B=100,     
        theta=0.50,
        tau=0.15,
        w_L_max=0.20,
        alpha_pr=0.15,
        use_stability=True,
        symmetrise=False,
    )

    print("\n>>> Running Setting S1 (Clean Linear, Non-Gaussian) ...")
    s1_rows = []
    for n in [500, 1000, 2000]:
        print(f"    n={n} ...", end=" ", flush=True)
        res = run_mc("S1", n, M_REPS, Q, TREAT_IDX, OUT_IDX, INTERCEPTS, DEFAULT_KW, MASTER_SEED)
        res["n"] = n
        s1_rows.append(res)
        print(f"ARI={res['ari_mean']:.2f}  SHD={res['shd_mean']:.1f}")

    print_table(
        "Table 1. Setting S1: Clean Linear DAG",
        s1_rows,
        ["n", "ari_mean", "shd_mean", "ova_mean", "rmse_c1", "rmse_c2", "rmse_c3"],
        ["n", "ARI", "SHD", "OVS_Acc", "RMSE_C1", "RMSE_C2", "RMSE_C3"],
    )

    print("\n>>> Running Setting S2 (Mixed Linearity) ...")
    s2_rows = []
    for n in [500, 1000, 2000]:
        print(f"    n={n} ...", end=" ", flush=True)
        res = run_mc("S2", n, M_REPS, Q, TREAT_IDX, OUT_IDX, INTERCEPTS, DEFAULT_KW, MASTER_SEED)
        res["n"] = n
        s2_rows.append(res)
        print(f"ARI={res['ari_mean']:.2f}  SHD={res['shd_mean']:.1f}")

    print_table(
        "Table 2. Setting S2: Mixed Linearity",
        s2_rows,
        ["n", "ari_mean", "shd_mean", "ova_mean", "rmse_c1", "rmse_c2", "rmse_c3"],
        ["n", "ARI", "SHD", "OVS_Acc", "RMSE_C1", "RMSE_C2", "RMSE_C3"],
    )

    print("\n>>> Running Setting S3 (Latent Confounder) ...")
    s3_rows = []
    for n in [500, 1000, 2000]:
        print(f"    n={n} ...", end=" ", flush=True)
        res = run_mc("S3", n, M_REPS, Q, TREAT_IDX, OUT_IDX, INTERCEPTS, DEFAULT_KW, MASTER_SEED, gamma=0.40)
        res["n"] = n
        s3_rows.append(res)
        print(f"ARI={res['ari_mean']:.2f}  SHD={res['shd_mean']:.1f}")

    print_table(
        "Table 3. Setting S3: Latent Confounder",
        s3_rows,
        ["n", "ari_mean", "shd_mean", "ova_mean", "rmse_c1", "rmse_c2", "rmse_c3"],
        ["n", "ARI", "SHD", "OVS_Acc", "RMSE_C1", "RMSE_C2", "RMSE_C3"],
    )


if __name__ == "__main__":
    Q = 8
    TREAT_IDX = 6
    OUT_IDX = 7
    INTERCEPTS = np.array([-3.0, 0.0, 3.0])

    M_ABLATION = 100
    n_samples = 1000

    print("=" * 70)
    print("  CaSPECT Ablation Analysis (Setting S1, n=1000)")
    print("=" * 70)

    ablation_results = {}

    configs = {
        "Full CaSPECT":        dict(w_L_max=0.20, use_stability=True,  symmetrise=False),
        "A1: PC-only orient":  dict(w_L_max=0.0,  use_stability=True,  symmetrise=False),
        "A2: No stab weight":  dict(w_L_max=0.20, use_stability=False, symmetrise=False),
        "A3: Symmetrisation":  dict(w_L_max=0.20, use_stability=True,  symmetrise=True),
    }

    for label, cfg in configs.items():
        print(f"\nRunning {label}...")

        pipeline_kwargs = dict(alpha_ci=0.05, B=100, theta=0.50, tau=0.15, alpha_pr=0.15)
        pipeline_kwargs.update(cfg)

        res = run_mc(
            setting="S1",
            n=n_samples,
            M=M_ABLATION,
            q=Q,
            treat_idx=TREAT_IDX,
            outcome_idx=OUT_IDX,
            cluster_intercepts=INTERCEPTS,
            pipeline_kwargs=pipeline_kwargs,
            master_seed=MASTER_SEED
        )

        ablation_results[label] = res

        mean_rmse = np.nanmean([res["rmse_c1"], res["rmse_c2"], res["rmse_c3"]])
        print(f"  -> ARI: {res['ari_mean']:.3f} | SHD: {res['shd_mean']:.1f} | RMSE: {mean_rmse:.3f}")

    print(f"\n{'─'*75}")
    print("  Table 4: Ablation Analysis (S1, n=1000)")
    print(f"{'─'*75}")
    header = f"{'Method':<20} | {'ARI':<12} | {'SHD':<10} | {'OVS Acc':<10} | {'Mean RMSE'}"
    print(header)
    print("-" * len(header))

    for label, r in ablation_results.items():
        mean_rmse = np.nanmean([r["rmse_c1"], r["rmse_c2"], r["rmse_c3"]])
        ari_str = f"{r['ari_mean']:.3f} ({r['ari_std']:.3f})"
        shd_str = f"{r['shd_mean']:.1f} ({r['shd_std']:.1f})"
        ovs_str = f"{r['ova_mean']:.3f} ({r['ova_std']:.3f})"

        print(f"{label:<20} | {ari_str:<12} | {shd_str:<10} | {ovs_str:<10} | {mean_rmse:.3f}")
