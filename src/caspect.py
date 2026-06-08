import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.linalg import eig
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera as jb_test

from pygam import LinearGAM, s

from lingam import DirectLiNGAM
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def plot_spectral_gap(L):
    """Plot eigenvalue spectrum and spectral gaps of the Chung Laplacian."""
    eigvals = np.linalg.eigvalsh(L)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(eigvals, marker='o', linewidth=1.5)
    axes[0].set_title("Eigenvalue Spectrum")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].grid(True)

    gaps = np.diff(eigvals)
    axes[1].plot(gaps, marker='o', color='tomato', linewidth=1.5)
    # Skip trivial gap at index 0 (FIX-2 companion visualisation)
    k_star_vis = int(np.argmax(gaps[1:]) + 1) if len(gaps) > 1 else 0
    axes[1].axvline(k_star_vis, color='navy', linestyle='--',
                    label=f"K* = {k_star_vis + 1}  (non-trivial)")
    axes[1].set_title("Spectral Gaps  (gap[0] is trivial – often largest)")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Gap")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_silhouette(X_embed):
    """Plot silhouette score against cluster count."""
    if X_embed.shape[1] < 2:
        print("[CSC-PC] Silhouette plot skipped: embedding is 1-D.")
        return

    scores = []
    Ks = range(2, min(10, X_embed.shape[0]))
    for k in Ks:
        labels = KMeans(n_clusters=k, n_init=20,
                        random_state=42).fit_predict(X_embed)
        scores.append(silhouette_score(X_embed, labels))

    plt.figure(figsize=(6, 4))
    plt.plot(list(Ks), scores, marker='o', linewidth=1.5)
    plt.title("Silhouette Score vs. K")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Silhouette score")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ── FIX-3: plot_clusters always does t-SNE on X_orig ─────────────────────────
def plot_clusters(X_embed, clusters, X_orig=None, perplexity=30, random_state=42):
    """
    Scatter-plot cluster assignments.

    If X_orig is supplied (recommended), t-SNE is always run on X_orig so
    that a 2-D projection is available even when K*=1.  When K*=1 we show
    both the 1-D spectral strip (top panel) and the t-SNE map (bottom panel).
    If X_orig is None we fall back to the old behaviour.
    """
    dim = X_embed.shape[1]

    if X_orig is not None:
        n_samples = X_orig.shape[0]
        perp = min(perplexity, n_samples - 1)
        X_2d = TSNE(n_components=2, perplexity=perp,
                    random_state=random_state).fit_transform(X_orig)

        if dim == 1:
            fig, axes = plt.subplots(2, 1, figsize=(8, 9))
            # top: 1-D spectral strip
            sc0 = axes[0].scatter(X_embed[:, 0], np.zeros_like(X_embed[:, 0]),
                                  c=clusters, cmap='tab10', alpha=0.7, s=20)
            axes[0].set_title("Cluster Assignments (1-D spectral embedding)")
            axes[0].set_yticks([])
            plt.colorbar(sc0, ax=axes[0], label="Cluster")
            # bottom: t-SNE on original data
            sc1 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                                  c=clusters, cmap='tab10', alpha=0.7, s=20)
            axes[1].set_title("Cluster Assignments (t-SNE on original features)")
            axes[1].set_xlabel("t-SNE dim 1")
            axes[1].set_ylabel("t-SNE dim 2")
            plt.colorbar(sc1, ax=axes[1], label="Cluster")
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            sc = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                            c=clusters, cmap='tab10', alpha=0.7, s=20)
            ax.set_title("Cluster Assignments (t-SNE on original features)")
            ax.set_xlabel("t-SNE dim 1")
            ax.set_ylabel("t-SNE dim 2")
            plt.colorbar(sc, ax=ax, label="Cluster")

        plt.tight_layout()
        plt.show()
        return

def plot_dag(A, var_names=None, contracted_pairs=None):
    """
    Draw the estimated causal DAG.

    Parameters
    ----------
    A                : weighted adjacency matrix
    var_names        : list of original variable name strings
    contracted_pairs : list of (i, j) pairs that were contracted.
                       The surviving node (min(i,j)) will be relabelled
                       "name_i / name_j" and its edges drawn dashed.
    """
    q = A.shape[0]
    labels = {i: (var_names[i] if var_names else f"X{i}") for i in range(q)}

    # ── FIX-1: build relabelled names for contracted nodes ────────────────────
    contracted_nodes = set()
    if contracted_pairs:
        for (i, j) in contracted_pairs:
            u_keep = min(i, j)
            u_drop = max(i, j)
            name_i = var_names[i] if var_names and i < len(var_names) else f"X{i}"
            name_j = var_names[j] if var_names and j < len(var_names) else f"X{j}"
            # Relabel the surviving node to show both names
            labels[u_keep] = f"{name_i}/{name_j}"
            contracted_nodes.add(u_keep)

    G = nx.DiGraph()
    G.add_nodes_from(range(q))
    contracted_edges = set()
    for i in range(q):
        for j in range(q):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=round(A[i, j], 3))
                # Mark edges incident to contracted nodes
                if i in contracted_nodes or j in contracted_nodes:
                    contracted_edges.add((i, j))

    pos = nx.spring_layout(G, seed=42)

    # Split edges for different styles
    normal_edges     = [(u, v) for u, v in G.edges() if (u, v) not in contracted_edges]
    contracted_edges = [(u, v) for u, v in G.edges() if (u, v) in contracted_edges]

    edge_labels = {(u, v): f"{d['weight']:.2f}"
                   for u, v, d in G.edges(data=True)}

    # Node colours: highlight contracted surviving nodes
    node_colors = ['tomato' if i in contracted_nodes else 'steelblue'
                   for i in range(q)]

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900)
    nx.draw_networkx_labels(G, pos, labels=labels, font_color='white', font_size=8)

    if normal_edges:
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges,
                               edge_color='gray', arrows=True,
                               arrowsize=20, width=1.5)
    if contracted_edges:
        nx.draw_networkx_edges(G, pos, edgelist=contracted_edges,
                               edge_color='tomato', style='dashed',
                               arrows=True, arrowsize=20, width=1.5)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    # Legend
    legend_handles = [
        mpatches.Patch(color='steelblue', label='Normal node'),
        mpatches.Patch(color='tomato', label='Contracted node (merged)'),
        plt.Line2D([0], [0], color='gray',   linestyle='-',  label='Normal edge'),
        plt.Line2D([0], [0], color='tomato', linestyle='--', label='Contracted-node edge'),
    ]
    plt.legend(handles=legend_handles, loc='upper left', fontsize=7)
    plt.title("Estimated Causal DAG (edge weights = |ATE|)\n"
              "Tomato nodes/edges = contracted (ambiguous orientation resolved by merging)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 1a  –  PC ALGORITHM
# ═════════════════════════════════════════════════════════════════════════════

def run_pc(data, alpha=0.05):
    """Run PC Algorithm and return a binary directed adjacency matrix."""
    result = pc(data, indep_test=fisherz, alpha=alpha,
                uc_rule=0, uc_priority=2,
                mvpc=False, verbose=False, show_progress=False)
    G = result.G.graph
    d = G.shape[0]
    adj = np.zeros((d, d), dtype=int)

    for i in range(d):
        for j in range(d):
            if G[i, j] == -1 and G[j, i] == -1:
                adj[i, j] = adj[j, i] = 1          # undirected
            elif G[i, j] == 1 and G[j, i] == -1:
                adj[i, j] = 1                       # i → j
            elif G[i, j] == -1 and G[j, i] == 1:
                adj[j, i] = 1                       # j → i
    return adj


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 1b  –  BOOTSTRAP STABILITY  (FIX-4)
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_pc(data, B=100, alpha=0.05, theta=0.50, seed=42):
    """
    Bootstrap the PC Algorithm B times.

    FIX-4: undirected edges now correctly increment f_count for BOTH
    endpoints so that the inclusion frequency is symmetric and edges that are
    consistently undirected still pass the stability threshold.

    Returns
    -------
    f_uv : (d, d) – skeleton inclusion frequency  (symmetric)
    g_uv : (d, d) – directed orientation frequency  (g[i,j] = P(i→j))
    rho  : (d, d) – orientation confidence  rho[i,j] = g[i,j] / f[i,j]
    """
    rng = np.random.default_rng(seed)
    n, d = data.shape
    f_count = np.zeros((d, d))
    g_count = np.zeros((d, d))

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        try:
            G = run_pc(data[idx], alpha=alpha)
        except Exception:
            continue

        for i in range(d):
            for j in range(i + 1, d):          # iterate upper triangle only
                if G[i, j] != 0 or G[j, i] != 0:
                    # Edge present in skeleton: increment both directions
                    f_count[i, j] += 1
                    f_count[j, i] += 1          # FIX-4: symmetric count
                if G[i, j] == 1 and G[j, i] == 0:
                    g_count[i, j] += 1          # i → j
                elif G[j, i] == 1 and G[i, j] == 0:
                    g_count[j, i] += 1          # j → i

    f_uv = f_count / B
    g_uv = g_count / B

    with np.errstate(invalid="ignore"):
        rho = np.where(f_uv > 0, g_uv / f_uv, 0.0)

    return f_uv, g_uv, rho


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 1c  –  LiNGAM + OVS
# ═════════════════════════════════════════════════════════════════════════════

def _jarque_bera_fraction(data, alpha=0.10):
    """Fraction of variables whose OLS residuals reject Gaussianity (JB test)."""
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


def run_lingam(data):
    """Fit DirectLiNGAM and return its adjacency matrix B̂."""
    model = DirectLiNGAM()
    model.fit(data)
    return model.adjacency_matrix_


def compute_ovs(f_uv, g_uv, rho, B_hat, w_L=0.20, tau=0.15, theta=0.50):
    """Compute the Orientation Validation Score for each stable edge."""
    w_pc = 1.0 - w_L
    d = f_uv.shape[0]
    ovs_matrix = np.zeros((d, d))
    orientations = {}

    for i in range(d):
        for j in range(i + 1, d):
            if f_uv[i, j] < theta:
                continue

            delta_pc = rho[i, j] - rho[j, i]
            delta_L  = float(np.sign(B_hat[i, j] - B_hat[j, i]))
            ovs = w_pc * f_uv[i, j] * delta_pc + w_L * delta_L

            ovs_matrix[i, j] =  ovs
            ovs_matrix[j, i] = -ovs

            if ovs > tau:
                orientations[(i, j)] = (i, j)
            elif ovs < -tau:
                orientations[(i, j)] = (j, i)
            else:
                orientations[(i, j)] = None

    return orientations, ovs_matrix


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 1d  –  DAG RESOLUTION + ACYCLICITY
# ═════════════════════════════════════════════════════════════════════════════

def _has_cycle(G_nx: nx.DiGraph) -> bool:
    return not nx.is_directed_acyclic_graph(G_nx)


def _apply_meek_r2(G_nx: nx.DiGraph) -> nx.DiGraph:
    """Meek Rule R2: a→b→c and a–c (undirected) → orient a→c."""
    changed = True
    while changed:
        changed = False
        for a, b in list(G_nx.edges()):
            for c in list(G_nx.successors(b)):
                if c == a:
                    continue
                if not G_nx.has_edge(a, c) and not G_nx.has_edge(c, a):
                    continue
                if G_nx.has_edge(a, c) and not G_nx.has_edge(c, a):
                    continue
                if G_nx.has_edge(a, c) and G_nx.has_edge(c, a):
                    G_nx.remove_edge(c, a)
                    changed = True
    return G_nx


def resolve_with_acyclicity(orientations, d, ovs_matrix=None,
                             domain_order=None):
    """
    Resolve ambiguous edges via four-step hierarchy.

    Returns
    -------
    resolved         : dict  {(i,j): (u,v)}
    G_nx             : nx.DiGraph of the final DAG
    contracted_pairs : list of (i, j) pairs that were contracted
    """
    G_nx = nx.DiGraph()
    G_nx.add_nodes_from(range(d))
    resolved = {}
    ambiguous = []
    contracted_pairs = []          # FIX-1: track which pairs were contracted

    # ── Step 1 ────────────────────────────────────────────────────────────────
    for edge, direction in orientations.items():
        if direction is None:
            ambiguous.append(edge)
            continue
        u, v = direction
        G_nx.add_edge(u, v)
        if _has_cycle(G_nx):
            G_nx.remove_edge(u, v)
            G_nx.add_edge(v, u)
            if _has_cycle(G_nx):
                G_nx.remove_edge(v, u)
                ambiguous.append(edge)
            else:
                resolved[edge] = (v, u)
        else:
            resolved[edge] = direction

    # ── Step 2 ────────────────────────────────────────────────────────────────
    G_nx = _apply_meek_r2(G_nx)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    still_ambiguous = []
    if domain_order is not None:
        order_map = {v: k for k, v in enumerate(domain_order)}
        for (i, j) in ambiguous:
            if i in order_map and j in order_map:
                u, v = (i, j) if order_map[i] < order_map[j] else (j, i)
                G_nx.add_edge(u, v)
                if not _has_cycle(G_nx):
                    resolved[(i, j)] = (u, v)
                else:
                    G_nx.remove_edge(u, v)
                    still_ambiguous.append((i, j))
            else:
                still_ambiguous.append((i, j))
    else:
        still_ambiguous = ambiguous

    # ── Step 4: contraction ───────────────────────────────────────────────────
    if ovs_matrix is not None:
        still_ambiguous.sort(key=lambda e: abs(ovs_matrix[e[0], e[1]]))
    for (i, j) in still_ambiguous:
        if not G_nx.has_node(i) or not G_nx.has_node(j):
            continue
        u_keep = min(i, j)
        u_drop = max(i, j)
        if not G_nx.has_node(u_drop):
            continue
        for pred in list(G_nx.predecessors(u_drop)):
            if pred != u_keep and G_nx.has_node(pred) \
                    and not G_nx.has_edge(pred, u_keep):
                G_nx.add_edge(pred, u_keep)
        for succ in list(G_nx.successors(u_drop)):
            if succ != u_keep and G_nx.has_node(succ) \
                    and not G_nx.has_edge(u_keep, succ):
                G_nx.add_edge(u_keep, succ)
        G_nx.remove_node(u_drop)
        resolved[(i, j)] = (u_keep, -1)
        contracted_pairs.append((i, j))        # FIX-1: record contraction

    return resolved, G_nx, contracted_pairs


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 2  –  CAUSAL EDGE WEIGHT ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

def _reset_test(y, X_reg, alpha=0.05):
    """RESET test for functional form. True = linear (use OLS)."""
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
        p = 1 - __import__('scipy').stats.f.cdf(F, 2, df_den)
        return p > alpha
    except Exception:
        return True


def _ols_ate(y, t, Z):
    """OLS: coefficient on t after adjusting for Z."""
    if Z.shape[1] > 0:
        X_reg = sm.add_constant(np.column_stack([t, Z]))
        coef_idx = 1
    else:
        X_reg = sm.add_constant(t)
        coef_idx = 1
    try:
        res = sm.OLS(y, X_reg).fit()
        return float(res.params[coef_idx])
    except Exception:
        return float(LinearRegression().fit(t.reshape(-1, 1), y).coef_[0])


def _dml_ate(y, t, Z, K=5, seed=42):
    """DML with K-fold cross-fitting."""
    n = len(y)
    V_res = np.zeros(n)
    U_res = np.zeros(n)
    kf = KFold(n_splits=K, shuffle=True, random_state=seed)

    for tr, te in kf.split(np.arange(n)):
        X_tr, X_te = Z[tr], Z[te]
        y_tr, y_te = y[tr], y[te]
        t_tr, t_te = t[tr], t[te]

        if Z.shape[1] == 0:
            V_res[te] = y_te - y_tr.mean()
            U_res[te] = t_te - t_tr.mean()
            continue

        try:
            n_sp = min(20, max(4, len(tr) // 10))
            terms = sum(s(j, n_splines=n_sp) for j in range(Z.shape[1]))
            m_y = LinearGAM(terms).fit(X_tr, y_tr)
            m_t = LinearGAM(terms).fit(X_tr, t_tr)
            V_res[te] = y_te - m_y.predict(X_te)
            U_res[te] = t_te - m_t.predict(X_te)
        except Exception:
            rf_y = RandomForestRegressor(100, random_state=seed).fit(X_tr, y_tr)
            rf_t = RandomForestRegressor(100, random_state=seed).fit(X_tr, t_tr)
            V_res[te] = y_te - rf_y.predict(X_te)
            U_res[te] = t_te - rf_t.predict(X_te)

    denom = np.sum(U_res ** 2)
    return float(np.sum(U_res * V_res) / denom) if denom > 1e-12 else 0.0


def _backdoor_set(G_nx: nx.DiGraph, u: int, v: int) -> list[int]:
    """Parents of u that are not descendants of u."""
    if not G_nx.has_node(u) or not G_nx.has_node(v):
        return []
    try:
        desc_u = nx.descendants(G_nx, u)
    except Exception:
        desc_u = set()
    try:
        parents = [w for w in G_nx.predecessors(u)
                   if w not in desc_u and w != v and G_nx.has_node(w)]
    except nx.NetworkXError:
        parents = []
    return parents


def estimate_weights(data, resolved_edges, G_nx, f_uv,
                     alpha_reset=0.05, dml_k=5):
    """Estimate bootstrap-stability-weighted |ATE| for every directed edge."""
    d = data.shape[1]
    A = np.zeros((d, d))

    for (i, j), direction in resolved_edges.items():
        u, v = direction
        if v == -1:
            continue
        if u >= d or v >= d:
            continue
        if not G_nx.has_node(u) or not G_nx.has_node(v):
            continue

        t  = data[:, u]
        y  = data[:, v]
        bd = _backdoor_set(G_nx, u, v)
        bd = [b for b in bd if b < d and G_nx.has_node(b)]
        Z  = data[:, bd] if bd else np.empty((len(y), 0))

        X_reg = np.column_stack([t, Z]) if Z.shape[1] > 0 else t.reshape(-1, 1)
        linear = _reset_test(y, X_reg, alpha=alpha_reset)

        ate = _ols_ate(y, t, Z) if linear else _dml_ate(y, t, Z, K=dml_k)
        fij = f_uv[u, v] if u < f_uv.shape[0] and v < f_uv.shape[1] else 1.0
        A[u, v] = fij * abs(ate)

    if A.sum() == 0:
        print("[CSC-PC] WARNING: Adjacency matrix is all-zero. "
              "No causal structure was recovered.")
    return A


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 3  –  CHUNG LAPLACIAN
# ═════════════════════════════════════════════════════════════════════════════

def compute_stationary_distribution(P):
    """Solve π = πP via left eigenvector for eigenvalue 1."""
    eigvals, eigvecs = eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx])
    pi = np.abs(pi)
    pi /= pi.sum()
    return pi


def compute_laplacian(A, alpha=0.15):
    """PageRank-normalise A → P, compute π, P*, and Chung Laplacian L."""
    d = A.shape[0]
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    P0 = A / row_sums_safe
    P0[row_sums.flatten() == 0] = 1.0 / d
    P = (1 - alpha) * P0 + (alpha / d) * np.ones((d, d))

    pi = compute_stationary_distribution(P)
    pi_safe = np.where(pi > 1e-12, pi, 1e-12)

    P_star = (pi[np.newaxis, :] / pi_safe[:, np.newaxis]) * P.T

    L = np.eye(d) - 0.5 * (P + P_star)
    L = 0.5 * (L + L.T)
    return L, P, pi


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 3b  –  SPECTRAL GAP  →  K*  (FIX-2 + FIX-5)
# ═════════════════════════════════════════════════════════════════════════════

def spectral_gap(L):
    """
    Eigendecompose L and select K* as the index of the largest spectral gap.

    FIX-2: The gap between eigenvalues 0 and 1 (index 0 in `gaps`) is the
    trivial algebraic-connectivity gap and almost always dominates for a
    nearly-connected graph, forcing K*=1.  We now search for the largest gap
    among gaps[1:] (i.e. starting from the second gap) so that the embedding
    dimension reflects genuine causal community structure.

    FIX-5: Guard against the degenerate case where all non-trivial gaps are
    equal (flat spectrum), which would give argmax=0; we default K*=2 there.
    """
    eigvals, eigvecs = np.linalg.eigh(L)
    order   = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    gaps = np.diff(eigvals)

    if len(gaps) <= 1:
        # Degenerate: only 2 eigenvalues
        K_star = 1
    else:
        # FIX-2: skip trivial gap at index 0
        non_trivial_gaps = gaps[1:]
        best_idx = int(np.argmax(non_trivial_gaps))   # index into gaps[1:]
        K_star = best_idx + 2                          # +1 for 0-based, +1 for skip
        K_star = max(2, K_star)                        # FIX-5: at least 2

    # Clamp to available eigenvectors (skip v_1)
    K_star = min(K_star, L.shape[0] - 2)
    K_star = max(1, K_star)

    V_K = eigvecs[:, 1:K_star + 1]    # skip trivial eigenvector v_1
    return V_K, K_star, eigvals


# ═════════════════════════════════════════════════════════════════════════════
#  STAGE 5  –  CLUSTERING  +  CLUSTER COUNT SELECTION
# ═════════════════════════════════════════════════════════════════════════════

def _gap_statistic(X_emb, k, B_ref=50, seed=42):
    """Gap statistic for a given k."""
    rng  = np.random.default_rng(seed)
    km   = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X_emb)
    wcss = km.inertia_

    ref_logs = []
    lo, hi = X_emb.min(axis=0), X_emb.max(axis=0)
    for _ in range(B_ref):
        Xr = rng.uniform(lo, hi, size=X_emb.shape)
        ref_logs.append(
            np.log(KMeans(n_clusters=k, n_init=5,
                          random_state=seed).fit(Xr).inertia_ + 1e-12)
        )
    gap = np.mean(ref_logs) - np.log(wcss + 1e-12)
    sdk = np.std(ref_logs) * np.sqrt(1 + 1 / B_ref)
    return gap, sdk


def embed_and_cluster(data, V_K, K_star, B_ref=50, seed=42):
    """Project data onto V_K and select k* via Silhouette + Gap statistic."""
    X_embed = data @ V_K

    k_range = range(2, min(K_star + 3, X_embed.shape[0]))
    sil_scores = {}
    gap_scores = {}
    gap_sds    = {}

    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=20,
                        random_state=seed).fit_predict(X_embed)
        sil_scores[k] = (silhouette_score(X_embed, labels)
                         if len(set(labels)) > 1 else -1)
        gap_scores[k], gap_sds[k] = _gap_statistic(X_embed, k,
                                                    B_ref=B_ref, seed=seed)

    k_sil = max(sil_scores, key=sil_scores.get)

    k_gap  = k_sil
    k_list = sorted(gap_scores)
    for idx, k in enumerate(k_list[:-1]):
        k_next = k_list[idx + 1]
        if gap_scores[k] >= gap_scores[k_next] - gap_sds[k_next]:
            k_gap = k
            break

    k_star = k_sil
    clusters = KMeans(n_clusters=k_star, n_init=50,
                      random_state=seed).fit_predict(X_embed)

    diagnostics = {
        "k_silhouette" : k_sil,
        "k_gap"        : k_gap,
        "sil_scores"   : sil_scores,
        "gap_scores"   : gap_scores,
        "criteria_agree": k_sil == k_gap,
    }
    return clusters, X_embed, k_star, diagnostics


# ═════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def CSC_PC_PIPELINE(
    data,
    var_names=None,
    alpha_ci=0.05,
    B=100,
    theta=0.50,
    tau=0.15,
    w_L_max=0.20,
    alpha_pr=0.15,
    alpha_reset=0.05,
    dml_k=5,
    domain_order=None,
    B_ref=50,
    seed=42,
    visualise=True,
    verbose=True,
):

    def log(msg):
        if verbose:
            print(f"[CSC-PC] {msg}")

    data = np.array(data, dtype=float)
    # Keep a copy of the standardised data BEFORE pipeline re-standardises
    # (used for t-SNE in plot_clusters — FIX-3)
    data_for_tsne = StandardScaler().fit_transform(data)

    data = StandardScaler().fit_transform(data)
    n, d = data.shape

    # ── Stage 1a ──────────────────────────────────────────────────────────────
    log("Stage 1a: PC Algorithm …")
    _ = run_pc(data, alpha=alpha_ci)

    # ── Stage 1b ──────────────────────────────────────────────────────────────
    log(f"Stage 1b: Bootstrap stability ({B} resamples) …")
    f_uv, g_uv, rho = bootstrap_pc(data, B=B, alpha=alpha_ci,
                                    theta=theta, seed=seed)

    # ── Stage 1c ──────────────────────────────────────────────────────────────
    log("Stage 1c: LiNGAM diagnostic + OVS …")
    ng_frac = _jarque_bera_fraction(data)
    w_L = min(ng_frac, 1.0) * w_L_max
    log(f"  Non-Gaussian fraction: {ng_frac:.2f} → w_L = {w_L:.3f}")

    B_hat = run_lingam(data) if w_L > 0 else np.zeros((d, d))
    orientations, ovs_matrix = compute_ovs(
        f_uv, g_uv, rho, B_hat, w_L=w_L, tau=tau, theta=theta,
    )
    n_oriented  = sum(1 for v in orientations.values() if v is not None)
    n_ambiguous = sum(1 for v in orientations.values() if v is None)
    log(f"  Oriented: {n_oriented}, Ambiguous: {n_ambiguous}")

    # ── Stage 1d ──────────────────────────────────────────────────────────────
    log("Stage 1d: Orientation resolution …")
    # FIX-1: resolve_with_acyclicity now returns contracted_pairs
    resolved_edges, G_nx, contracted_pairs = resolve_with_acyclicity(
        orientations, d, ovs_matrix=ovs_matrix, domain_order=domain_order,
    )
    log(f"  Final DAG: {G_nx.number_of_nodes()} nodes, "
        f"{G_nx.number_of_edges()} edges")
    if contracted_pairs:
        cp_names = []
        for (i, j) in contracted_pairs:
            ni = var_names[i] if var_names and i < len(var_names) else f"X{i}"
            nj = var_names[j] if var_names and j < len(var_names) else f"X{j}"
            cp_names.append(f"{ni}/{nj}")
        log(f"  Contracted pairs: {cp_names}")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    log("Stage 2: Estimating causal edge weights …")
    A = estimate_weights(data, resolved_edges, G_nx, f_uv,
                         alpha_reset=alpha_reset, dml_k=dml_k)
    log(f"  A_stable range: [{A.min():.4f}, {A.max():.4f}]")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    log("Stage 3: Building Chung Laplacian …")
    L, P, pi = compute_laplacian(A, alpha=alpha_pr)

    # ── Stage 3b ──────────────────────────────────────────────────────────────
    log("Stage 3b: Selecting embedding dimension K* …")
    V_K, K_star, eigvals = spectral_gap(L)   # FIX-2 applied inside
    gaps = np.diff(eigvals)
    log(f"  K* = {K_star}  (largest non-trivial gap = "
        f"{gaps[K_star - 1] if K_star - 1 < len(gaps) else float('nan'):.4f})")

    # ── Stages 4–5 ────────────────────────────────────────────────────────────
    log("Stages 4–5: Embedding and clustering …")
    clusters, X_embed, k_star, diagnostics = embed_and_cluster(
        data, V_K, K_star, B_ref=B_ref, seed=seed,
    )
    log(f"  k* = {k_star}  "
        f"(Silhouette k={diagnostics['k_silhouette']}, "
        f"Gap k={diagnostics['k_gap']}, "
        f"agree={diagnostics['criteria_agree']})")
    log("Done.")

    # ── Visualisation ─────────────────────────────────────────────────────────
    if visualise:
        plot_spectral_gap(L)
        plot_silhouette(X_embed)
        # FIX-3: pass X_orig so t-SNE runs on full feature matrix
        plot_clusters(X_embed, clusters, X_orig=data_for_tsne)
        # FIX-1: pass contracted_pairs so DAG relabels them
        plot_dag(A, var_names=var_names, contracted_pairs=contracted_pairs)

    return {
        "clusters"         : clusters,
        "embedding"        : X_embed,
        "adjacency"        : A,
        "laplacian"        : L,
        "laplacian_P"      : P,
        "pi"               : pi,
        "edges"            : resolved_edges,
        "K_star"           : K_star,
        "k_star"           : k_star,
        "eigenvalues"      : eigvals,
        "diagnostics"      : diagnostics,
        "contracted_pairs" : contracted_pairs,       # FIX-1: exposed in result
        "ovs_matrix"       : ovs_matrix,
        "f_uv"             : f_uv,
        "g_uv"             : g_uv,
    }
