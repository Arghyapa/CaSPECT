"""
CaSPECT -- Causal Spectral Clustering
=====================================
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import networkx as nx
from scipy import stats
from scipy.linalg import eig, expm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera

# ─────────────────────────────────────────────────────────────────────────────
#  Optional dependencies
# ─────────────────────────────────────────────────────────────────────────────
try:
    from pygam import LinearGAM, s as gam_s
    HAS_GAM = True
except ImportError:  # pragma: no cover
    HAS_GAM = False

try:
    from lingam import DirectLiNGAM
    HAS_LINGAM = True
except ImportError:  # pragma: no cover
    HAS_LINGAM = False

try:
    from causallearn.search.ConstraintBased.PC import pc as _cl_pc
    from causallearn.utils.cit import fisherz as _cl_fisherz
    HAS_CAUSALLEARN = True
except ImportError:  # pragma: no cover
    HAS_CAUSALLEARN = False


# Hyperparameters used throughout the paper (Table 2).
DEFAULTS = dict(
    alpha_ci=0.05,     # alpha_n      PC conditional-independence level
    B=100,             # B            bootstrap resamples for PC
    theta=0.55,        # theta        skeleton stability threshold
    gamma=0.15,        # gamma        edge orientation threshold
    w_max=0.20,        # w_max        cap on the LiNGAM weight
    alpha_jb=0.10,     # alpha_JB     Jarque-Bera level
    alpha_reset=0.05,  # alpha_RESET  RESET test level
    alpha_pr=0.15,     # alpha_PR     PageRank teleportation
    n_fold=5,          # n_fold       DML cross-fitting folds
    k_max=9,           # k_max        largest k searched
)

def _pc_causallearn(data: np.ndarray, alpha: float) -> np.ndarray:
    """PC via causal-learn, decoded into the adjacency convention above.

    causal-learn encodes i -> j as G[i, j] = -1 and G[j, i] = 1, and i -- j as
    G[i, j] = G[j, i] = -1.  A bidirected mark (both +1, conflicting
    v-structures) is treated as undirected.
    """
    cg = _cl_pc(data, alpha=alpha, indep_test=_cl_fisherz, uc_rule=0,
                uc_priority=2, mvpc=False, verbose=False, show_progress=False)
    G = cg.G.graph
    d = G.shape[0]
    adj = np.zeros((d, d), dtype=int)
    for i in range(d):
        for j in range(i + 1, d):
            a, b = G[i, j], G[j, i]
            if a == -1 and b == 1:          # i -> j
                adj[i, j] = 1
            elif a == 1 and b == -1:        # j -> i
                adj[j, i] = 1
            elif a != 0 or b != 0:          # i -- j  (or i <-> j)
                adj[i, j] = adj[j, i] = 1
    return adj


def _fisher_z_pvalue(C: np.ndarray, n: int, i: int, j: int, S) -> float:
    idx = [i, j, *S]
    P = np.linalg.pinv(C[np.ix_(idx, idx)])
    denom = np.sqrt(abs(P[0, 0] * P[1, 1])) + 1e-300
    r = float(np.clip(-P[0, 1] / denom, -1 + 1e-12, 1 - 1e-12))
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(max(n - len(S) - 3, 1))
    return float(2 * stats.norm.sf(abs(z)))


def pc_simple(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Self-contained PC (Fisher-z skeleton, v-structures, Meek R1-R4).

    Used only when causal-learn is unavailable or fails.
    """
    n, d = data.shape
    C = np.corrcoef(data, rowvar=False)
    adj = ~np.eye(d, dtype=bool)
    sepset: dict = {}
    level = 0
    while True:
        tested = False
        for i in range(d):
            for j in range(d):
                if i == j or not adj[i, j]:
                    continue
                nbrs = [k for k in range(d) if k not in (i, j) and adj[i, k]]
                if len(nbrs) < level:
                    continue
                tested = True
                for S in itertools.combinations(nbrs, level):
                    if _fisher_z_pvalue(C, n, i, j, S) > alpha:
                        adj[i, j] = adj[j, i] = False
                        sepset[frozenset((i, j))] = set(S)
                        break
        if not tested:
            break
        level += 1

    skeleton = [(i, j) for i in range(d) for j in range(i + 1, d) if adj[i, j]]
    pdag = _PDAG(d, skeleton)
    pdag.U = {frozenset(e) for e in skeleton}
    for i, j in itertools.combinations(range(d), 2):        # v-structures
        if adj[i, j]:
            continue
        for k in range(d):
            if adj[i, k] and adj[j, k] and k not in sepset.get(frozenset((i, j)), set()):
                for x in (i, j):
                    if pdag.is_undirected(x, k):
                        pdag.orient(x, k)
    apply_meek_rules(pdag, check_acyclic=False)
    return pdag.to_matrix()


def run_pc(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Run PC at level alpha and return the PC adjacency matrix (see module doc)."""
    if HAS_CAUSALLEARN:
        try:
            return _pc_causallearn(data, alpha)
        except Exception:
            pass
    return pc_simple(data, alpha)


def bootstrap_pc(data: np.ndarray, B: int = 100, alpha: float = 0.05,
                 seed: int = 42):
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
        f_count += ((G + G.T) > 0)
        g_count += (G == 1) & (G.T == 0)
    np.fill_diagonal(f_count, 0)
    np.fill_diagonal(g_count, 0)
    f_uv = f_count / B
    f_dir = g_count / B
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(f_uv > 0, f_dir / np.where(f_uv > 0, f_uv, 1.0), 0.0)
    return f_uv, f_dir, rho

def jarque_bera_count(data: np.ndarray, alpha_jb: float = 0.10) -> int:
    """n_NG: number of variables whose OLS residuals (each variable regressed
    on all others) reject Gaussianity at level alpha_JB."""
    n, d = data.shape
    count = 0
    for j in range(d):
        others = [k for k in range(d) if k != j]
        Xo = sm.add_constant(data[:, others], has_constant="add")
        try:
            resid = sm.OLS(data[:, j], Xo).fit().resid
            if jarque_bera(resid)[1] < alpha_jb:
                count += 1
        except Exception:
            pass
    return count


def run_lingam(data: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """DirectLiNGAM coefficients in the paper's orientation, B_L[u, v] = B_vu.

    lingam's ``adjacency_matrix_`` follows x = Bx + e, where B[v, u] is the
    coefficient of x_u in the equation of x_v; we return its transpose so that
    B_L[u, v] refers to the edge u -> v (Section 6, Step 1).
    """
    d = data.shape[1]
    if not HAS_LINGAM:
        warnings.warn("lingam is not installed; the LiNGAM term of the OVS is set to 0.")
        return np.zeros((d, d))
    model = DirectLiNGAM(random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)
    return np.asarray(model.adjacency_matrix_, dtype=float).T


def lingam_weight(n_ng: int, q: int, w_max: float = 0.20) -> float:
    """w_L = (n_NG / q) * w_max."""
    return (n_ng / q) * w_max


def compute_ovs(f_uv: np.ndarray, rho: np.ndarray, B_L: np.ndarray,
                w_L: float, gamma: float = 0.15, theta: float = 0.55,
                use_lingam: bool = True):
    """Orientation Validation Score (Eq. 2) on the stable skeleton.

        OVS_uv = w_PC * f_uv * e_PC + w_L * e_L,       w_PC = 1 - w_L,
        e_PC   = rho_uv - rho_vu,
        e_L    = sign(|B_L_uv| - |B_L_vu|).

    ``use_lingam=False`` forces e_L = 0 for every edge while keeping
    w_PC = 1 - w_L (ablation A1, Section 7).

    Returns
    -------
    orientations : {(i, j): (u, v) or None} for every (i < j) in the stable
                   skeleton S = {f_uv >= theta}; None means |OVS| <= gamma.
    ovs_matrix   : antisymmetric matrix with ovs_matrix[i, j] = OVS_ij.
    """
    w_pc = 1.0 - w_L
    d = f_uv.shape[0]
    ovs_matrix = np.zeros((d, d))
    orientations = {}
    for i in range(d):
        for j in range(i + 1, d):
            if f_uv[i, j] < theta or f_uv[i, j] <= 0:
                continue
            e_pc = rho[i, j] - rho[j, i]
            e_l = float(np.sign(abs(B_L[i, j]) - abs(B_L[j, i]))) if use_lingam else 0.0
            ovs = w_pc * f_uv[i, j] * e_pc + w_L * e_l
            ovs_matrix[i, j], ovs_matrix[j, i] = ovs, -ovs
            if ovs > gamma:
                orientations[(i, j)] = (i, j)
            elif ovs < -gamma:
                orientations[(i, j)] = (j, i)
            else:
                orientations[(i, j)] = None
    return orientations, ovs_matrix


class _PDAG:
    """Partially directed graph over a fixed skeleton."""

    def __init__(self, d: int, skeleton):
        self.d = d
        self.adj = {frozenset(e) for e in skeleton}
        self.D = nx.DiGraph()
        self.D.add_nodes_from(range(d))
        self.U: set = set()

    def adjacent(self, a, b) -> bool:
        return frozenset((a, b)) in self.adj

    def is_undirected(self, a, b) -> bool:
        return frozenset((a, b)) in self.U

    def creates_cycle(self, a, b) -> bool:
        """Would adding a -> b close a directed cycle?"""
        return nx.has_path(self.D, b, a)

    def orient(self, a, b):
        self.U.discard(frozenset((a, b)))
        self.D.add_edge(a, b)

    def to_matrix(self) -> np.ndarray:
        G = np.zeros((self.d, self.d), dtype=int)
        for e in self.U:
            a, b = tuple(e)
            G[a, b] = G[b, a] = 1
        for a, b in self.D.edges():
            G[a, b] = 1
        return G


def _meek_rule(p: _PDAG, a: int, b: int) -> int:
    """Return the Meek rule (1-4) that orients the undirected edge a -- b as
    a -> b, or 0 if none applies."""
    # R1: c -> a, c and b non-adjacent.
    for c in p.D.predecessors(a):
        if c != b and not p.adjacent(c, b):
            return 1
    # R2: a -> c -> b.
    for c in p.D.successors(a):
        if p.D.has_edge(c, b):
            return 2
    # R3: a -- c -> b and a -- e -> b with c, e non-adjacent.
    cs = [c for c in p.D.predecessors(b) if c != a and p.is_undirected(a, c)]
    for c, e in itertools.combinations(cs, 2):
        if not p.adjacent(c, e):
            return 3
    # R4: a -- c -> e -> b with c, b non-adjacent and a, e adjacent.
    for c in range(p.d):
        if c in (a, b) or not p.is_undirected(a, c) or p.adjacent(c, b):
            continue
        for e in p.D.successors(c):
            if e not in (a, b) and p.D.has_edge(e, b) and p.adjacent(a, e):
                return 4
    return 0


def apply_meek_rules(p: _PDAG, check_acyclic: bool = True, log: Optional[dict] = None):
    """Apply Meek's rules R1-R4 exhaustively; with ``check_acyclic`` an
    orientation is accepted only if it does not close a directed cycle."""
    changed = True
    while changed:
        changed = False
        for e in sorted(p.U, key=sorted):
            if e not in p.U:
                continue
            i, j = sorted(e)
            for a, b in ((i, j), (j, i)):
                r = _meek_rule(p, a, b)
                if r and not (check_acyclic and p.creates_cycle(a, b)):
                    p.orient(a, b)
                    if log is not None:
                        log[(i, j)] = f"meek_R{r}"
                    changed = True
                    break


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def notears_h(W: np.ndarray) -> float:
    """NOTEARS acyclicity function h(W) = tr(exp(W o W)) - d; zero iff DAG."""
    W = np.asarray(W, dtype=float)
    return float(np.trace(expm(W * W)) - W.shape[0])


@dataclass
class Resolution:
    """Fully oriented DAG G_hat = (V*, E_hat) produced by Step 1."""
    d: int
    skeleton: list            # stable skeleton S_hat, (i, j) with i < j
    final_direction: dict     # (i, j) -> (u, v), or None if contracted/symmetrised
    step: dict                # (i, j) -> how the edge was resolved
    groups: list              # V*: tuples of original indices (N_S has |S| > 1)
    group_of: np.ndarray      # original index -> index in V*
    G_star: nx.DiGraph        # DAG over V*
    member_edges: dict        # (a, b) in G_star -> original directed edges (u, v)
    contracted_pairs: list    # pairs merged in Step 1.4 or by cycle breaking
    symmetric_pairs: list     # ablation A3: residual ambiguous edges kept symmetric

    def node_names(self, var_names: Sequence[str]) -> list:
        return ["/".join(var_names[m] for m in g) for g in self.groups]

    def merged_nodes(self) -> list:
        return [k for k, g in enumerate(self.groups) if len(g) > 1]

    def backdoor_members(self, a: int, exclude: Optional[int] = None) -> list:
        """Original variables in the parents of node a of G_hat (the backdoor
        set W read off G_hat), excluding node ``exclude``."""
        out = []
        for p in self.G_star.predecessors(a):
            if p != exclude:
                out.extend(self.groups[p])
        return sorted(out)


def _contract(D: nx.DiGraph, uf: _UnionFind, d: int):
    members: dict = {}
    for v in range(d):
        members.setdefault(uf.find(v), []).append(v)
    groups = sorted((tuple(sorted(g)) for g in members.values()), key=lambda g: g[0])
    group_of = np.empty(d, dtype=int)
    for k, g in enumerate(groups):
        group_of[list(g)] = k
    G = nx.DiGraph()
    G.add_nodes_from(range(len(groups)))
    member_edges: dict = {}
    for u, v in D.edges():
        a, b = int(group_of[u]), int(group_of[v])
        if a != b:
            G.add_edge(a, b)
            member_edges.setdefault((a, b), []).append((u, v))
    return G, member_edges, groups, group_of


def resolve_orientations(orientations: dict, d: int, ovs_matrix: np.ndarray,
                         domain_order: Optional[Sequence[int]] = None,
                         ambiguous: str = "contract") -> Resolution:
    if ambiguous not in ("contract", "symmetrise"):
        raise ValueError("ambiguous must be 'contract' or 'symmetrise'")
    skeleton = sorted(orientations)
    p = _PDAG(d, skeleton)
    step: dict = {}
    for e, direction in orientations.items():
        if direction is None:
            p.U.add(frozenset(e))
        else:
            p.D.add_edge(*direction)
            step[e] = "ovs"

    # 1. Acyclicity filter
    changed = True
    while changed:
        changed = False
        for e in sorted(p.U, key=sorted):
            i, j = sorted(e)
            c_ij, c_ji = p.creates_cycle(i, j), p.creates_cycle(j, i)
            if c_ij != c_ji:
                p.orient(*((j, i) if c_ij else (i, j)))
                step[(i, j)] = "acyclicity"
                changed = True

    # 2. Meek rules
    apply_meek_rules(p, check_acyclic=True, log=step)

    # 3. Domain ordering
    if domain_order is not None:
        rank = {v: r for r, v in enumerate(domain_order)}
        for e in sorted(p.U, key=sorted):
            i, j = sorted(e)
            if i in rank and j in rank and rank[i] != rank[j]:
                u, v = (i, j) if rank[i] < rank[j] else (j, i)
                if not p.creates_cycle(u, v):
                    p.orient(u, v)
                    step[(i, j)] = "domain_order"

    # 4. Contraction (or symmetrisation for A3)
    uf = _UnionFind(d)
    contracted, symmetric = [], []
    residual = sorted((tuple(sorted(e)) for e in p.U),
                      key=lambda e: (abs(ovs_matrix[e]), e))
    for i, j in residual:
        if ambiguous == "symmetrise":
            symmetric.append((i, j))
            step[(i, j)] = "symmetrised"
        else:
            uf.union(i, j)
            contracted.append((i, j))
            step[(i, j)] = "contracted"

    # Global acyclicity check (NOTEARS) and cycle breaking
    while True:
        G_star, member_edges, groups, group_of = _contract(p.D, uf, d)
        W = nx.to_numpy_array(G_star, nodelist=range(len(groups)))
        if notears_h(W) <= 1e-8 and nx.is_directed_acyclic_graph(G_star):
            break
        cycle = nx.find_cycle(G_star)
        candidates = [e for (a, b, *_) in cycle for e in member_edges[(a, b)]]
        u, v = min(candidates, key=lambda e: (abs(ovs_matrix[e]), e))
        uf.union(u, v)
        key = (min(u, v), max(u, v))
        contracted.append(key)
        step[key] = "cycle_contracted"

    symmetric = [(i, j) for (i, j) in symmetric if group_of[i] != group_of[j]]
    final_direction = {}
    for i, j in skeleton:
        if (i, j) in symmetric or group_of[i] == group_of[j]:
            final_direction[(i, j)] = None
        elif p.D.has_edge(i, j):
            final_direction[(i, j)] = (i, j)
        elif p.D.has_edge(j, i):
            final_direction[(i, j)] = (j, i)
        else:
            final_direction[(i, j)] = None

    return Resolution(d=d, skeleton=skeleton, final_direction=final_direction,
                      step=step, groups=groups, group_of=group_of, G_star=G_star,
                      member_edges=member_edges, contracted_pairs=contracted,
                      symmetric_pairs=symmetric)


def reset_test_pvalue(y: np.ndarray, X_reg: np.ndarray) -> float:
    """Ramsey RESET p-value: adds the squared and cubed fitted values of the
    OLS regression of y on X_reg and F-tests their joint significance."""
    try:
        Xc = sm.add_constant(X_reg, has_constant="add")
        res0 = sm.OLS(y, Xc).fit()
        yhat = res0.fittedvalues
        sd = yhat.std()
        if sd < 1e-12:
            return 1.0
        z = (yhat - yhat.mean()) / sd          # same column space, better conditioned
        X_aug = np.column_stack([Xc, z ** 2, z ** 3])
        res1 = sm.OLS(y, X_aug).fit()
        df_den = len(y) - X_aug.shape[1]
        if df_den <= 0 or res1.ssr < 1e-12:
            return 1.0
        F = ((res0.ssr - res1.ssr) / 2) / (res1.ssr / df_den)
        return float(stats.f.sf(F, 2, df_den))
    except Exception:
        return 1.0


def ols_ace(y: np.ndarray, t: np.ndarray, W: np.ndarray) -> float:
    """Coefficient on t in the OLS regression of y on (t, W)."""
    X = np.column_stack([t, W]) if W.shape[1] else t.reshape(-1, 1)
    return float(sm.OLS(y, sm.add_constant(X, has_constant="add")).fit().params[1])


def _fit_nuisance(W_tr, y_tr, seed: int, learner: str):
    if learner == "gam" and HAS_GAM:
        try:
            n_sp = min(20, max(4, len(y_tr) // 10))
            terms = gam_s(0, n_splines=n_sp)
            for j in range(1, W_tr.shape[1]):
                terms = terms + gam_s(j, n_splines=n_sp)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gam = LinearGAM(terms).fit(W_tr, y_tr)
            return gam.predict
        except Exception:
            pass
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                               random_state=seed, n_jobs=1).fit(W_tr, y_tr)
    return rf.predict


def dml_ace(y: np.ndarray, t: np.ndarray, W: np.ndarray, n_fold: int = 5,
            seed: int = 42, learner: str = "gam", return_se: bool = False):
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    W = np.asarray(W, float).reshape(len(y), -1)
    n = len(y)
    r_y, r_t = np.zeros(n), np.zeros(n)
    for tr, te in KFold(n_splits=n_fold, shuffle=True, random_state=seed).split(y):
        if W.shape[1] == 0:
            r_y[te] = y[te] - y[tr].mean()
            r_t[te] = t[te] - t[tr].mean()
            continue
        m_y = _fit_nuisance(W[tr], y[tr], seed, learner)
        m_t = _fit_nuisance(W[tr], t[tr], seed, learner)
        r_y[te] = y[te] - m_y(W[te])
        r_t[te] = t[te] - m_t(W[te])
    denom = np.sum(r_t ** 2)
    est = float(np.sum(r_t * r_y) / denom) if denom > 1e-12 else 0.0
    if not return_se:
        return est
    J = np.mean(r_t ** 2)
    psi = (r_y - est * r_t) * r_t
    se = float(np.sqrt(np.mean(psi ** 2) / (J ** 2) / n)) if J > 1e-12 else np.nan
    return est, se


def edge_ace(data: np.ndarray, u: int, v: int, W_idx: Sequence[int],
             alpha_reset: float = 0.05, n_fold: int = 5, seed: int = 42):
    """tau_hat for u -> v with backdoor set W: OLS if RESET (p > alpha_RESET)
    does not reject linearity, DML otherwise.  Returns (tau, track, p_RESET)."""
    y, t = data[:, v], data[:, u]
    W = data[:, list(W_idx)] if len(W_idx) else np.empty((len(y), 0))
    X_reg = np.column_stack([t, W]) if W.shape[1] else t.reshape(-1, 1)
    p_reset = reset_test_pvalue(y, X_reg)
    if p_reset > alpha_reset:
        try:
            return ols_ace(y, t, W), "OLS", p_reset
        except Exception:
            pass
    return dml_ace(y, t, W, n_fold=n_fold, seed=seed), "DML", p_reset


def estimate_edge_weights(data: np.ndarray, res: Resolution, f_weight: np.ndarray,
                          alpha_reset: float = 0.05, n_fold: int = 5, seed: int = 42):
    m = len(res.groups)
    A = np.zeros((m, m))
    records = []
    for (a, b), edges in sorted(res.member_edges.items()):
        W_idx = res.backdoor_members(a, exclude=b)
        total = 0.0
        for u, v in edges:
            tau, track, p = edge_ace(data, u, v, W_idx, alpha_reset, n_fold, seed)
            a_stab = f_weight[u, v] * abs(tau)
            total += a_stab
            records.append(dict(u=u, v=v, node_u=a, node_v=b, backdoor=tuple(W_idx),
                                tau_hat=tau, A_uv=abs(tau), f_uv=f_weight[u, v],
                                A_stab=a_stab, track=track, reset_p=p, symmetric=False))
        A[a, b] = total / (len(res.groups[a]) * len(res.groups[b]))
    for i, j in res.symmetric_pairs:
        a, b = int(res.group_of[i]), int(res.group_of[j])
        tau_ij, tr_ij, p_ij = edge_ace(data, i, j, res.backdoor_members(a, b),
                                       alpha_reset, n_fold, seed)
        tau_ji, tr_ji, p_ji = edge_ace(data, j, i, res.backdoor_members(b, a),
                                       alpha_reset, n_fold, seed)
        a_sym = f_weight[i, j] * 0.5 * (abs(tau_ij) + abs(tau_ji))
        scale = len(res.groups[a]) * len(res.groups[b])
        A[a, b] += a_sym / scale
        A[b, a] += a_sym / scale
        for (u, v, na, nb, tau, tr, p) in ((i, j, a, b, tau_ij, tr_ij, p_ij),
                                          (j, i, b, a, tau_ji, tr_ji, p_ji)):
            records.append(dict(u=u, v=v, node_u=na, node_v=nb, backdoor=(),
                                tau_hat=tau, A_uv=abs(tau), f_uv=f_weight[i, j],
                                A_stab=a_sym, track=tr, reset_p=p, symmetric=True))
    return A, records



def transition_matrix(A: np.ndarray, alpha_pr: float = 0.15) -> np.ndarray:
    """Eq. (4): P = (1 - alpha_PR) D^{-1} A + (alpha_PR / |V*|) 11^T, with
    dangling rows (zero out-degree) of D^{-1} A replaced by the uniform row."""
    m = A.shape[0]
    row = A.sum(axis=1)
    P0 = np.empty_like(A, dtype=float)
    dangling = row <= 0
    P0[~dangling] = A[~dangling] / row[~dangling, None]
    P0[dangling] = 1.0 / m
    return (1 - alpha_pr) * P0 + (alpha_pr / m) * np.ones((m, m))


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    vals, vecs = eig(P.T)
    pi = np.abs(np.real(vecs[:, np.argmin(np.abs(vals - 1.0))]))
    return pi / pi.sum()


def compute_laplacian(A: np.ndarray, alpha_pr: float = 0.15):
    P = transition_matrix(A, alpha_pr)
    pi = stationary_distribution(P)
    s = np.sqrt(pi)
    M = (s[:, None] * P) / s[None, :]          # Pi^{1/2} P Pi^{-1/2}
    L = np.eye(A.shape[0]) - 0.5 * (M + M.T)
    return 0.5 * (L + L.T), P, pi                # remove round-off asymmetry


def select_dimension(L: np.ndarray):
    """K_hat = argmax_{2 <= j <= |V*|-1} (lambda_{j+1} - lambda_j) and
    Phi_K = [phi_2, ..., phi_{K+1}] (the trivial gap lambda_2 - lambda_1 is
    excluded).  Returns (Phi_K, K_hat, eigvals, eigvecs)."""
    m = L.shape[0]
    if m < 2:
        raise ValueError("V* has fewer than two nodes; no spectral embedding exists.")
    eigvals, eigvecs = np.linalg.eigh(L)
    if m == 2:
        K = 1
    else:
        gaps = np.diff(eigvals)                 # gaps[j-1] = lambda_{j+1} - lambda_j
        js = np.arange(2, m)
        K = int(js[np.argmax(gaps[js - 1])])
    return eigvecs[:, 1:K + 1], K, eigvals, eigvecs


# ═════════════════════════════════════════════════════════════════════════════
#  STEP 4 ─ CAUSAL SPECTRAL EMBEDDING AND CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════

def build_X_std(data: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    """Column-standardized data over V*: each merged node N_S is the
    element-wise average of the standardized columns in S."""
    Z = StandardScaler().fit_transform(np.asarray(data, float))
    return np.column_stack([Z[:, list(g)].mean(axis=1) for g in groups])


def gap_statistic(X: np.ndarray, k: int, B_ref: int = 50, seed: int = 42):
    """Gap statistic of Tibshirani et al. with uniform reference data over the
    bounding box of X.  Returns (Gap(k), s_k)."""
    rng = np.random.default_rng(seed)
    wk = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).inertia_
    lo, hi = X.min(axis=0), X.max(axis=0)
    ref = [np.log(KMeans(n_clusters=k, n_init=5, random_state=seed)
                  .fit(rng.uniform(lo, hi, size=X.shape)).inertia_ + 1e-12)
           for _ in range(B_ref)]
    return float(np.mean(ref) - np.log(wk + 1e-12)), float(np.std(ref) * np.sqrt(1 + 1 / B_ref))


def relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Relabel clusters 0, 1, ... in order of decreasing size."""
    uniq, counts = np.unique(labels, return_counts=True)
    order = uniq[np.argsort(-counts, kind="stable")]
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[x] for x in labels])


def embed_and_cluster(X_std: np.ndarray, Phi: np.ndarray, k_max: int = 9,
                      B_ref: int = 50, seed: int = 42, gap: bool = True):
    """X_tilde = X_std Phi_K (Eq. 7); k-means for k in {2, ..., k_max};
    k* maximizes the Silhouette score.  The Gap-statistic choice (smallest k
    with Gap(k) >= Gap(k+1) - s_{k+1}) is reported as a diagnostic."""
    X_emb = X_std @ Phi
    n = X_emb.shape[0]
    ks = list(range(2, min(k_max, n - 1) + 1))
    sil, gaps, sds = {}, {}, {}
    for k in ks:
        labels = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(X_emb)
        sil[k] = silhouette_score(X_emb, labels) if len(set(labels)) > 1 else -1.0
        if gap:
            gaps[k], sds[k] = gap_statistic(X_emb, k, B_ref=B_ref, seed=seed)
    k_star = max(sil, key=sil.get)
    k_gap = None
    if gap:
        k_gap = ks[-1]
        for k, k_next in zip(ks[:-1], ks[1:]):
            if gaps[k] >= gaps[k_next] - sds[k_next]:
                k_gap = k
                break
    clusters = KMeans(n_clusters=k_star, n_init=50, random_state=seed).fit_predict(X_emb)
    diagnostics = dict(k_silhouette=k_star, k_gap=k_gap, sil_scores=sil,
                       gap_scores=gaps, gap_sds=sds,
                       criteria_agree=(k_gap == k_star) if gap else None)
    return relabel_by_size(clusters), X_emb, k_star, diagnostics



def _as_index(x, names):
    if x is None:
        return None
    return names.index(x) if isinstance(x, str) else int(x)


def caspect(data, var_names: Optional[Sequence[str]] = None,
            treat_idx=None, outcome_idx=None, *,
            alpha_ci: float = DEFAULTS["alpha_ci"], B: int = DEFAULTS["B"],
            theta: float = DEFAULTS["theta"], gamma: float = DEFAULTS["gamma"],
            w_max: float = DEFAULTS["w_max"], alpha_jb: float = DEFAULTS["alpha_jb"],
            alpha_reset: float = DEFAULTS["alpha_reset"],
            alpha_pr: float = DEFAULTS["alpha_pr"], n_fold: int = DEFAULTS["n_fold"],
            k_max: int = DEFAULTS["k_max"], domain_order=None, B_ref: int = 50,
            seed: int = 42, pc_only: bool = False, no_stability: bool = False,
            symmetrise: bool = False, gap_statistic_diag: bool = True,
            visualise: bool = False, save_dir: Optional[str] = None,
            verbose: bool = True) -> dict:
    def log(msg):
        if verbose:
            print(f"[CaSPECT] {msg}")

    X = np.asarray(data, dtype=float)
    n, d = X.shape
    names = list(var_names) if var_names is not None else [f"X{i}" for i in range(d)]
    z_idx, y_idx = _as_index(treat_idx, names), _as_index(outcome_idx, names)
    order_idx = [_as_index(v, names) for v in domain_order] if domain_order else None
    Xs = StandardScaler().fit_transform(X)
    if not 0 < w_max < 0.5:
        warnings.warn("The paper requires w_max in (0, 0.5).")

    # ── Step 1: DAG formation ────────────────────────────────────────────────
    log(f"Step 1a: PC algorithm (alpha_n = {alpha_ci}) ...")
    G_pc = run_pc(Xs, alpha=alpha_ci)

    log(f"Step 1b: bootstrap stability ({B} resamples) ...")
    f_uv, f_dir, rho = bootstrap_pc(Xs, B=B, alpha=alpha_ci, seed=seed)
    f_eff = (f_uv > 0).astype(float) if no_stability else f_uv
    n_skel = int(np.sum(np.triu(f_eff >= theta, 1) & np.triu(f_eff > 0, 1)))
    log(f"  stable skeleton (f_uv >= {theta}): {n_skel} edges")

    log("Step 1c: Jarque-Bera, DirectLiNGAM and OVS ...")
    n_ng = jarque_bera_count(Xs, alpha_jb=alpha_jb)
    w_L = lingam_weight(n_ng, d, w_max)
    log(f"  n_NG = {n_ng}/{d}  ->  w_L = {w_L:.3f},  w_PC = {1 - w_L:.3f}")
    if w_L <= gamma:
        log(f"  note: w_L <= gamma = {gamma}; Proposition 1(ii) (reversible edges) does not apply")
    if w_L >= (1 - gamma) / 2:
        log("  note: w_L >= (1 - gamma)/2; Proposition 1(i) robustness does not apply")
    use_lingam = (not pc_only) and w_L > 0
    B_L = run_lingam(Xs, seed=seed) if use_lingam else np.zeros((d, d))
    orientations, ovs_matrix = compute_ovs(f_eff, rho, B_L, w_L, gamma=gamma,
                                           theta=theta, use_lingam=use_lingam)
    n_amb = sum(v is None for v in orientations.values())
    log(f"  oriented by OVS: {len(orientations) - n_amb},  |OVS| <= gamma: {n_amb}")

    log("Step 1d: resolution hierarchy ...")
    res = resolve_orientations(orientations, d, ovs_matrix, domain_order=order_idx,
                               ambiguous="symmetrise" if symmetrise else "contract")
    node_names = res.node_names(names)
    counts: dict = {}
    for s_ in res.step.values():
        counts[s_] = counts.get(s_, 0) + 1
    log(f"  resolution steps used: {counts}")
    log(f"  G_hat: |V*| = {len(res.groups)} nodes, {res.G_star.number_of_edges()} edges")
    for i, j in res.contracted_pairs:
        log(f"  contracted {names[i]} <-> {names[j]}  (OVS = {ovs_matrix[i, j]:+.3f})")

    # ── Step 2: edge weights ─────────────────────────────────────────────────
    log("Step 2: causal edge weights (RESET -> OLS / DML) ...")
    A_stab, records = estimate_edge_weights(Xs, res, f_eff, alpha_reset=alpha_reset,
                                            n_fold=n_fold, seed=seed)
    n_dml = sum(r["track"] == "DML" for r in records)
    log(f"  {len(records)} member edges: {len(records) - n_dml} OLS, {n_dml} DML")
    if A_stab.sum() == 0:
        warnings.warn("A_stab is all zero: no causal structure was recovered.")

    X_std = build_X_std(X, res.groups)
    degenerate = len(res.groups) < 2
    if degenerate:
        # Contraction merged every variable into one node: no Laplacian
        # spectrum to embed with, so all units form a single cluster.
        warnings.warn("|V*| = 1 after contraction; no spectral embedding exists and "
                      "all units are assigned to one cluster.")
        L = P = pi = eigvals = eigvecs = Phi = None
        K_hat, k_star = 0, 1
        clusters, X_emb = np.zeros(n, dtype=int), np.empty((n, 0))
        diag = dict(k_silhouette=None, k_gap=None, sil_scores={}, gap_scores={},
                    gap_sds={}, criteria_agree=None)
    else:
        # ── Step 3: Laplacian ────────────────────────────────────────────────
        log("Step 3: Chung directed Laplacian ...")
        L, P, pi = compute_laplacian(A_stab, alpha_pr=alpha_pr)
        Phi, K_hat, eigvals, eigvecs = select_dimension(L)
        gap_val = eigvals[K_hat] - eigvals[K_hat - 1] if K_hat < len(eigvals) else np.nan
        log(f"  K_hat = {K_hat}  (gap lambda_{K_hat + 1} - lambda_{K_hat} = {gap_val:.4f})")

        # ── Step 4: embedding and clustering ─────────────────────────────────
        log("Step 4: embedding and k-means ...")
        clusters, X_emb, k_star, diag = embed_and_cluster(X_std, Phi, k_max=k_max,
                                                          B_ref=B_ref, seed=seed,
                                                          gap=gap_statistic_diag)
        log(f"  k* = {k_star} (Silhouette); Gap-statistic k = {diag['k_gap']}")

    # ── Treatment / outcome diagnostics ──────────────────────────────────────
    zy = None
    if z_idx is not None and y_idx is not None:
        gz, gy = int(res.group_of[z_idx]), int(res.group_of[y_idx])
        merged = gz == gy
        path = (not merged) and nx.has_path(res.G_star, gz, gy)
        zy = dict(node_Z=gz, node_Y=gy, merged=merged, directed_path=path,
                  A_stab_ZY=float(A_stab[gz, gy]) if not merged else np.nan,
                  backdoor_Z=res.backdoor_members(gz, exclude=gy))
        if merged:
            log(f"  WARNING: treatment and outcome were merged into '{node_names[gz]}'")
        elif not path:
            log("  WARNING: no directed path treatment ~> outcome in G_hat; the "
                "treatment-outcome pathway is absent from the graph")

    log("Done.")
    result = dict(
        clusters=clusters, embedding=X_emb, X_std=X_std, adjacency=A_stab,
        laplacian=L, laplacian_P=P, pi=pi, eigenvalues=eigvals, eigenvectors=eigvecs,
        Phi=Phi, K_hat=K_hat, K_star=K_hat, k_star=k_star, diagnostics=diag,
        resolution=res, groups=res.groups, node_names=node_names,
        edges=res.final_direction, contracted_pairs=res.contracted_pairs,
        edge_records=records, ovs_matrix=ovs_matrix, orientations=orientations,
        f_uv=f_uv, f_dir=f_dir, g_uv=f_dir, rho=rho, B_L=B_L, n_NG=n_ng, w_L=w_L,
        G_pc=G_pc, treatment_outcome=zy, var_names=names,
    )
    if visualise and not degenerate:
        plot_spectral_diagnostics(eigvals, diag["sil_scores"], K_hat, k_star, save_dir=save_dir)
        plot_dag(A_stab, node_names, res.merged_nodes(), save_dir=save_dir)
        plot_tsne(Xs, clusters, k_star, K_hat, seed=seed, save_dir=save_dir)
    return result


def CSC_PC_PIPELINE(data, var_names=None, tau=None, w_L_max=None, dml_k=None,
                    visualise=True, **kwargs):
    """Backward-compatible wrapper around :func:`caspect` that accepts the old
    argument names (tau -> gamma, w_L_max -> w_max, dml_k -> n_fold)."""
    if tau is not None:
        kwargs.setdefault("gamma", tau)
    if w_L_max is not None:
        kwargs.setdefault("w_max", w_L_max)
    if dml_k is not None:
        kwargs.setdefault("n_fold", dml_k)
    return caspect(data, var_names=var_names, visualise=visualise, **kwargs)


def edge_summary(result: dict, stability_bins=(0.70, 0.90)):

    import pandas as pd
    names, node_names = result["var_names"], result["node_names"]
    A, recs = result["adjacency"], result["edge_records"]
    lo, hi = stability_bins
    rows = []
    by_node: dict = {}
    for r in recs:
        by_node.setdefault((r["node_u"], r["node_v"]), []).append(r)
    for (a, b), rs in by_node.items():
        f = float(np.mean([r["f_uv"] for r in rs]))
        tracks = sorted({r["track"] for r in rs})
        sym = any(r["symmetric"] for r in rs)
        rows.append(dict(edge=f"{node_names[a]} {'<->' if sym else '->'} {node_names[b]}",
                         A_stab=float(A[a, b]), f_uv=f,
                         stability="High" if f >= hi else "Moderate" if f >= lo else "Low",
                         track="+".join(tracks), n_member_edges=len(rs)))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("A_stab", ascending=False).drop_duplicates("edge")
    ovs = result["ovs_matrix"]
    for i, j in result["contracted_pairs"]:
        df = pd.concat([df, pd.DataFrame([dict(
            edge=f"{names[i]} <-> {names[j]}", A_stab=np.nan, f_uv=result["f_uv"][i, j],
            stability=f"Contracted, OVS = {ovs[i, j]:+.2f}", track="-", n_member_edges=0)])],
            ignore_index=True)
    return df.reset_index(drop=True)


def _finish(fig, save_dir, fname):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[CaSPECT] figure saved -> {path}")
    else:
        plt.show()


def plot_spectral_diagnostics(eigvals, sil_scores, K_hat, k_star, title=None, save_dir=None):
    """Eigenvalue spectrum, spectral gaps (largest non-trivial gap marked) and
    Silhouette score against k (Figures 2, 5 and 8)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(eigvals, marker="o", lw=1.5)
    ax[0].set(title="Eigenvalue Spectrum", xlabel="Index", ylabel="Eigenvalue")
    gaps = np.diff(eigvals)
    ax[1].plot(gaps, marker="o", color="tomato", lw=1.5)
    ax[1].axvline(K_hat - 1, color="navy", ls="--", label=f"K* = {K_hat}  (non-trivial)")
    ax[1].set(title="Spectral Gaps", xlabel="Index", ylabel="Gap")
    ax[1].legend()
    ks = sorted(sil_scores)
    ax[2].plot(ks, [sil_scores[k] for k in ks], marker="o", lw=1.5)
    ax[2].axvline(k_star, color="gray", ls=":", lw=1)
    ax[2].set(title="Silhouette Score", xlabel="k", ylabel="Silhouette score")
    from matplotlib.ticker import MaxNLocator
    for a in ax:
        a.grid(True)
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    if title:
        fig.suptitle(title)
    _finish(fig, save_dir, "spectral_diagnostics.png")


def plot_dag(A, node_names, merged_nodes=(), title=None, save_dir=None):
    """Estimated causal DAG over V* with A_stab edge labels (Figures 1, 4, 7).
    Contracted nodes and their edges are drawn in tomato."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    m = A.shape[0]
    merged = set(merged_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(range(m))
    for i in range(m):
        for j in range(m):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j])
    pos = nx.spring_layout(G, seed=42)
    special = [(u, v) for u, v in G.edges() if u in merged or v in merged]
    normal = [(u, v) for u, v in G.edges() if (u, v) not in special]
    fig, ax = plt.subplots(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900,
                           node_color=["tomato" if i in merged else "steelblue" for i in range(m)])
    def _wrap(name, per_line=2):
        parts = name.split("/")
        return "/\n".join("/".join(parts[i:i + per_line]) for i in range(0, len(parts), per_line))
    nx.draw_networkx_labels(G, pos, ax=ax, labels={i: _wrap(nm) for i, nm in enumerate(node_names)},
                            font_size=8, bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                                   ec="none", alpha=0.75))
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=normal, edge_color="gray",
                           arrows=True, arrowsize=20, width=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=special, edge_color="tomato",
                           style="dashed", arrows=True, arrowsize=20, width=1.5)
    nx.draw_networkx_edge_labels(G, pos, ax=ax, font_size=7,
                                 edge_labels={(u, v): f"{d['weight']:.2f}"
                                              for u, v, d in G.edges(data=True)})
    ax.legend(handles=[mpatches.Patch(color="steelblue", label="Normal node"),
                       mpatches.Patch(color="tomato", label="Contracted node (merged)"),
                       Line2D([0], [0], color="gray", label="Normal edge"),
                       Line2D([0], [0], color="tomato", ls="--", label="Contracted-node edge")],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)
    ax.set_title(title or "Estimated causal DAG (edge labels = bootstrap-stability-weighted |ACE|)")
    ax.axis("off")
    _finish(fig, save_dir, "dag.png")


def plot_tsne(X, clusters, k_star=None, K_hat=None, seed=42, title=None, save_dir=None):
    """t-SNE map of the standardized data coloured by cluster (Figures 3, 6, 9)."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    X2 = TSNE(n_components=2, perplexity=min(30, len(X) - 1),
              random_state=seed).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X2[:, 0], X2[:, 1], c=clusters, cmap="tab10", alpha=0.75, s=12,
               vmin=0, vmax=9)
    ax.set(xlabel="dim 1", ylabel="dim 2",
           title=title or f"t-SNE of CaSPECT clusters (k* = {k_star}, K = {K_hat})")
    _finish(fig, save_dir, "tsne_clusters.png")
