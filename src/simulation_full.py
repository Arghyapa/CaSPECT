"""
CaSPECT simulation study (Section 7) and baseline comparison (Section 8).

Design (Section 7.1)
--------------------
* q = 8 variables V = {X1, ..., X6, Z, Y}; Z and Y are columns 6 and 7.
* Random topological order; each forward pair is an edge with probability
  rho = 0.35; beta_uv ~ s_uv * Unif(0.3, 0.8) with a Rademacher sign.
* Three planted clusters of approximately n/3 units each (multinomial sizes);
  the cluster enters only the treatment equation through the intercept
  eta = (-1, 0, +1), so tau_1 = tau_2 = tau_3 = tau, the total effect of Z on Y
  (Eq. 8).  This is a propensity-shift benchmark.
* S1  linear SEM, standardized t5 errors, n in {500, 1000, 2000}.
* S2  as S1, but the three edges with the largest |beta| enter as
      beta_uv * sin(pi * u).
* S3  as S1, plus a latent H ~ N(0, 1) with coefficients zeta = 0.4 on two
      random variables a, b in V \\ {Z, Y}.
* Ablations (S1, n = 1000): A1 PC-only orientation (e_L = 0), A2 no stability
  weighting (f_uv = 1), A3 symmetrization instead of contraction.
* Baselines (S1, n = 1000): see ``baselines.py``.

Metrics
-------
ARI against the planted labels; orientation accuracy Acc_or = share of edges
of the stable skeleton whose final direction (OVS or resolution hierarchy)
matches G0; cluster-level Bias_c and RMSE_c of the DML estimate restricted to
the estimated cluster that the Hungarian algorithm matches to planted
cluster c.

Usage
-----
    python simulation_full.py                       # everything (slow, Table 2 settings)
    python simulation_full.py --part S1 --n 1000
    python simulation_full.py --part ablation --n-jobs 8
    python simulation_full.py --part baselines
    python simulation_full.py --quick               # smoke test (tiny M and B)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caspect import caspect, dml_ace, DEFAULTS          # noqa: E402

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover
    tabulate = None

# Table 2 plus the DGP constants of Section 7.1
CONFIG = dict(DEFAULTS, M=100, q=8, treat_idx=6, outcome_idx=7, p_edge=0.35,
              eta=(-1.0, 0.0, 1.0), zeta=0.4, n_nonlinear=3, min_cluster=20)

N_LIST = {"S1": [500, 1000, 2000], "S2": [500, 1000, 2000], "S3": [500, 1000, 2000]}
TITLES = {"S1": "Table 3: Simulation results for linear DAG (S1)",
          "S2": "Table 4: Simulation results for mixed linearity (S2)",
          "S3": "Table 5: Simulation results with latent confounder (S3)"}
ABLATIONS = {
    "CaSPECT":             dict(),
    "A1: PC-only orient":  dict(pc_only=True),
    "A2: No stab weight":  dict(no_stability=True),
    "A3: Symmetrization":  dict(symmetrise=True),
}


# ═════════════════════════════════════════════════════════════════════════════
#  DATA GENERATING PROCESS (Section 7.1)
# ═════════════════════════════════════════════════════════════════════════════

def sample_dag(q: int, p_edge: float = 0.35, seed: int = 0) -> np.ndarray:
    """Random topological order, then u -> v with probability p_edge for every
    pair with u preceding v."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(q)
    A = np.zeros((q, q), dtype=int)
    for i in range(q):
        for j in range(i + 1, q):
            if rng.random() < p_edge:
                A[perm[i], perm[j]] = 1
    return A


def sample_coefficients(A: np.ndarray, seed: int = 0) -> np.ndarray:
    """beta_uv = s_uv * Unif(0.3, 0.8), s_uv uniform on {-1, +1}."""
    rng = np.random.default_rng(seed)
    B = np.zeros(A.shape)
    mask = A > 0
    k = int(mask.sum())
    B[mask] = rng.uniform(0.3, 0.8, size=k) * rng.choice([-1, 1], size=k)
    return B


def total_effect(B: np.ndarray, u: int, v: int) -> float:
    """Sum over directed paths u ~> v of the products of beta (Eq. 8)."""
    return float(np.linalg.inv(np.eye(B.shape[0]) - B)[u, v])


def generate_data(n: int, A: np.ndarray, B: np.ndarray, treat_idx: int,
                  error_dist: str = "t5", nonlinear_edges=None,
                  latent_targets=None, zeta: float = 0.4,
                  eta=(-1.0, 0.0, 1.0), seed: int = 0):
    """Recursive SEM v = sum_u beta_uv u + eps_v with the planted intercept
    eta_c added to the equation of Z only.  Returns (X, labels)."""
    rng = np.random.default_rng(seed)
    q = A.shape[0]
    order = list(nx.topological_sort(nx.from_numpy_array(A, create_using=nx.DiGraph)))
    sizes = rng.multinomial(n, [1 / 3] * 3)
    labels = rng.permutation(np.repeat([0, 1, 2], sizes))
    if error_dist == "t5":
        E = rng.standard_t(5, size=(n, q)) / np.sqrt(5 / 3)     # unit variance
    else:
        E = rng.standard_normal(size=(n, q))
    H = rng.standard_normal(n) if latent_targets else None
    nl = set(map(tuple, nonlinear_edges or []))
    X = np.zeros((n, q))
    for v in order:
        xv = E[:, v].copy()
        for u in np.flatnonzero(A[:, v]):
            xv += B[u, v] * (np.sin(np.pi * X[:, u]) if (u, v) in nl else X[:, u])
        if latent_targets and v in latent_targets:
            xv += zeta * H
        if v == treat_idx:
            xv += np.asarray(eta)[labels]
        X[:, v] = xv
    return X, labels


def make_replication(setting: str, m: int, cfg: dict, base_seed: int = 0):
    """True DAG, coefficients and setting-specific arguments for replication m."""
    q, z, y = cfg["q"], cfg["treat_idx"], cfg["outcome_idx"]
    seed_dag = base_seed + 1000 * m
    A = sample_dag(q, cfg["p_edge"], seed=seed_dag)
    B = sample_coefficients(A, seed=seed_dag + 1)
    nl = None
    if setting == "S2":
        edges = sorted(((u, v) for u in range(q) for v in range(q) if A[u, v]),
                       key=lambda e: -abs(B[e]))
        nl = edges[:cfg["n_nonlinear"]]
    lat = None
    if setting == "S3":
        others = [i for i in range(q) if i not in (z, y)]
        lat = tuple(int(i) for i in np.random.default_rng(seed_dag + 7)
                    .choice(others, size=2, replace=False))
    return dict(A=A, B=B, nonlinear_edges=nl, latent_targets=lat,
                tau=total_effect(B, z, y), seed=seed_dag + 2)


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════════════════

def orientation_accuracy(res, A_true: np.ndarray) -> float:
    """Share of stable-skeleton edges whose final direction matches G0.
    Spurious, contracted and symmetrised edges count as not correct."""
    if not res.skeleton:
        return np.nan
    correct = sum(1 for e in res.skeleton
                  if res.final_direction[e] is not None and A_true[res.final_direction[e]] == 1)
    return correct / len(res.skeleton)


def cluster_aces(X: np.ndarray, clusters: np.ndarray, z: int, y: int, W_idx,
                 n_fold: int = 5, seed: int = 42, min_size: int = 20) -> dict:
    """DML estimate of the Z -> Y effect restricted to each estimated cluster,
    on the raw (unstandardized) scale of the DGP, adjusting for the backdoor
    set W read off the estimated DAG."""
    out = {}
    for c in np.unique(clusters):
        mask = clusters == c
        if mask.sum() < min_size or np.std(X[mask, z]) < 1e-8:
            out[c] = np.nan
            continue
        W = X[np.ix_(mask, list(W_idx))] if len(W_idx) else np.empty((mask.sum(), 0))
        out[c] = dml_ace(X[mask, y], X[mask, z], W, n_fold=n_fold, seed=seed)
    return out


def hungarian_match(true_labels: np.ndarray, est_labels: np.ndarray, k_true: int = 3):
    """Planted cluster c -> estimated label (or None if unmatched)."""
    est = np.unique(est_labels)
    C = np.array([[np.sum((true_labels == c) & (est_labels == e)) for e in est]
                  for c in range(k_true)])
    rows, cols = linear_sum_assignment(-C)
    match = {c: None for c in range(k_true)}
    for r, cidx in zip(rows, cols):
        match[int(r)] = est[cidx]
    return match


def matched_errors(X, est_labels, true_labels, tau, z, y, W_idx, cfg, seed):
    """tau_hat_c - tau for c = 1, 2, 3 (NaN when cluster c is unmatched or too
    small)."""
    aces = cluster_aces(X, est_labels, z, y, W_idx, cfg["n_fold"], seed, cfg["min_cluster"])
    match = hungarian_match(true_labels, est_labels)
    return np.array([aces[match[c]] - tau if match[c] is not None else np.nan
                     for c in range(3)])


# ═════════════════════════════════════════════════════════════════════════════
#  ONE REPLICATION
# ═════════════════════════════════════════════════════════════════════════════

def one_replication(setting: str, n: int, m: int, cfg: dict, ablation: Optional[dict] = None,
                    with_baselines: bool = False, base_seed: int = 0) -> dict:
    ablation = ablation or {}
    rep = make_replication(setting, m, cfg, base_seed)
    z, y, seed = cfg["treat_idx"], cfg["outcome_idx"], rep["seed"]
    X, labels = generate_data(n, rep["A"], rep["B"], z,
                              nonlinear_edges=rep["nonlinear_edges"],
                              latent_targets=rep["latent_targets"],
                              zeta=cfg["zeta"], eta=cfg["eta"], seed=seed)
    fit = caspect(X, treat_idx=z, outcome_idx=y, alpha_ci=cfg["alpha_ci"], B=cfg["B"],
                  theta=cfg["theta"], gamma=cfg["gamma"], w_max=cfg["w_max"],
                  alpha_jb=cfg["alpha_jb"], alpha_reset=cfg["alpha_reset"],
                  alpha_pr=cfg["alpha_pr"], n_fold=cfg["n_fold"], k_max=cfg["k_max"],
                  seed=seed, gap_statistic_diag=False, verbose=False, **ablation)
    W_idx = fit["treatment_outcome"]["backdoor_Z"]
    out = dict(setting=setting, n=n, m=m, tau=rep["tau"], k_star=fit["k_star"],
               K_hat=fit["K_hat"], n_contracted=len(fit["contracted_pairs"]),
               ari=adjusted_rand_score(labels, fit["clusters"]),
               acc_or=orientation_accuracy(fit["resolution"], rep["A"]))
    err = matched_errors(X, fit["clusters"], labels, rep["tau"], z, y, W_idx, cfg, seed)
    out.update({f"err{c + 1}": err[c] for c in range(3)})

    if rep["nonlinear_edges"]:                       # RESET routing (S2)
        nl = set(map(tuple, rep["nonlinear_edges"]))
        dml = {(r["u"], r["v"]) for r in fit["edge_records"] if r["track"] == "DML"}
        est = {(r["u"], r["v"]) for r in fit["edge_records"]}
        lin = est - nl
        out["dml_rate_nonlinear"] = len(dml & nl) / len(nl)
        out["dml_rate_linear"] = len(dml & lin) / len(lin) if lin else np.nan

    if with_baselines:
        from baselines import run_all_baselines
        X_std = StandardScaler().fit_transform(X)
        cov = [i for i in range(cfg["q"]) if i not in (z, y)]
        for name, (lab, k) in run_all_baselines(X_std, X_std[:, cov], X[:, z], X[:, y],
                                                k_max=cfg["k_max"], seed=seed).items():
            e = matched_errors(X, lab, labels, rep["tau"], z, y, W_idx, cfg, seed)
            out[f"{name}|ari"] = adjusted_rand_score(labels, lab)
            out[f"{name}|k"] = k
            out.update({f"{name}|err{c + 1}": e[c] for c in range(3)})
    return out


def run_replications(setting, n, cfg, ablation=None, with_baselines=False,
                     n_jobs=1, base_seed=0, verbose=True) -> pd.DataFrame:
    def task(m):
        try:
            return one_replication(setting, n, m, cfg, ablation, with_baselines, base_seed)
        except Exception as exc:              # keep the run going; report failures
            return dict(setting=setting, n=n, m=m, error=repr(exc))
    if n_jobs == 1:
        rows = [task(m) for m in range(cfg["M"])]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(delayed(task)(m) for m in range(cfg["M"]))
    df = pd.DataFrame(rows)
    if verbose and "error" in df and df["error"].notna().any():
        print(f"  {df['error'].notna().sum()} replication(s) failed, e.g. "
              f"{df['error'].dropna().iloc[0]}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  SUMMARIES AND TABLES
# ═════════════════════════════════════════════════════════════════════════════

def summarise(df: pd.DataFrame, prefix: str = "") -> dict:
    """Replication means; RMSE_c = sqrt(mean((tau_hat_c - tau)^2)) and
    Bias_c = mean(tau_hat_c - tau) over the replications in which cluster c
    was matched."""
    ari = df[f"{prefix}ari"].dropna()
    row = dict(M=len(ari), ARI=ari.mean(), ARI_sd=ari.std(ddof=1))
    if f"{prefix}acc_or" in df:
        acc = df[f"{prefix}acc_or"].dropna()
        row.update(Acc_or=acc.mean(), Acc_or_sd=acc.std(ddof=1))
    for c in (1, 2, 3):
        e = df[f"{prefix}err{c}"].dropna()
        row[f"RMSE{c}"] = float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan
        row[f"Bias{c}"] = float(e.mean()) if len(e) else np.nan
        row[f"M_matched{c}"] = len(e)
    row["Mean_RMSE"] = np.nanmean([row["RMSE1"], row["RMSE2"], row["RMSE3"]])
    if f"{prefix}k" in df:
        row["share_k3"] = float((df[f"{prefix}k"] == 3).mean())
    elif f"{prefix}k_star" in df:
        row["share_k3"] = float((df[f"{prefix}k_star"] == 3).mean())
    for col in ("dml_rate_nonlinear", "dml_rate_linear"):
        if col in df and not prefix:
            row[col] = df[col].mean()
    return row


def _print(rows, headers, title):
    print(f"\n{'─' * 78}\n  {title}\n{'─' * 78}")
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".3f"))
    else:
        print("\t".join(headers))
        for r in rows:
            print("\t".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r))


def print_setting_table(summary: pd.DataFrame, setting: str):
    sub = summary[summary.setting == setting]
    rows = [[int(r.n), r.ARI, r.Acc_or, r.RMSE1, r.RMSE2, r.RMSE3] for r in sub.itertuples()]
    _print(rows, ["n", "ARI", "Acc_or", "RMSE1", "RMSE2", "RMSE3"], TITLES[setting])
    if setting == "S2" and "dml_rate_nonlinear" in sub:
        print("  RESET routing to DML: " + ", ".join(
            f"n={int(r.n)}: nonlinear {r.dml_rate_nonlinear:.2f} / linear {r.dml_rate_linear:.2f}"
            for r in sub.itertuples()))


def print_ablation_table(summary: pd.DataFrame, n: int):
    rows = [[r.method, f"{r.ARI:.3f} ({r.ARI_sd:.3f})", f"{r.Acc_or:.3f} ({r.Acc_or_sd:.3f})",
             r.Mean_RMSE] for r in summary.itertuples()]
    _print(rows, ["Method", "ARI", "Acc_or", "Mean RMSE"],
           f"Table 6: Ablation study under the first scenario (n = {n})")


def baseline_summary(df: pd.DataFrame) -> pd.DataFrame:
    names = sorted({c.split("|")[0] for c in df.columns if "|" in c},
                   key=lambda s: list(BASELINE_ORDER).index(s) if s in BASELINE_ORDER else 99)
    rows = [dict(method=nm, **summarise(df, prefix=f"{nm}|")) for nm in names]
    rows.append(dict(method="CaSPECT (Ours)", **summarise(df)))
    out = pd.DataFrame(rows)
    ref = out[out.method == "CaSPECT (Ours)"].iloc[0]
    se_ref = ref.ARI_sd / np.sqrt(ref.M)
    out["ARI_se"] = out.ARI_sd / np.sqrt(out.M)
    out["ARI_ci_low"] = out.ARI - 1.96 * out.ARI_se
    out["ARI_ci_high"] = out.ARI + 1.96 * out.ARI_se
    out["delta_vs_CaSPECT"] = out.ARI - ref.ARI
    se_delta = np.sqrt(out.ARI_se ** 2 + se_ref ** 2)
    out["z_vs_CaSPECT"] = np.where(out.method == "CaSPECT (Ours)", np.nan,
                                   out.delta_vs_CaSPECT / se_delta)
    return out


BASELINE_ORDER = ("k-means on X", "Spectral clustering on X", "Causal k-means",
                  "Hierarchical causal clustering", "R-learner CACE + k-means",
                  "Causal forest + k-means")


def print_baseline_table(summary: pd.DataFrame, n: int, M: int):
    rows = [[r.method, f"{r.ARI:.3f} ({r.ARI_sd:.3f})", r.Mean_RMSE,
             "" if np.isnan(r.z_vs_CaSPECT) else f"{r.z_vs_CaSPECT:+.2f}"]
            for r in summary.itertuples()]
    _print(rows, ["Method", "ARI", "Mean RMSE", "z vs CaSPECT"],
           f"Table 7: Baseline comparison, n = {n}, M = {M} replications")
    ref = summary[summary.method == "CaSPECT (Ours)"].iloc[0]
    print(f"  CaSPECT ARI {ref.ARI:.3f}, SE {ref.ARI_se:.4f}, "
          f"95% interval [{ref.ARI_ci_low:.3f}, {ref.ARI_ci_high:.3f}]")


def plot_results(summary: pd.DataFrame, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for setting, color in (("S1", "steelblue"), ("S2", "tomato"), ("S3", "seagreen")):
        sub = summary[summary.setting == setting]
        if sub.empty:
            continue
        ax[0].plot(sub.n, sub.ARI, marker="o", color=color, label=setting)
        ax[1].plot(sub.n, sub.Mean_RMSE, marker="^", color=color, label=setting)
    for a, yl, t in zip(ax, ["ARI", "Mean ACE RMSE"],
                        ["ARI vs propensity-shift labels", "Cluster-level ACE error"]):
        a.set(xlabel="Sample size n", ylabel=yl, title=t)
        a.grid(True, alpha=0.4)
        a.legend(fontsize=9)
    fig.suptitle("CaSPECT simulation study", fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "simulation_results.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  figure saved -> {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="all",
                    choices=["all", "S1", "S2", "S3", "ablation", "baselines"])
    ap.add_argument("--n", type=int, nargs="*", help="sample sizes (default: Section 7)")
    ap.add_argument("--M", type=int, default=CONFIG["M"], help="Monte Carlo replications")
    ap.add_argument("--B", type=int, default=CONFIG["B"], help="bootstrap resamples")
    ap.add_argument("--n-jobs", type=int, default=1, help="parallel replications (joblib)")
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--out-dir", default="simulation_output")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: M=2, B=10, n=500 (numbers are meaningless)")
    args = ap.parse_args(argv)

    cfg = dict(CONFIG, M=args.M, B=args.B)
    if args.quick:
        cfg.update(M=2, B=10)
    os.makedirs(args.out_dir, exist_ok=True)
    warnings.filterwarnings("ignore")
    parts = ["S1", "S2", "S3", "ablation", "baselines"] if args.part == "all" else [args.part]
    print("=" * 78 + f"\n  CaSPECT simulation study   M = {cfg['M']}, B = {cfg['B']}\n" + "=" * 78)

    setting_rows, rep_frames = [], []
    for part in [p for p in parts if p in N_LIST]:
        ns = args.n or ([500] if args.quick else N_LIST[part])
        for n in ns:
            t0 = time.time()
            df = run_replications(part, n, cfg, n_jobs=args.n_jobs, base_seed=args.seed)
            rep_frames.append(df)
            setting_rows.append(dict(setting=part, n=n, **summarise(df)))
            r = setting_rows[-1]
            print(f"  {part} n={n:5d}  ARI={r['ARI']:.3f}  Acc_or={r['Acc_or']:.3f}  "
                  f"RMSE=({r['RMSE1']:.3f}, {r['RMSE2']:.3f}, {r['RMSE3']:.3f})  "
                  f"[{time.time() - t0:.0f}s]")
    if setting_rows:
        summary = pd.DataFrame(setting_rows)
        for s in summary.setting.unique():
            print_setting_table(summary, s)
        summary.to_csv(os.path.join(args.out_dir, "simulation_summary.csv"), index=False)
        pd.concat(rep_frames).to_csv(os.path.join(args.out_dir, "simulation_replications.csv"),
                                     index=False)
        plot_results(summary, args.out_dir)

    n_ab = (args.n or [1000])[0] if not args.quick else 500
    if "ablation" in parts:
        rows, frames = [], []
        for label, abl in ABLATIONS.items():
            print(f"  ablation: {label} ...")
            df = run_replications("S1", n_ab, cfg, ablation=abl, n_jobs=args.n_jobs,
                                  base_seed=args.seed)
            df["method"] = label
            frames.append(df)
            rows.append(dict(method=label, **summarise(df)))
        ab = pd.DataFrame(rows)
        print_ablation_table(ab, n_ab)
        ab.to_csv(os.path.join(args.out_dir, "ablation_summary.csv"), index=False)
        pd.concat(frames).to_csv(os.path.join(args.out_dir, "ablation_replications.csv"), index=False)

    if "baselines" in parts:
        print("  baselines ...")
        df = run_replications("S1", n_ab, cfg, with_baselines=True, n_jobs=args.n_jobs,
                              base_seed=args.seed)
        bs = baseline_summary(df)
        print_baseline_table(bs, n_ab, cfg["M"])
        bs.to_csv(os.path.join(args.out_dir, "baseline_summary.csv"), index=False)
        df.to_csv(os.path.join(args.out_dir, "baseline_replications.csv"), index=False)

    print("\n" + "=" * 78 + "\n  Simulation complete.\n" + "=" * 78)


if __name__ == "__main__":
    main()
