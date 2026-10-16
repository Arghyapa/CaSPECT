import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from caspect import DEFAULTS, edge_summary

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")


def common_parser(description: str, default_fig_dir: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--B", type=int, default=DEFAULTS["B"], help="bootstrap resamples")
    ap.add_argument("--theta", type=float, default=DEFAULTS["theta"])
    ap.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    ap.add_argument("--w-max", type=float, default=DEFAULTS["w_max"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-gap", action="store_true", help="skip the Gap statistic")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--fig-dir", default=default_fig_dir,
                    help="directory for figures ('' shows them interactively)")
    return ap


def pipeline_kwargs(args) -> dict:
    return dict(B=args.B, theta=args.theta, gamma=args.gamma, w_max=args.w_max,
                seed=args.seed, gap_statistic_diag=not args.no_gap,
                visualise=not args.no_plots, save_dir=args.fig_dir or None)


def header(title: str):
    print("\n" + "=" * 72 + f"\n  {title}\n" + "=" * 72)


def section(title: str):
    print(f"\n── {title} " + "─" * max(0, 66 - len(title)))


def welch(y1: np.ndarray, y0: np.ndarray) -> dict:
    """Difference in means with Welch's t-test and 95% interval."""
    y1, y0 = np.asarray(y1, float), np.asarray(y0, float)
    v1, v0 = y1.var(ddof=1) / len(y1), y0.var(ddof=1) / len(y0)
    est, se = y1.mean() - y0.mean(), np.sqrt(v1 + v0)
    dfw = (v1 + v0) ** 2 / (v1 ** 2 / (len(y1) - 1) + v0 ** 2 / (len(y0) - 1))
    t = est / se
    tc = stats.t.ppf(0.975, dfw)
    return dict(ACE=est, t=t, p=2 * stats.t.sf(abs(t), dfw),
                ci_low=est - tc * se, ci_high=est + tc * se)


def ace_table(treat: np.ndarray, y: np.ndarray, clusters: np.ndarray,
              min_arm: int = 5, labels=None) -> pd.DataFrame:
    """Global and cluster-wise naive (difference-in-means) ACE, as in Table 10.
    N/A when a cluster has no treated or no control units (positivity fails)."""
    rows = []
    groups = [("Global", np.ones(len(y), bool))]
    groups += [(f"Cluster {c + 1}" + (f" ({labels[c]})" if labels and c in labels else ""),
                clusters == c) for c in sorted(np.unique(clusters))]
    for name, mask in groups:
        t1, t0 = y[mask & (treat == 1)], y[mask & (treat == 0)]
        row = dict(sample=name, N1=len(t1), N0=len(t0))
        if len(t1) == 0 or len(t0) == 0:
            row.update(ACE=np.nan, note="N/A (positivity violated)")
        elif len(t1) < min_arm or len(t0) < min_arm:
            row.update(ACE=np.nan, note="too few treated or controls")
        else:
            row.update(welch(t1, t0), note="")
        rows.append(row)
    return pd.DataFrame(rows)


def report_structure(result: dict, stability_bins=(0.70, 0.90)):
    """Spectral choices, contraction, Table 8-style edge list and cluster sizes."""
    res = result["resolution"]
    names = result["var_names"]
    diag = result["diagnostics"]
    section("Spectral embedding")
    print(f"  w_L = {result['w_L']:.3f}  (n_NG = {result['n_NG']} of {len(names)})")
    print(f"  embedding dimension K_hat : {result['K_hat']}")
    print(f"  cluster count k*          : {result['k_star']}  (Silhouette)")
    print(f"  Gap-statistic k           : {diag['k_gap']}")
    print("  Silhouette by k           : " +
          ", ".join(f"{k}: {v:.4f}" for k, v in sorted(diag["sil_scores"].items())))
    if diag["gap_scores"]:
        print("  Gap statistic by k        : " +
              ", ".join(f"{k}: {v:.3f}" for k, v in sorted(diag["gap_scores"].items())))
    section("Contracted variables")
    if res.contracted_pairs:
        for i, j in res.contracted_pairs:
            print(f"  {names[i]} <-> {names[j]}   OVS = {result['ovs_matrix'][i, j]:+.3f}   "
                  f"[{res.step[(i, j)]}]")
        print("  merged nodes: " + ", ".join(result["node_names"][k] for k in res.merged_nodes()))
    else:
        print("  none")
    isolated = [result["node_names"][k] for k in res.G_star.nodes if res.G_star.degree(k) == 0]
    if isolated:
        print("  isolated nodes: " + ", ".join(isolated))
    section("Directed edges of G_hat (cf. Table 8)")
    tbl = edge_summary(result, stability_bins)
    with pd.option_context("display.width", 140, "display.max_rows", 200):
        print(tbl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    zy = result["treatment_outcome"]
    if zy:
        section("Treatment -> outcome")
        nn = result["node_names"]
        print(f"  Z node '{nn[zy['node_Z']]}', Y node '{nn[zy['node_Y']]}', merged = {zy['merged']}, "
              f"directed path in G_hat = {zy['directed_path']}, "
              f"A_stab(Z, Y) = {zy['A_stab_ZY']:.3f}")
        print("  backdoor set of Z read off G_hat: " +
              (", ".join(names[i] for i in zy["backdoor_Z"]) or "(empty)"))
    return tbl


def cluster_sizes(clusters: np.ndarray, treat: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in sorted(np.unique(clusters)):
        m = clusters == c
        rows.append(dict(cluster=c + 1, N=int(m.sum()), share=m.mean(),
                         treated=int((treat[m] == 1).sum()), control=int((treat[m] == 0).sum())))
    return pd.DataFrame(rows)


def profiles(df: pd.DataFrame, clusters: np.ndarray, cols) -> pd.DataFrame:
    out = df[list(cols)].groupby(clusters).mean()
    out.index = [f"Cluster {c + 1} (N={int((clusters == c).sum()):,})" for c in out.index]
    return out
