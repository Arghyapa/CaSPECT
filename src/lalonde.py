import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats

df = pd.read_csv("lalonde_cps1.csv")

treatment  = "treat"
outcome    = "re78"
covariates = [c for c in df.columns if c not in [treatment, outcome]]

ordered_cols = covariates + [treatment, outcome]
df = df[ordered_cols]

df = df.dropna().reset_index(drop=True)

for col in ["re74", "re75", "re78"]:
    if col in df.columns:
        df[col] = np.log1p(df[col])

X_raw = df.values.astype(float)
n_cov = len(covariates)

scaler = StandardScaler()
X_raw[:, :n_cov] = scaler.fit_transform(X_raw[:, :n_cov])

var_names   = ordered_cols
treat_idx   = ordered_cols.index(treatment)
outcome_idx = ordered_cols.index(outcome)

result = CSC_PC_PIPELINE(
    X_raw,
    var_names    = var_names,
    alpha_ci     = 0.05,
    B            = 100,
    theta        = 0.50,
    tau          = 0.15,
    w_L_max      = 0.20,
    alpha_pr     = 0.15,
    alpha_reset  = 0.05,
    dml_k        = 5,
    domain_order = None,
    B_ref        = 50,
    seed         = 42,
    visualise    = True,
    verbose      = True,
)

clusters         = result["clusters"]
embedding        = result["embedding"]
A                = result["adjacency"]
edges            = result["edges"]
K_star           = result["K_star"]
k_star           = result["k_star"]
diag             = result["diagnostics"]
contracted_pairs = result["contracted_pairs"] 

df["cluster"] = clusters

print("\n" + "=" * 52)
print("  CSC-PC RESULTS  –  LaLonde CPS1")
print("=" * 52)

print("\n── Spectral embedding ──────────────────────────────")
print(f"  Embedding dimension  K*  : {K_star}")
print(f"  Cluster count        k*  : {k_star}")
print(f"  Silhouette k             : {diag['k_silhouette']}")
print(f"  Gap-statistic k          : {diag['k_gap']}")
print(f"  Criteria agree           : {diag['criteria_agree']}")

print("\n── Contracted variable pairs ───────────────────────")
if contracted_pairs:
    for (i, j) in contracted_pairs:
        ni = var_names[i] if i < len(var_names) else f"X{i}"
        nj = var_names[j] if j < len(var_names) else f"X{j}"
        print(f"  {ni}  ←merged with→  {nj}  "
              f"(surviving node index: {min(i,j)}, relabelled '{ni}/{nj}')")
else:
    print("  None")

print("\n── Cluster sizes ───────────────────────────────────")
print(df["cluster"].value_counts().sort_index().to_string())

print("\n── Adjacency matrix (bootstrap-weighted |ATE|) ─────")
A_df = pd.DataFrame(A, index=var_names, columns=var_names)
print(A_df.round(4).to_string())

print("\n── Directed edges ──────────────────────────────────")
for (i, j), direction in edges.items():
    u, v = direction
    if v == -1:
        u_name = var_names[u] if u < len(var_names) else f"Node{u}"
        vi_name = var_names[i] if i < len(var_names) else f"Node{i}"
        vj_name = var_names[j] if j < len(var_names) else f"Node{j}"
        print(f"  [contracted]  {vi_name} ←→ {vj_name}  "
              f"→ merged into node '{u_name}'")
    elif u < len(var_names) and v < len(var_names):
        print(f"  {var_names[u]}  →  {var_names[v]}")


print("\n[Plot] Generating standalone t-SNE cluster map …")
X_scaled = StandardScaler().fit_transform(X_raw)
perp = min(30, X_scaled.shape[0] - 1)
X_tsne = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                c=clusters, cmap="viridis", alpha=0.75, s=20)
# plt.colorbar(sc, ax=ax, label="Cluster")
ax.set_title(f"t-SNE Cluster Map  (k*={k_star}, K*={K_star})\n"
             "CSC-PC  –  LaLonde CPS1")
ax.set_xlabel("t-SNE dim 1")
ax.set_ylabel("t-SNE dim 2")
plt.tight_layout()
plt.show()


treated_all = df[df[treatment] == 1][outcome]
control_all = df[df[treatment] == 0][outcome]
global_ate  = treated_all.mean() - control_all.mean()
t_stat, t_pval = stats.ttest_ind(treated_all, control_all, equal_var=False)

print("\n── Global ATE ──────────────────────────────────────")
print(f"  Treated N         : {len(treated_all)}")
print(f"  Control N         : {len(control_all)}")
print(f"  ATE (log re78)    : {global_ate:+.4f}")
print(f"  Welch t / p-value : {t_stat:.3f} / {t_pval:.4f}")

print("\n── Cluster-wise ATE ────────────────────────────────")

cluster_results = []
for c in sorted(df["cluster"].unique()):
    sub     = df[df["cluster"] == c]
    treated = sub[sub[treatment] == 1][outcome]
    control = sub[sub[treatment] == 0][outcome]

    row = {
        "cluster"   : c,
        "size"      : len(sub),
        "n_treated" : len(treated),
        "n_control" : len(control),
        "ate_raw"   : np.nan,
        "t_stat"    : np.nan,
        "p_value"   : np.nan,
        "ci_low"    : np.nan,
        "ci_high"   : np.nan,
        "note"      : "",
    }

    if len(treated) > 5 and len(control) > 5:
        ate = treated.mean() - control.mean()
        t_s, p_v = stats.ttest_ind(treated, control, equal_var=False)

        se = np.sqrt(treated.var(ddof=1)/len(treated) +
                     control.var(ddof=1)/len(control))
        df_w = (treated.var(ddof=1)/len(treated) +
                control.var(ddof=1)/len(control))**2 / (
                (treated.var(ddof=1)/len(treated))**2/(len(treated)-1) +
                (control.var(ddof=1)/len(control))**2/(len(control)-1))
        t_crit = stats.t.ppf(0.975, df_w)

        row.update({
            "ate_raw" : ate,
            "t_stat"  : t_s,
            "p_value" : p_v,
            "ci_low"  : ate - t_crit * se,
            "ci_high" : ate + t_crit * se,
        })
    else:
        row["note"] = "insufficient treated/control split"

    cluster_results.append(row)

    print(f"\n  Cluster {c}:")
    print(f"    Size              : {row['size']}")
    print(f"    Treated / Control : {row['n_treated']} / {row['n_control']}")
    if row["note"]:
        print(f"    NOTE: {row['note']}")
    else:
        print(f"    ATE (log re78)    : {row['ate_raw']:+.4f}")
        print(f"    95% CI            : [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]")
        print(f"    t / p-value       : {row['t_stat']:.3f} / {row['p_value']:.4f}")
        sig = ("***" if row["p_value"] < 0.01 else
               "**"  if row["p_value"] < 0.05 else
               "*"   if row["p_value"] < 0.10 else "")
        if sig:
            print(f"    Significance      : {sig}")

print("\n── Cluster covariate profiles ──────────────────────")
profile_cols = covariates + [treatment, outcome]
profile = (df.groupby("cluster")[profile_cols].mean().round(3))
print(profile.to_string())


valid = [r for r in cluster_results if not r["note"]]
if valid:
    labels = [f"Cluster {r['cluster']}" for r in valid]
    ates   = [r["ate_raw"]  for r in valid]
    ci_lo  = [r["ate_raw"] - r["ci_low"]  for r in valid]
    ci_hi  = [r["ci_high"] - r["ate_raw"] for r in valid]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, ates, yerr=[ci_lo, ci_hi], capsize=5,
           color="steelblue", alpha=0.75, error_kw={"elinewidth": 1.5})
    ax.axhline(global_ate, color="tomato", linestyle="--",
               linewidth=1.5, label=f"Global ATE = {global_ate:+.3f}")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ATE  (log re78)")
    ax.set_title("Cluster-wise ATE with 95% CI\n(CSC-PC, LaLonde CPS1)")
    ax.legend()
    plt.tight_layout()
    plt.show()


print("\n── Silhouette scores by k ──────────────────────────")
for k, sc in sorted(diag["sil_scores"].items()):
    marker = "  ← k*" if k == k_star else ""
    print(f"  k={k}: {sc:.4f}{marker}")

print("\n── Gap statistic by k ──────────────────────────────")
for k, gp in sorted(diag["gap_scores"].items()):
    marker = "  ← k_gap" if k == diag["k_gap"] else ""
    print(f"  k={k}: {gp:.4f}{marker}")
