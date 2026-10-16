# CaSPECT: Discovering Causally Homogeneous Subgroups via Directed Spectral Clustering

This repository contains the Python implementation of **CaSPECT (Causal Spectral Clustering)**.

CaSPECT is a spectral clustering framework in which the representation of the units is derived from an **estimated causal graph over the variables** rather than from the geometry of the raw covariates. The resulting embedding is a *fixed linear map of the standardized variables whose loadings are determined by the causal topology*. CaSPECT is therefore a method for causal-graph-informed dimension reduction and subgroup discovery. It does **not** cluster unit-level counterfactual contrasts.

---

## 📖 Table of Contents

* [Overview](#-overview)
* [What CaSPECT estimates](#-what-caspect-estimates)
* [Method](#-method)
* [Theory](#-theory)
* [Hyperparameters](#-hyperparameters)
* [Repository Structure](#-repository-structure)
* [Installation](#️-installation)
* [Usage](#-usage)
* [Experiments](#-experiments)
---

# Overview

Given observational data on covariates, a treatment $Z$ and an outcome $Y$, CaSPECT builds a low-dimensional representation of the units whose directions are chosen by the estimated causal mechanism. Principal components choose directions by marginal variance, and kernel spectral clustering by isotropic similarity. CaSPECT then partitions the units in that representation.

1. **DAG formation.** The skeleton is recovered with a bootstrap-stabilized PC algorithm. Edge directions are set by the **Orientation Validation Score (OVS)**, which combines PC bootstrap evidence with DirectLiNGAM. Edges the OVS cannot orient go through a resolution hierarchy that ends in edge contraction.
2. **Edge weights.** Each directed edge is weighted by the magnitude of its backdoor-identified average causal effect (the *total* effect), estimated by OLS or by double machine learning (DML). The weight is then multiplied by its bootstrap inclusion frequency.
3. **Laplacian.** A PageRank-teleported transition matrix and **Chung's directed Laplacian** provide a spectral basis.
4. **Clustering.** The standardized data are projected onto the leading non-trivial eigenvectors, and $k$-means is applied.

CaSPECT should be useful when the treatment-relevant structure among the units is aligned with directions of strong causal propagation in the graph. One example is a lack of overlap driven by covariates that are strong causes of treatment, as in the LaLonde analysis. It should not be expected to outperform methods that cluster estimated effect functions when the heterogeneity is confined to the treatment–outcome edge and is invisible in the variable-level graph.

---

# What CaSPECT estimates

* **Population target (Definition 1).** The population CaSPECT embedding is the linear map $\mathbf s \mapsto \boldsymbol\Phi_{K_0}^{\top}\mathbf s$, where $\mathbf s$ is the vector of standardized variables and $\boldsymbol\Phi_{K_0}$ holds the $K_0$ leading non-trivial eigenvectors of the population Chung Laplacian $L_0$. The population partition is the $k$-means partition of the law of $\boldsymbol\Phi_{K_0}^{\top}\mathbf s$.
* **Causally homogeneous subgroups (Definition 2).** A partition is $\varepsilon$-causally homogeneous if $\sum_j \frac{|C_j|}{n}\widehat{\mathrm{Var}}_{C_j}(\tau(x_i)) \le \varepsilon$.
* **Alignment condition (Proposition 2).** If $|\tau(x_i)-\tau(x_{i'})| \le M\lVert\tilde{\mathbf x}_i-\tilde{\mathbf x}_{i'}\rVert$, then $\sum_j \frac{|C_j|}{n}\widehat{\mathrm{Var}}_{C_j}(\tau) \le \frac{M^2}{n}\,\mathrm{WCSS}$. Compact clusters in the embedding are homogeneous in effect *only* under this (unverifiable) condition.
* **Treatment and outcome are projected.** $\mathbf X^{\mathrm{std}}$ contains the $Z$ and $Y$ columns, so a partition can reflect treatment assignment or observed outcomes directly. Clusters are **descriptive** subgroups, and within-cluster effect estimates are conditional on a data-dependent partition.

---

# Method

### Step 1: DAG formation

* Run PC at level $\alpha_n$ to get a CPDAG, then re-run PC on $B$ bootstrap resamples. Record the inclusion frequency $f_{uv}$ (the fraction of resamples in which $u$ and $v$ are adjacent; $f_{uv}=f_{vu}$) and the orientation frequency $f^{\to}_{uv}$ (the fraction in which $u\to v$ is returned). A resample that returns the edge undirected counts towards neither direction. Only edges with $f_{uv}\ge\theta$ enter the stable skeleton $\hat{\mathcal S}$.
* A per-variable Jarque–Bera test ($\alpha_{\mathrm{JB}}$) on OLS residuals gives $n_{\mathrm{NG}}$, the number of variables that reject Gaussianity. The LiNGAM weight is

  $$w_L = \frac{n_{\mathrm{NG}}}{q}\,w_{\max},\qquad w_{\mathrm{PC}} = 1-w_L .$$

* **Orientation Validation Score** (Eq. 2), for each edge in $\hat{\mathcal S}$:

  $$\mathrm{OVS}_{uv} = w_{\mathrm{PC}}\, f_{uv}\, e^{\mathrm{PC}}_{uv} + w_L\, e^{L}_{uv},\qquad
  e^{\mathrm{PC}}_{uv} = \rho_{uv}-\rho_{vu},\qquad
  e^{L}_{uv} = \mathrm{sign}\big(|\hat B^{L}_{uv}|-|\hat B^{L}_{vu}|\big),$$

  where $\rho_{uv}=f^{\to}_{uv}/f_{uv}$ and $\hat B^{L}_{uv}$ is the DirectLiNGAM coefficient of $u$ in the structural equation of $v$. The LiNGAM term compares **magnitudes**, because the coefficients are signed. Orient $u\to v$ if $\mathrm{OVS}_{uv}>\gamma$, orient $v\to u$ if $\mathrm{OVS}_{uv}<-\gamma$, and otherwise pass the edge on.
* **Resolution hierarchy** for $|\mathrm{OVS}_{uv}|\le\gamma$, with acyclicity checked after each step:
  1. *Acyclicity filter.* If exactly one orientation is cycle-free, use it.
  2. *Meek rules.* All four propagation rules, applied exhaustively.
  3. *Domain or temporal ordering* (`domain_order`).
  4. *Edge contraction.* Merge $u$ and $v$ into a node $N_S$, labelled by its members (e.g. `re74/re75`).

### Step 2: Causal edge weights

* For each directed edge $u\to v$, the backdoor set $\mathbf W_{uv}$ is read off $\hat G$ (the parents of $u$'s node). A RESET test at level $\alpha_{\mathrm{RESET}}$ routes the edge either to **OLS** (the coefficient on $u$ in the regression of $v$ on $u$ and $\mathbf W_{uv}$) or to **DML** with $n_{\mathrm{fold}}$-fold cross-fitting and GAM or random-forest nuisances:

  $$\hat\tau^{\mathrm{DML}}_{u\rightsquigarrow v} = \frac{\sum_i \tilde R^u_i\tilde R^v_i}{\sum_i(\tilde R^u_i)^2},\qquad \hat A_{uv} = |\hat\tau_{u\rightsquigarrow v}| .$$

* Stability weighting: $\hat A^{\mathrm{stab}}_{uv} = f_{uv}\,\hat A_{uv}$.
* **Merged nodes:** $\hat A^{\mathrm{stab}}_{N_S w} = |S|^{-1}\sum_{u\in S}\hat A^{\mathrm{stab}}_{uw}$ and $\hat A^{\mathrm{stab}}_{w N_S} = |S|^{-1}\sum_{u\in S}\hat A^{\mathrm{stab}}_{wu}$, where an absent edge counts as 0.

### Step 3: Chung Laplacian and spectral decomposition

$$\hat P = (1-\alpha_{\mathrm{PR}})\,\hat D^{-1}\hat{\mathbf A}^{\mathrm{stab}} + \frac{\alpha_{\mathrm{PR}}}{|\mathbf V^*|}\mathbf 1\mathbf 1^{\top}\quad\text{(Eq. 4, dangling rows of } \hat D^{-1}\hat{\mathbf A}^{\mathrm{stab}}\text{ replaced by the uniform row)}$$

$$\hat L = \mathbf I - \tfrac12\big(\hat\Pi^{1/2}\hat P\,\hat\Pi^{-1/2} + \hat\Pi^{-1/2}\hat P^{\top}\hat\Pi^{1/2}\big),\qquad \hat\Pi=\mathrm{diag}(\hat\pi)\quad\text{(Eq. 5)}$$

$\hat L$ is symmetric and positive semi-definite, with $\hat\lambda_1=0$ and $\hat{\boldsymbol\varphi}_1\propto\hat\pi^{1/2}$. The embedding dimension is the index of the largest *non-trivial* gap,

$$\hat K = \arg\max_{2\le j\le |\mathbf V^*|-1}(\hat\lambda_{j+1}-\hat\lambda_j),\qquad \hat{\boldsymbol\Phi}_{\hat K} = [\hat{\boldsymbol\varphi}_2,\dots,\hat{\boldsymbol\varphi}_{\hat K+1}] .$$

### Step 4: Causal spectral embedding and clustering

$$\tilde{\mathbf X} = \mathbf X^{\mathrm{std}}\,\hat{\boldsymbol\Phi}_{\hat K}\in\mathbb R^{n\times\hat K}\quad\text{(Eq. 7)},$$

where $\mathbf X^{\mathrm{std}}$ has one standardized column per element of $\mathbf V^*$; a merged node is the element-wise average of its members' standardized columns. $k$-means is run for $k\in\{2,\dots,k_{\max}\}$. $k^*$ maximizes the Silhouette score, and the Gap statistic is reported alongside.

---

# Theory

Under faithfulness, causal sufficiency, non-Gaussianity, per-edge linearity or identifiable nonlinearity, causal identification, spectral regularity and bootstrap PC consistency (Assumptions 1–7):

* **Proposition 1 (OVS consistency).** On a compelled edge, $\mathbb P[\mathrm{OVS}_{uv}>\gamma]\to1$. If Assumption 3 fails, the same holds provided $w_L<(1-\gamma)/2$. On a reversible edge it holds provided $w_L>\gamma$. Asymptotically, no edge reaches the resolution hierarchy.
* **Theorem 1 and Corollary 1.** $\hat A^{\mathrm{stab}}_{uv}\xrightarrow{p}|\tau_{u\rightsquigarrow v}|$ for every true edge, and $\hat A^{\mathrm{stab}}_{uv}\xrightarrow{p}0$ for spurious pairs retained at finite $n$.
* **Theorem 2.** The skeleton, orientation, edge weights, Laplacian, selected dimension and eigenspace are **consistent in probability**, with $\lVert\sin\Theta(\boldsymbol\Phi_{K_0},\hat{\boldsymbol\Phi}_{\hat K})\rVert_F=\mathcal O_p(q\,r_n/\delta_{K_0})$.

The theorem does **not** cover the final $k$-means partition or the choice of $k^*$. Bound (1) on false selections is used only as a heuristic motivation for thresholding at $\theta$.

The pipeline prints whether the configured $w_L$ satisfies the conditions of Proposition 1.

---

# Hyperparameters

The defaults are the configuration used throughout the paper (Table 2):

| Symbol | Argument | Meaning | Value |
|---|---|---|---|
| $M$ | `--M` | Monte Carlo replications (simulation) | 100 |
| $B$ | `B` | Bootstrap resamples for PC | 100 |
| $\alpha_n$ | `alpha_ci` | PC conditional-independence level | 0.05 |
| $\theta$ | `theta` | Skeleton stability threshold | 0.55 |
| $\gamma$ | `gamma` | Edge orientation threshold | 0.15 |
| $w_{\max}$ | `w_max` | Cap on the LiNGAM weight | 0.20 |
| $w_{\mathrm{PC}}$ | — | PC weight, $1-w_L$ | 0.80 |
| $\alpha_{\mathrm{JB}}$ | `alpha_jb` | Jarque–Bera level | 0.10 |
| $\alpha_{\mathrm{RESET}}$ | `alpha_reset` | RESET test level | 0.05 |
| $\alpha_{\mathrm{PR}}$ | `alpha_pr` | PageRank teleportation | 0.15 |
| $n_{\mathrm{fold}}$ | `n_fold` | DML cross-fitting folds | 5 |
| $k_{\max}$ | `k_max` | Largest $k$ searched | 9 |
| $q$ | — | Variables in the simulation | 8 |
| $\rho$ | — | DGP edge probability | 0.35 |

---

# Repository Structure

```text
├── LICENSE
├── README.md
├── CHANGELOG.md            what changed to bring the code in line with the paper
├── requirements.txt
├── Images/                 DAG figures (LaLonde, IHDP, 401(k))
├── datasets/
│   ├── lalonde_cps1.csv    LaLonde NSW treated + CPS1 controls (Section 9.1)
│   ├── ihdp.csv            IHDP NPCI replicate 1, no header (Section 9.2)
│   └── 401(k).csv          SIPP 401(k) eligibility data (Section 9.3)
└── src/
    ├── caspect.py          the CaSPECT pipeline (Algorithm 1) and plots
    ├── simulation_full.py  simulation study (Section 7: Tables 3–6) + baselines (Table 7)
    ├── baselines.py        comparators for Section 8
    ├── analysis_utils.py   shared reporting for the case studies
    ├── lalonde.py          Section 9.1 (Tables 8–10, Figures 1–3)
    ├── ihdp.py             Section 9.2 (Figures 4–6)
    └── k401.py             Section 9.3 (Figures 7–9)
```

---

# Installation

```bash
git clone https://github.com/Arghyapa/CaSPECT.git
cd CaSPECT
pip install -r requirements.txt
```

`causal-learn` (PC), `lingam` (DirectLiNGAM) and `pygam` (GAM nuisances) are used when available. If they are missing, the pipeline falls back to a built-in PC, drops the LiNGAM term, or uses random forests, respectively. `econml` is needed only for the causal-forest baseline.

---

# Usage

### Python API

```python
import sys; sys.path.insert(0, "src")
from caspect import caspect, edge_summary

result = caspect(
    X,                                   # n x q array: covariates, treatment, outcome
    var_names=names,
    treat_idx="treat", outcome_idx="re78",
    domain_order=None,                   # optional, earliest first
    visualise=True, save_dir="figures",  # Figures 1-3 style plots
)
result["clusters"]        # labels 0..k*-1, ordered by decreasing cluster size
result["K_hat"], result["k_star"]
result["node_names"]      # V*, merged nodes joined by '/'
edge_summary(result)      # Table 8 style edge list
```

Ablations: `pc_only=True` (A1, $e^L_{uv}=0$), `no_stability=True` (A2, $f_{uv}=1$), `symmetrise=True` (A3, symmetrization instead of contraction). The old entry point `CSC_PC_PIPELINE` is kept as a wrapper and accepts the old argument names (`tau`, `w_L_max`, `dml_k`).

### Case studies

```bash
cd src
python lalonde.py            # Section 9.1
python ihdp.py               # Section 9.2
python k401.py               # Section 9.3
python lalonde.py --help     # --B, --theta, --gamma, --w-max, --seed, --no-gap, --no-plots, --fig-dir
```

### Simulation study

```bash
cd src
python simulation_full.py                         # Tables 3-7 with M = 100 (several hours on one core)
python simulation_full.py --part S1 --n 1000      # one setting / sample size
python simulation_full.py --part ablation --n-jobs 8
python simulation_full.py --part baselines
python simulation_full.py --quick                 # smoke test
```

Outputs, including per-replication CSVs, summaries and `simulation_results.png`, are written to `--out-dir` (default `simulation_output/`).

---

# Experiments

## Simulation study (Section 7)

The DGP is a linear SEM on $q=8$ variables $\{X_1,\dots,X_6,Z,Y\}$ with a random topological order. Edges are included with probability $\rho=0.35$, and $\beta_{uv}\sim s_{uv}\cdot\mathrm{Unif}(0.3,0.8)$ with a Rademacher sign. Three planted clusters of about $n/3$ units each differ only in the treatment intercept, $Z_i=\eta_c+\sum_{u\in\mathrm{Pa}(Z)}\beta_{uZ}X_{iu}+\varepsilon_{iZ}$ with $\eta=(-1,0,+1)$. Because the coefficients are shared, $\tau_1=\tau_2=\tau_3$: this is a **propensity-shift benchmark**, so the ARI column tests something other than recovery of effect heterogeneity.

| Setting | Description |
|---|---|
| S1 | Linear DAG, standardized $t_5$ errors (all assumptions hold), $n\in\{500,1000,2000\}$ |
| S2 | Mixed linearity: the three largest-$\lvert\beta\rvert$ edges enter as $\beta_{uv}\sin(\pi u)$ |
| S3 | Latent confounder $H\sim\mathcal N(0,1)$ on two variables $a,b\notin\{Z,Y\}$, $\zeta_a=\zeta_b=0.4$ |
| Ablation | S1, $n=1000$: A1 PC-only orientation, A2 no stability weighting, A3 symmetrization |
| Baselines | S1, $n=1000$: $k$-means and spectral clustering on $\mathbf X$, causal $k$-means, hierarchical causal clustering, R-learner + $k$-means, causal forest + $k$-means |

The metrics are:

* ARI against the planted labels.
* $\mathrm{Acc}_{\mathrm{or}}$: the share of stable-skeleton edges whose final direction matches $G_0$.
* Cluster-level $\mathrm{Bias}_c$ and $\mathrm{RMSE}_c$ of the DML estimate. Each estimate is restricted to the estimated cluster that the Hungarian algorithm matches to planted cluster $c$.

All methods share the $k$-selection rule, the DML estimator and the matching. The baseline table reports $z=\Delta/\mathrm{SE}(\Delta)$ for $\Delta=\mathrm{ARI}_{\text{method}}-\mathrm{ARI}_{\text{CaSPECT}}$.

> **Note.** This release corrects the decoding of edge directions (see `CHANGELOG.md`). The orientation accuracy it produces is therefore not comparable with the values in the current tables of the paper.

## LaLonde CPS1 (Section 9.1)

The data are 185 NSW treated units and 15,992 CPS controls, with earnings on the $\log(1+y)$ scale. The paper reports a two-cluster partition. Cluster 1 holds 13,047 units, all controls, so positivity fails there. Cluster 2 holds all 185 treated units and 2,945 controls. Within Cluster 2 the naive effect estimate changes sign:

| Sample | $N_1$ | $N_0$ | ACE (log1p re78), as reported in the paper |
|---|---|---|---|
| Global | 185 | 15,992 | −1.7011 |
| Cluster 1 (incomparable) | 0 | 13,047 | N/A |
| Cluster 2 (comparable) | 185 | 2,945 | +1.3998 |

This is an **exploratory** finding about overlap, not confirmatory causal evidence. The partition depends on `treat` and `re78`, and no covariate-balance or overlap diagnostics are reported.

> **Reproducibility note.** The published partition was produced by the previous code, which decoded edge directions in reverse. With the corrected decoding in this release, `lalonde.py` gives a different graph, with `treat` contracted into a merged node, and five clusters. See `CHANGELOG.md`, section G.

## IHDP (Section 9.2)

The data are NPCI replicate 1: 25 covariates plus a treatment, with the outcome Box-Cox transformed. The pipeline concentrates the treated infants in one cluster, which is evidence of structural heterogeneity, partly mechanical because the treatment column is projected. The ground-truth cluster effects $\tau_c=\overline{\mu_1-\mu_0}$ are reported for comparison. This is the only analysis where the effects are known, and it rests on a single replicate.

## 401(k) (Section 9.3)

The treatment is `e401` and the outcome is `net_tfa` (signed-log in the pipeline), with ten covariates. A standalone DML cross-check of the global ACE (the paper reports \$8,795, 95% CI [\$7,827, \$9,762]) and of the cluster-level ACEs is run outside the DAG. In the reported run, `pira` and `e401` were contracted, which removed the treatment–outcome pathway from the graph. The pipeline now warns when this happens. With the corrected direction decoding in this release, that contraction does not occur and `e401 → net_tfa` stays in the graph.

> Numbers quoted in this section are those reported in the paper. Rerunning the scripts in this release may give different values (see `CHANGELOG.md`).

---
