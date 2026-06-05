# CaSPECT: Discovering Causally Homogeneous Subgroups via Directed Spectral Embedding

This repository contains the official Python implementation of **CaSPECT (Causal Spectral Clustering)**, a novel framework for discovering **causally homogeneous subgroups** from observational data.

CaSPECT integrates **causal discovery**, **average treatment effect estimation**, and **directed spectral graph theory** into a unified pipeline. Unlike conventional clustering methods that rely on covariate similarity, CaSPECT defines similarity through the topology of a learned **Directed Acyclic Graph (DAG)** and the propagation of causal influence across variables.

---

## 📖 Table of Contents

* [Overview](#-overview)
* [Key Contributions](#-key-contributions)
* [Methodology](#-methodology)
* [Repository Structure](#-repository-structure)
* [Installation](#️-installation)
* [Usage](#-usage)
* [Experimental Results](#-experimental-results)
* [Ablation Studies](#-ablation-studies)
* [Theoretical Guarantees](#-theoretical-guarantees)
* [Citation](#-citation)

---

# 🔬 Overview

Traditional clustering methods group observations according to Euclidean distance or density in feature space. Such approaches often fail when the underlying heterogeneity is driven by **causal mechanisms rather than covariate similarity**.

CaSPECT addresses this challenge by:

1. Recovering a causal DAG using a bootstrap-stabilized PC algorithm.
2. Resolving orientation ambiguities through a novel **Orientation Validation Score (OVS)** that combines PC and DirectLiNGAM evidence.
3. Estimating directed causal edge strengths using backdoor-adjusted Average Treatment Effects (ATEs).
4. Constructing Chung's Directed Laplacian from the resulting causal graph.
5. Embedding observations into a causal spectral space.
6. Discovering causally homogeneous subgroups through clustering in the embedded space.

Unlike propensity-score matching, trimming, or manual subgroup construction, CaSPECT discovers comparable populations automatically through causal geometry.


### Orientation Validation Score (OVS)

For every candidate edge ((u,v)),


  $$ \OVS_{uv} = w_{\mathrm{PC}} \cdot f_{uv} \cdot \underbrace{(\rho_{uv} - \rho_{vu})}_{\delta_{uv}^{\mathrm{PC}}} + w_L \cdot \underbrace{\sign(\hat{B}_{uv} - \hat{B}_{vu})}_{\delta_{uv}^{L}},$$

where

* (f_{uv}) = bootstrap inclusion frequency
* (\rho_{uv}) = orientation frequency
* (\hat B_{uv}) = DirectLiNGAM coefficient
* (w_{PC}=0.8)
* (w_L=0.2)

The score combines causal discovery and non-Gaussian identifiability information to orient DAG edges robustly.

---

# ✨ Key Contributions

### 🔹 Causal Topology-Based Clustering

Rather than clustering on observed covariates, CaSPECT clusters individuals according to the pathways through which causal influence propagates.

### 🔹 Orientation Validation Score (OVS)

A novel orientation mechanism combining:

* Bootstrap-stable PC orientations
* DirectLiNGAM directional evidence

to obtain robust edge directions in finite samples.

### 🔹 Causal Edge Weighting

Directed edges are weighted by backdoor-identified Average Treatment Effects:

* Ordinary Least Squares (OLS) for linear edges.
* Double Machine Learning (DML) for nonlinear edges.

### 🔹 Directed Spectral Geometry

CaSPECT employs Chung's Directed Laplacian

$$ L = I - \frac{1}{2}(P + P^*) $$

to capture causal diffusion over the graph.

### 🔹 Automatic Common Support Discovery

Structurally incomparable units are embedded far apart in spectral space, naturally separating populations that violate positivity assumptions.

### 🔹 Theoretical Guarantees

The framework provides:

* OVS consistency
* Consistent ACE estimation
* Spectral embedding stability
* Almost sure convergence of the full pipeline

under standard causal assumptions.

---

# 📂 Repository Structure

```text
├── LICENSE
├── README.md
├── main_pipeline.py
├── run_all.sh

├── src/
│   ├── caspect.py
│   ├── dag_builder.py
│   ├── ovs.py
│   ├── edge_weight_estimator.py
│   ├── directed_laplacian.py
│   ├── spectral_embedding.py
│   ├── clustering.py
│   ├── simulations.py
│   └── utils.py

├── datasets/
│   ├── lalonde/
│   ├── ihdp/
│   └── 401k/

├── results/
│   ├── figures/
│   └── tables/
```

---

# ⚙️ Installation

Clone the repository and install the required dependencies.

```bash
git clone https://github.com/Arghyapa/CaSPECT.git

cd CaSPECT

pip install numpy scipy pandas scikit-learn networkx

pip install causallearn lingam econml statsmodels

pip install matplotlib seaborn
```

---

# 🚀 Usage

### Run the Full Pipeline

```bash
bash run_all.sh
```

### Run on the LaLonde Dataset

```bash
python main_pipeline.py \
    --dataset lalonde \
    --alpha_ci 0.05 \
    --bootstrap 100 \
    --theta 0.60
```

### Run Simulation Studies

```bash
python simulations.py \
    --setting S1 \
    --sample_size 1000
```

---

# 📊 Experimental Results

CaSPECT was evaluated on both synthetic causal graphs and real-world causal inference benchmarks.

## Synthetic Simulations

Three increasingly challenging settings were studied.

### Setting S1: Linear DAG + Non-Gaussian Errors

* Fully satisfies all assumptions.
* Demonstrates asymptotic consistency of OVS and ACE estimation.
* Serves as the performance ceiling of the method.

### Setting S2: Mixed Linearity

Three randomly selected nonlinear edges are introduced.

* RESET identifies nonlinear mechanisms.
* DML estimates nonlinear ACEs.
* Tests robustness under model misspecification.

### Setting S3: Latent Confounding

A hidden variable introduces violations of causal sufficiency.

Despite this violation:

* Graph recovery remains stable.
* OVS retains high orientation accuracy.
* ACE estimation remains reliable.

---

## LaLonde Benchmark

CaSPECT automatically separates:

### Cluster 1

* Economically stable CPS participants.
* No treated individuals.
* Positivity violated.

### Cluster 2

* Contains all treated units.
* Structurally comparable controls.
* Valid causal effect estimation.

### Average Treatment Effects

| Sample             | ACE (log re78) |
| ------------------ | -------------- |
| Global Population  | -1.701         |
| Comparable Cluster | +1.400         |

The positive treatment effect recovered within the comparable subgroup aligns closely with the original NSW experimental findings.

---

## IHDP Dataset

CaSPECT identifies latent treatment-effect heterogeneity and discovers causally meaningful subgroup structures.

---

## 401(k) Dataset

The framework uncovers heterogeneous retirement-savings effects and demonstrates scalability to larger observational datasets.

---

# 🔬 Ablation Studies

Extensive ablations validate the importance of each component.

## 1. OVS vs PC-Only Orientation

Removing LiNGAM evidence reduces orientation accuracy and increases DAG ambiguity.

## 2. Bootstrap Stability Weighting

Replacing

[
A^{stable}_{uv}
===============

f_{uv}A_{uv}
]

with unweighted edges introduces spurious causal connections and degrades embedding quality.

## 3. Edge Contraction vs Symmetrization

Symmetrization creates artificial cycles

[
u \rightarrow v \rightarrow u
]

which violate DAG assumptions.

Edge contraction preserves acyclicity and causal interpretability.

---

# 📐 Theoretical Guarantees

### Proposition 1: OVS Consistency

The Orientation Validation Score satisfies

[
OVS_{uv}
\xrightarrow{a.s.}
\pm 1
]

for all true edges.

---

### Theorem 1: Consistent Causal Edge Weights

Bootstrap-weighted adjacency entries converge to the true ACE magnitudes:

[
A^{stable}*{uv}
\xrightarrow{a.s.}
|\tau*{uv}|
]

---

### Spectral Stability

Applying Davis–Kahan perturbation theory yields

[
|
\sin\Theta(V_K,\hat V_K)
|_F
===

O!\left(
\frac{|V|}
{\delta_{K^*}\sqrt n}
\right)
]

ensuring stable spectral embeddings as sample size grows.

---

# 🌟 Why CaSPECT?

Most causal clustering approaches operate directly on estimated treatment effects.

CaSPECT instead clusters through the **causal geometry of the data-generating process itself**. By combining causal discovery, causal effect estimation, and directed spectral graph theory, the framework uncovers latent subgroups that share common causal propagation pathways rather than merely similar covariates.

This allows CaSPECT to:

* Discover causally homogeneous populations.
* Automatically enforce common support.
* Separate comparable and incomparable individuals.
* Recover treatment effects hidden by severe confounding.
* Provide interpretable graph-based explanations for subgroup formation.

---

# 📜 Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{pratihar2026caspect,
  title={CaSPECT: Discovering Causally Homogeneous Subgroups via Directed Spectral Embedding},
  author={Pratihar, Arghya and Chakraborty, Shinjon and Das, Swagatam},
  journal={Stat},
  year={2026}
}
```

---

## ⭐ Acknowledgements

This work was developed at the **Indian Statistical Institute (ISI), Kolkata**, and explores the intersection of:

* Causal Inference
* Spectral Graph Theory
* Directed Laplacians
* Treatment Effect Heterogeneity
* Unsupervised Learning

We hope CaSPECT serves as a useful framework for researchers interested in discovering latent causal structure from observational data.

