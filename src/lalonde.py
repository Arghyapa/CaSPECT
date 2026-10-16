"""
CaSPECT on the LaLonde CPS1 data (Section 9.1).
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caspect import caspect                                  # noqa: E402
from analysis_utils import (DATA_DIR, ace_table, cluster_sizes, common_parser,  # noqa: E402
                            header, pipeline_kwargs, profiles, report_structure, section)

TREATMENT, OUTCOME = "treat", "re78"
EARNINGS = ["re74", "re75", "re78"]


def load(path=os.path.join(DATA_DIR, "lalonde_cps1.csv")) -> pd.DataFrame:
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    covariates = [c for c in df.columns if c not in (TREATMENT, OUTCOME)]
    df = df[covariates + [TREATMENT, OUTCOME]].copy()
    for col in EARNINGS:
        df[col] = np.log1p(df[col])
    return df


def main(argv=None):
    ap = common_parser(__doc__, default_fig_dir="figures/lalonde")
    ap.add_argument("--domain-order", nargs="*", default=None,
                    help="optional variable names, earliest first (e.g. re74 re75)")
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    df = load()
    names = list(df.columns)
    header(f"CaSPECT  -  LaLonde CPS1   (n = {len(df):,}, q = {len(names)})")
    result = caspect(df.values, var_names=names, treat_idx=TREATMENT, outcome_idx=OUTCOME,
                     domain_order=args.domain_order, **pipeline_kwargs(args))
    clusters = result["clusters"]
    treat = df[TREATMENT].values

    report_structure(result)

    section("Cluster sizes")
    print(cluster_sizes(clusters, treat).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    section("Mean covariate profiles by cluster (cf. Table 9; earnings on log1p scale)")
    cols = ["age", "educ", "black", "hisp", "married", "nodegr", "re74", "re75", TREATMENT, OUTCOME]
    print(profiles(df, clusters, cols).round(3).to_string())

    section("Average causal effect estimates, log1p re78 (cf. Table 10)")
    tbl = ace_table(treat, df[OUTCOME].values, clusters)
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n  Difference in means; t, p and the 95% interval use Welch's test. These are\n"
          "  exploratory: no covariate-balance or overlap diagnostics are reported, and the\n"
          "  partition itself depends on treat and re78.")
    return result


if __name__ == "__main__":
    main()
