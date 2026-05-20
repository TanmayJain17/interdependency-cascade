"""
Fit Weibull buffer distributions from observed Sandy 2012 outage records.

Reads observed time-to-failure data from CSVs in data/sandy_validation/
and fits per-infrastructure-type Weibull (median, shape) parameters via MLE
with right-censoring (facilities that didn't fail are censored observations).

Updates config/buffer_distributions.yaml in place for any type with sufficient
data, marking source='sandy_fit'. Writes a diagnostic report to outputs/.

CSV format (one per infra type):
    facility:        identifier
    hours_to_failure: hours from t=0 (flood onset) to backup failure
    censored:        True if facility didn't fail in observation window
    flood_depth_m:   flood depth at facility (informational)
    source:          citation
    notes:           free text

Run from project root:
    python scripts/fit_buffer_distributions.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cascade.stochastic_buffer import (  # noqa: E402
    fit_weibull_from_observations,
    load_buffer_config,
    weibull_survival,
)

CONFIG_PATH = ROOT / "config" / "buffer_distributions.yaml"
SANDY_DIR = ROOT / "data" / "sandy_validation"
REPORT_PATH = ROOT / "outputs" / "sandy_fit_report.md"

MIN_FAILURES_TO_FIT = 3  # Below this, keep the engineering prior

SANDY_FILES = {
    "hospital": SANDY_DIR / "sandy_hospital_outages.csv",
    "telecom": SANDY_DIR / "sandy_telecom_outages.csv",
    "subway": SANDY_DIR / "sandy_subway_outages.csv",
}


def load_observations(csv_path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df = df[df["hours_to_failure"] > 0].copy()
    df["censored"] = df["censored"].astype(bool)
    return df


def fit_one_type(infra_type, obs_df, prior):
    n_total = len(obs_df)
    n_failed = int((~obs_df["censored"]).sum())
    n_censored = int(obs_df["censored"].sum())

    result = {
        "infra_type": infra_type,
        "n_total": n_total,
        "n_failed": n_failed,
        "n_censored": n_censored,
        "prior_median": prior["median_hours"],
        "prior_shape": prior["shape"],
        "fitted_median": None,
        "fitted_shape": None,
    }

    if n_failed < MIN_FAILURES_TO_FIT:
        result["status"] = f"skipped: only {n_failed} failures (need {MIN_FAILURES_TO_FIT})"
        return result

    times = obs_df["hours_to_failure"].values
    censored = obs_df["censored"].values

    try:
        median, shape, ll = fit_weibull_from_observations(times, censored)
        result["status"] = "fit_ok"
        result["fitted_median"] = median
        result["fitted_shape"] = shape
        result["log_likelihood"] = ll
    except Exception as e:
        result["status"] = f"fit_failed: {e}"

    return result


def main():
    config = load_buffer_config(CONFIG_PATH)
    REPORT_PATH.parent.mkdir(exist_ok=True)

    lines = [
        "# Sandy 2012 Buffer Distribution Fitting Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Method",
        "",
        "Weibull(median, shape) parameters fit by MLE with right-censoring on Sandy 2012",
        "post-incident outage records. Censored observations represent facilities that did",
        "not fail during the observation window (~120h post-landfall).",
        "",
        "## Per-type results",
        "",
    ]

    fits = []
    for infra_type, csv_path in SANDY_FILES.items():
        obs_df = load_observations(csv_path)
        if obs_df is None:
            lines += [
                f"### {infra_type}",
                f"_No data file at `{csv_path.relative_to(ROOT)}` — skipped._",
                "",
            ]
            continue

        prior = config["defaults"].get(infra_type)
        if prior is None:
            lines += [f"### {infra_type}", "_No prior in config — skipped._", ""]
            continue

        r = fit_one_type(infra_type, obs_df, prior)
        fits.append(r)

        lines += [
            f"### {infra_type}",
            "",
            f"- Observations: {r['n_total']} ({r['n_failed']} failed, {r['n_censored']} censored)",
            f"- Prior: median={r['prior_median']}h, shape={r['prior_shape']}",
        ]
        if r["status"] == "fit_ok":
            lines.append(
                f"- **Fitted: median={r['fitted_median']:.1f}h, shape={r['fitted_shape']:.2f}**"
            )
            lines.append(f"- Log-likelihood: {r['log_likelihood']:.2f}")
            for t in [12, 24, 48, 96]:
                p_alive = weibull_survival(t, r["fitted_median"], r["fitted_shape"])
                lines.append(f"- P(alive at t={t}h) = {p_alive:.2%}")
        else:
            lines.append(f"- Status: {r['status']}")
        lines.append("")

    n_updated = 0
    for r in fits:
        if r["status"] == "fit_ok":
            d = config["defaults"][r["infra_type"]]
            d["median_hours"] = round(r["fitted_median"], 2)
            d["shape"] = round(r["fitted_shape"], 2)
            d["source"] = "sandy_fit"
            d["fit_n_observations"] = r["n_total"]
            d["fit_n_failures"] = r["n_failed"]
            d["fit_date"] = datetime.now().date().isoformat()
            n_updated += 1

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    lines += [
        "## Summary",
        "",
        f"Updated `{CONFIG_PATH.relative_to(ROOT)}` for **{n_updated}** infrastructure types.",
        "Types without sufficient failure data retain their engineering-estimate priors.",
        "",
        "## Caveats",
        "",
        "- Sample sizes are small (NYC has ~50 hospitals; Sandy hit a small subset).",
        "  Confidence intervals on shape are wide. Treat fitted values as informed",
        "  priors, not ground truth.",
        "- Times-to-failure depend on observation framing. Confirm CSV entries against",
        "  primary sources before publication: CDC MMWR 62(02), NYC DOHMH after-action",
        "  reviews, NYU Langone post-Sandy report, Bellevue evacuation records.",
        "- Right-censored observations (pre-evacuated facilities) contribute to the fit",
        "  via the survival function but do not provide failure-time information.",
    ]

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote fit report to {REPORT_PATH}")
    print(f"Updated config at {CONFIG_PATH}")
    if fits:
        print(f"\n{n_updated}/{len(fits)} types fit successfully.")


if __name__ == "__main__":
    main()