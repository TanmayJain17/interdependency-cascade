"""
STAGED / EXPERIMENTAL — EAGLE-I county power-outage timing overlay (Suffolk County, Jan 2018).

This is NOT a headline result and must not be presented as validated accuracy. It is a
directional, aggregate timing comparison only, with two structural confounds:
  1. EAGLE-I is COUNTY-level (Suffolk = FIPS 25025) — it cannot validate node-level cascade
     predictions; it can only be compared to the model's aggregate power-failure curve.
  2. The January 2018 Boston outages were partly WIND/SNOW-driven, not purely coastal flood,
     so the real curve mixes hazards the flood model does not represent.

Data: ORNL EAGLE-I historic dataset (county-level, 15-min, 2014-2022), Figshare article
24237376, file eaglei_outages_2018.csv (~1 GB national). Rather than store ~1 GB for one
county-week, we STREAM the remote CSV and keep only rows for FIPS 25025 (fast first-field
prefix filter), restricted to the event window. If the network path fails, a user-provided
full-year file dropped at data/boston/raw/eagle_i/eaglei_outages_2018.csv is filtered instead.

Schema: fips_code, county, state, customers_out, run_start_time (UTC-naive 15-min stamps).

Outputs (data/boston/validation/staged/):
    suffolk_2018_outages.csv                 filtered Suffolk County customers-out series
    eagle_i_vs_model_power_timing.png        outage curve vs model aggregate power-failure curve
    README.md                                the confound caveat
"""
from __future__ import annotations

import io
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from src.cities import boston  # noqa: E402

log = logging.getLogger(__name__)

LAYER = "eagle_i"
RAW_DIR = boston.RAW_DIR / "eagle_i"
OUT_DIR = boston.DATA_DIR / "validation" / "staged"

FIGSHARE_2018 = "https://ndownloader.figshare.com/files/42547879"
SUFFOLK_FIPS = "25025"
WINDOW = ("2018-01-01", "2018-01-08")  # inclusive of Jan 1-7, +1 day margin
# EAGLE-I run_start_time stamps are UTC, so we anchor the model t=0 to the storm peak IN UTC:
# 2018-01-04 17:42 UTC (= 12:42 EST), which is also the NOAA Phase-1 observed-peak timestamp.
# Keeping everything on one clock (UTC) avoids a 5-hour offset on the overlay.
STORM_PEAK = pd.Timestamp("2018-01-04 17:42")
TIMEOUT = 1200


def _stream_filter() -> pd.DataFrame:
    log.info("Streaming EAGLE-I 2018 and keeping FIPS %s only (no full-file storage)...", SUFFOLK_FIPS)
    prefix = f"{SUFFOLK_FIPS},"
    kept = []
    header = "fips_code,county,state,customers_out,run_start_time"
    with requests.get(FIGSHARE_2018, stream=True, timeout=TIMEOUT) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line and line.startswith(prefix):
                kept.append(line)
    log.info("Kept %d Suffolk County rows", len(kept))
    df = pd.read_csv(io.StringIO(header + "\n" + "\n".join(kept)))
    return df


def _local_filter(path) -> pd.DataFrame:
    log.info("Filtering local EAGLE-I file %s to FIPS %s", path.name, SUFFOLK_FIPS)
    chunks = []
    for ch in pd.read_csv(path, dtype={"fips_code": str}, chunksize=500_000):
        chunks.append(ch[ch["fips_code"] == SUFFOLK_FIPS])
    return pd.concat(chunks, ignore_index=True)


def acquire(force: bool = False) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "suffolk_2018_outages.csv"
    if out.exists() and not force:
        log.info("%s present — skipping acquisition", out.name)
        return pd.read_csv(out, parse_dates=["run_start_time"])

    local = RAW_DIR / "eaglei_outages_2018.csv"
    try:
        df = _stream_filter()
    except Exception as exc:  # noqa: BLE001
        log.warning("Stream acquisition failed (%s)", exc)
        if local.exists():
            df = _local_filter(local)
        else:
            raise RuntimeError(
                "EAGLE-I stream failed and no local fallback. Download eaglei_outages_2018.csv "
                f"from Figshare article 24237376 and place it at {local}, then re-run."
            ) from exc

    df["run_start_time"] = pd.to_datetime(df["run_start_time"])
    df = df[(df["run_start_time"] >= WINDOW[0]) & (df["run_start_time"] < WINDOW[1])]
    df = df.sort_values("run_start_time").reset_index(drop=True)
    df.to_csv(out, index=False)
    log.info("Saved %d Suffolk rows (%s..%s) -> %s", len(df), WINDOW[0], WINDOW[1], out)
    return df


def _model_power_curve() -> pd.DataFrame:
    """Aggregate expected power-node failures over t=6..96 h, anchored to the storm peak."""
    per = json.load(open(boston.SIMULATION_DIR / "zeroshot_per_node.json"))
    ts = per["timesteps"]  # [6, 24, 48, 96]
    sp = per["scenarios"]["slr09_aep01"]  # near-term matched scenario
    power = {k: v for k, v in sp.items() if k.startswith("power_")}
    # Expected failed power nodes = sum of per-node failure probabilities at each model timestep.
    # (No synthetic t=0 point: rec = [depth, p6, p24, p48, p96] has no t=0 probability, and mixing
    # a hard node count at t=0 with expectations at t>0 would create a spurious dip.)
    rows = [{"hours": t, "clock": STORM_PEAK + pd.Timedelta(hours=t),
             "exp_power_failures": sum(v[1 + i] for v in power.values())}
            for i, t in enumerate(ts)]
    return pd.DataFrame(rows)


def _overlay_plot(outages: pd.DataFrame, model: pd.DataFrame, out) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    ax1.plot(outages["run_start_time"], outages["customers_out"], color="#444444", lw=1.2,
             label="Suffolk County customers out (EAGLE-I, real)")
    ax1.set_ylabel("customers without power (county)", color="#444444")
    ax1.set_xlabel("Jan 2018 (UTC-naive)")
    ax1.axvline(STORM_PEAK, color="#1f5fa8", ls=":", lw=1.2)
    ax1.annotate("storm peak\n(record water level)", xy=(STORM_PEAK, ax1.get_ylim()[1] * 0.9),
                 fontsize=8, color="#1f5fa8", ha="left")

    ax2 = ax1.twinx()
    ax2.plot(model["clock"], model["exp_power_failures"], "o-", color="#d1495b", lw=1.6,
             label="model expected power-node failures (slr09_aep01)")
    ax2.set_ylabel("model expected power-node failures", color="#d1495b")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d %H:%M"))
    fig.autofmt_xdate()
    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=8)
    ax1.set_title("STAGED / DIRECTIONAL — Suffolk County outages vs model power-failure timing, Jan 2018\n"
                  "(county-level; partly wind/snow-driven; NOT a node-level accuracy claim)", fontsize=11)
    ax1.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    log.info("Saved staged overlay -> %s", out)


def _readme(outages: pd.DataFrame, out) -> None:
    peak = outages.loc[outages["customers_out"].idxmax()] if len(outages) else None
    out.write_text(
        "# STAGED / EXPERIMENTAL — EAGLE-I power-outage timing overlay\n\n"
        "**This is not a headline result and must not be presented as validated accuracy.**\n"
        "It is a directional, aggregate timing comparison with structural confounds:\n\n"
        "- **County-level, not node-level.** EAGLE-I reports Suffolk County (FIPS 25025) "
        "customers-without-power; it cannot validate which infrastructure NODES failed. It is "
        "only comparable to the model's *aggregate* power-failure curve.\n"
        "- **Mixed hazard.** January 2018 Boston outages were partly wind/snow-driven, not purely "
        "coastal flood. The flood model represents only the flood driver, so the real curve "
        "contains outages the model cannot (and should not) reproduce.\n"
        "- **Clock anchoring.** The model has no absolute clock; t=0 is anchored to the storm "
        "peak **2018-01-04 17:42 UTC** (= 12:42 EST, the NOAA Phase-1 observed-peak timestamp) and "
        "the model horizon (t=6..96 h) is laid on that. EAGLE-I run_start_time stamps are UTC, so "
        "both curves share one clock.\n\n"
        f"Real Suffolk peak: {int(peak['customers_out']) if peak is not None else 'n/a'} customers out "
        f"@ {peak['run_start_time'] if peak is not None else 'n/a'} UTC.\n\n"
        "**Key confound made visible:** the real Suffolk outage peak lands on **Jan 7**, ~3 days "
        "AFTER the Jan-4 flood/storm peak. That lag is the clearest evidence the January 2018 Boston "
        "outages were largely NOT flood-driven (wind/snow/cold), so the model's flood-driven power "
        "ramp should NOT be expected to reproduce it. Use only as a sanity check that the model's "
        "ramp is not grossly mistimed relative to the event window. Real node-level outage validation "
        "requires utility circuit/feeder records, which we do not have.\n"
    )
    log.info("Wrote staged README -> %s", out)


def download(force: bool = False) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outages = acquire(force)
    model = _model_power_curve()
    _overlay_plot(outages, model, OUT_DIR / "eagle_i_vs_model_power_timing.png")
    _readme(outages, OUT_DIR / "README.md")
    outages.to_csv(OUT_DIR / "suffolk_2018_outages.csv", index=False)
    peak = outages.loc[outages["customers_out"].idxmax()] if len(outages) else None
    return [{
        "layer": LAYER, "sub_layer": "suffolk_outages_staged",
        "path": str(OUT_DIR / "eagle_i_vs_model_power_timing.png"),
        "record_count": int(len(outages)),
        "peak_customers_out": int(peak["customers_out"]) if peak is not None else None,
        "peak_time": str(peak["run_start_time"]) if peak is not None else None,
    }]


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    for e in download():
        print(f"  STAGED {e['sub_layer']}: {e['record_count']} rows, "
              f"peak {e['peak_customers_out']} customers out @ {e['peak_time']}")
        print(f"    -> {e['path']}")
