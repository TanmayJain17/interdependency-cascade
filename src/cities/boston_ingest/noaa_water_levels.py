"""
NOAA CO-OPS ingestion for the January 2018 nor'easter (Boston, station 8443970).

Pulls real observed and predicted water levels plus barometric pressure for the
2018-01-01 -> 2018-01-07 window around the January 4, 2018 nor'easter, which set
Boston's all-time record water level. This is the first REAL hazard ground truth
for the Boston pilot (prior weeks used synthetic Climate Ready Boston extents).

Products (CO-OPS Data Retrieval API, datagetter):
    water_level   6-min verified observed levels   datum required
    predictions   6-min astronomical tide          same datum as observed
    air_pressure  barometric pressure              no datum
    wind          speed/direction/gusts            NOT offered at 8443970 (skipped)

Datum: the repo's DEM / Climate Ready Boston layers are NAVD 88, so observed and
predicted are pulled in NAVD directly (verified available for this station-product),
avoiding a hand-applied MLLW->NAVD88 conversion. The full published datum table is
still recorded in datum_reference.json for downstream provenance, and the observed
peak is cross-checked against the known record (15.16 ft MLLW / 9.66 ft NAVD 88).

Surge is derived as observed - predicted on the shared 6-min timestamps; because it
is a difference, surge is datum-independent.

Outputs (data/boston/raw/noaa/):
    boston_2018_water_levels.csv   timestamp, observed_m, predicted_m, surge_m, datum
    boston_2018_met.csv            timestamp, air_pressure_mb (+ wind if available)
    datum_reference.json           every datum offset used downstream
    boston_2018_water_levels.png   QC plot (observed vs predicted vs surge)

No external endpoint is assumed blindly: the live JSON shape was inspected before
this parser was written (observed/pressure under `data`, predictions under
`predictions`; NOAA error payloads arrive as {"error": {...}}).
"""
from __future__ import annotations

import datetime as dt
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from src.cities import boston  # noqa: E402

log = logging.getLogger(__name__)

LAYER = "noaa"
OUT_DIR = boston.RAW_DIR / "noaa"

STATION = "8443970"
BEGIN = "20180101"
END = "20180107"
DATAGETTER = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
DATUMS_URL = (
    f"https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations/{STATION}/datums.json"
)
APPLICATION = "cera-boston-pilot"
PREFERRED_DATUM = "NAVD"  # matches the repo's DEM / CRB layers; MLLW is the fallback

# Published record for the Jan 4, 2018 peak (NOAA / USGS), for a sanity cross-check.
RECORD_MLLW_FT = 15.16
RECORD_NAVD88_FT = 9.66
M_PER_FT = 0.3048

TIMEOUT = 60


# ---------------------------------------------------------------------------
# Low-level API access
# ---------------------------------------------------------------------------
def _datagetter(product: str, *, datum: str | None = None, interval: str | None = None) -> dict:
    """One datagetter call. Returns the parsed JSON (may carry an `error` key)."""
    params = {
        "station": STATION,
        "begin_date": BEGIN,
        "end_date": END,
        "time_zone": "GMT",
        "units": "metric",
        "format": "json",
        "application": APPLICATION,
        "product": product,
    }
    if datum is not None:
        params["datum"] = datum
    if interval is not None:
        params["interval"] = interval
    r = requests.get(DATAGETTER, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _to_frame(payload: dict, key: str, product: str) -> pd.DataFrame:
    """Pull the row list out of a datagetter payload (`data` or `predictions`)."""
    if "error" in payload:
        raise RuntimeError(f"NOAA {product}: {payload['error'].get('message', payload['error'])}")
    rows = payload.get(key)
    if not rows:
        raise RuntimeError(f"NOAA {product}: empty `{key}` in response.")
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Datums
# ---------------------------------------------------------------------------
def fetch_datums(force: bool = False) -> dict:
    """Pull the published station datum table and persist datum_reference.json."""
    out = OUT_DIR / "datum_reference.json"
    r = requests.get(DATUMS_URL, params={"units": "metric"}, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()
    offsets = {d["name"]: float(d["value"]) for d in j.get("datums", []) if d.get("value") is not None}
    log.info("Station datums (m above station datum): %s",
             {k: offsets[k] for k in ("STND", "MLLW", "MSL", "MHHW", "NAVD88") if k in offsets})

    navd88 = offsets.get("NAVD88")
    mllw = offsets.get("MLLW")
    navd88_minus_mllw = (navd88 - mllw) if (navd88 is not None and mllw is not None) else None

    ref = {
        "station": STATION,
        "station_name": j.get("self", ""),
        "units": "metric (meters)",
        "tidal_epoch": j.get("epoch"),
        "orthometric_datum": j.get("OrthometricDatum"),
        "offsets_above_station_datum_m": offsets,
        "navd88_minus_mllw_m": navd88_minus_mllw,
        "conversion_note": (
            "level_NAVD88 = level_MLLW - (NAVD88 - MLLW). Observed/predicted in this "
            "ingestion are pulled directly in NAVD, so no conversion is applied to the "
            "saved series; this table documents the published offsets for provenance."
        ),
        "observed_datum_pulled": None,  # filled in by download()
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(ref, indent=2))
    log.info("NAVD88 - MLLW = %.3f m  -> %s", navd88_minus_mllw, out)
    return ref


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------
def fetch_water_levels() -> tuple[pd.DataFrame, str]:
    """Observed 6-min water level. Tries NAVD, falls back to MLLW. Returns (df, datum)."""
    for datum in (PREFERRED_DATUM, "MLLW"):
        payload = _datagetter("water_level", datum=datum)
        if "error" not in payload:
            df = _to_frame(payload, "data", "water_level")
            log.info("water_level datum=%s: %d rows; first=%s", datum, len(df),
                     payload["data"][0])
            return df.rename(columns={"v": "observed_m"})[["t", "observed_m"]], datum
        log.warning("water_level datum=%s unavailable: %s", datum,
                    payload["error"].get("message"))
    raise RuntimeError("water_level: neither NAVD nor MLLW returned data.")


def fetch_predictions(datum: str) -> pd.DataFrame:
    payload = _datagetter("predictions", datum=datum, interval="6")
    df = _to_frame(payload, "predictions", "predictions")
    log.info("predictions datum=%s: %d rows; first=%s", datum, len(df),
             payload["predictions"][0])
    return df.rename(columns={"v": "predicted_m"})[["t", "predicted_m"]]


def fetch_met() -> pd.DataFrame:
    """Barometric pressure (always) and wind (if the station offers it)."""
    frames = []
    payload = _datagetter("air_pressure")
    df = _to_frame(payload, "data", "air_pressure")
    log.info("air_pressure: %d rows; first=%s", len(df), payload["data"][0])
    frames.append(df.rename(columns={"v": "air_pressure_mb"})[["t", "air_pressure_mb"]])

    payload = _datagetter("wind")
    if "error" in payload:
        log.warning("wind not offered at %s: %s — met output is pressure-only",
                    STATION, payload["error"].get("message"))
    else:
        w = pd.DataFrame(payload["data"])
        w["t"] = pd.to_datetime(w["t"], utc=True)
        for c in ("s", "g", "d"):
            if c in w.columns:
                w[c] = pd.to_numeric(w[c], errors="coerce")
        w = w.rename(columns={"s": "wind_speed_ms", "g": "wind_gust_ms", "d": "wind_dir_deg"})
        keep = [c for c in ("t", "wind_speed_ms", "wind_gust_ms", "wind_dir_deg") if c in w.columns]
        log.info("wind: %d rows", len(w))
        frames.append(w[keep])

    met = frames[0]
    for extra in frames[1:]:
        met = met.merge(extra, on="t", how="outer")
    return met.sort_values("t").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------
def derive_surge(obs: pd.DataFrame, pred: pd.DataFrame, datum: str) -> pd.DataFrame:
    """Inner-join observed/predicted on the 6-min stamps; surge = observed - predicted."""
    df = obs.merge(pred, on="t", how="inner").sort_values("t").reset_index(drop=True)
    df["surge_m"] = df["observed_m"] - df["predicted_m"]
    df["datum"] = datum
    df = df.rename(columns={"t": "timestamp"})
    return df[["timestamp", "observed_m", "predicted_m", "surge_m", "datum"]]


def summarize(wl: pd.DataFrame, datum: str, ref: dict) -> dict:
    """Peak observed, peak surge, astro-high lag, and the record cross-check."""
    obs_peak = wl.loc[wl["observed_m"].idxmax()]
    surge_peak = wl.loc[wl["surge_m"].idxmax()]

    # Astronomical high tide nearest the observed peak (+/- 12 h window).
    win = wl[(wl["timestamp"] >= obs_peak["timestamp"] - pd.Timedelta(hours=12)) &
             (wl["timestamp"] <= obs_peak["timestamp"] + pd.Timedelta(hours=12))]
    astro_high = win.loc[win["predicted_m"].idxmax()]
    lag_hr = (surge_peak["timestamp"] - astro_high["timestamp"]).total_seconds() / 3600.0

    # Cross-check against the published record. Convert the observed NAVD peak to ft,
    # and the published MLLW record to NAVD using the station offset.
    navd_minus_mllw = ref.get("navd88_minus_mllw_m")
    record_navd_from_mllw_m = RECORD_MLLW_FT * M_PER_FT - (navd_minus_mllw or 0.0)
    obs_peak_ft = obs_peak["observed_m"] / M_PER_FT if datum == "NAVD" else None

    summary = {
        "datum": datum,
        "n_obs": int(len(wl)),
        "window": [BEGIN, END],
        "peak_observed": {
            "value_m": round(float(obs_peak["observed_m"]), 4),
            "value_ft": round(float(obs_peak_ft), 3) if obs_peak_ft is not None else None,
            "timestamp_utc": obs_peak["timestamp"].isoformat(),
        },
        "peak_surge": {
            "value_m": round(float(surge_peak["surge_m"]), 4),
            "timestamp_utc": surge_peak["timestamp"].isoformat(),
        },
        "astro_high_near_peak": {
            "value_m": round(float(astro_high["predicted_m"]), 4),
            "timestamp_utc": astro_high["timestamp"].isoformat(),
        },
        "surge_lag_after_astro_high_hr": round(float(lag_hr), 2),
        "record_crosscheck": {
            "published_mllw_ft": RECORD_MLLW_FT,
            "published_navd88_ft": RECORD_NAVD88_FT,
            "published_navd88_from_mllw_m": round(record_navd_from_mllw_m, 4),
            "observed_peak_navd88_m": round(float(obs_peak["observed_m"]), 4) if datum == "NAVD" else None,
        },
    }
    log.info("Peak observed %.3f m %s @ %s", summary["peak_observed"]["value_m"], datum,
             summary["peak_observed"]["timestamp_utc"])
    log.info("Peak surge %.3f m @ %s (lag %.2f h after astro high)",
             summary["peak_surge"]["value_m"], summary["peak_surge"]["timestamp_utc"], lag_hr)
    log.info("Cross-check: observed peak %.3f m NAVD88 vs published %.3f m NAVD88 (from %.2f ft MLLW)",
             summary["record_crosscheck"]["observed_peak_navd88_m"] or float("nan"),
             record_navd_from_mllw_m, RECORD_MLLW_FT)
    return summary


def _qc_plot(wl: pd.DataFrame, summary: dict, out: "Path") -> None:  # type: ignore[name-defined]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(wl["timestamp"], wl["observed_m"], color="#1f5fa8", lw=1.0, label="observed")
    ax.plot(wl["timestamp"], wl["predicted_m"], color="#888888", lw=0.9, ls="--", label="predicted (astro)")
    ax.plot(wl["timestamp"], wl["surge_m"], color="#d1495b", lw=1.0, label="surge (obs - pred)")
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    pk = pd.Timestamp(summary["peak_observed"]["timestamp_utc"])
    ax.axvline(pk, color="#1f5fa8", lw=0.8, ls=":", alpha=0.7)
    ax.annotate(f"peak {summary['peak_observed']['value_m']:.2f} m NAVD88",
                xy=(pk, summary["peak_observed"]["value_m"]),
                xytext=(8, 6), textcoords="offset points", fontsize=9, color="#1f5fa8")
    ax.set_title("Boston (8443970) — Jan 2018 nor'easter water levels (NAVD 88)")
    ax.set_xlabel("UTC")
    ax.set_ylabel("water level / surge (m)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    log.info("Saved QC plot -> %s", out)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def download(force: bool = False) -> list[dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wl_path = OUT_DIR / "boston_2018_water_levels.csv"
    met_path = OUT_DIR / "boston_2018_met.csv"
    png_path = OUT_DIR / "boston_2018_water_levels.png"
    ref_path = OUT_DIR / "datum_reference.json"
    summary_path = OUT_DIR / "boston_2018_summary.json"

    if wl_path.exists() and not force:
        log.info("%s present — skipping NOAA pull (use force=True to refresh)", wl_path.name)
        wl = pd.read_csv(wl_path, parse_dates=["timestamp"])
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    else:
        ref = fetch_datums(force=force)
        obs, datum = fetch_water_levels()
        pred = fetch_predictions(datum)
        wl = derive_surge(obs, pred, datum)
        met = fetch_met()

        # Backfill the datum actually used into datum_reference.json.
        ref["observed_datum_pulled"] = datum
        ref_path.write_text(json.dumps(ref, indent=2))

        summary = summarize(wl, datum, ref)
        wl.to_csv(wl_path, index=False)
        met.to_csv(met_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2))
        _qc_plot(wl, summary, png_path)
        log.info("Saved %d water-level rows -> %s", len(wl), wl_path)
        log.info("Saved %d met rows -> %s", len(met), met_path)

    return [{
        "layer": LAYER,
        "path": str(wl_path),
        "source_url": DATAGETTER,
        "record_count": int(len(wl)),
        "sub_layer": "water_levels",
        "station": STATION,
        "summary": summary,
    }]


if __name__ == "__main__":
    from src.cities.boston_ingest._logging import configure as configure_logging

    configure_logging()
    entries = download()
    for e in entries:
        print(f"  {e['layer']:8s} [{e['sub_layer']}]  {e['path']}  ({e['record_count']} rows)")
        s = e.get("summary", {})
        if s:
            print(f"    peak observed : {s['peak_observed']['value_m']:.3f} m NAVD88 "
                  f"({s['peak_observed']['value_ft']:.2f} ft) @ {s['peak_observed']['timestamp_utc']}")
            print(f"    peak surge    : {s['peak_surge']['value_m']:.3f} m @ {s['peak_surge']['timestamp_utc']}")
            rc = s["record_crosscheck"]
            print(f"    record check  : observed {rc['observed_peak_navd88_m']:.3f} m NAVD88 vs "
                  f"published {rc['published_navd88_from_mllw_m']:.3f} m NAVD88 "
                  f"({rc['published_mllw_ft']} ft MLLW / {rc['published_navd88_ft']} ft NAVD88)")
