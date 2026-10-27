#!/usr/bin/env python3
"""
windpower.py - Wind resource assessment and turbine energy yield.

Inputs:
  1) Hourly CSV with wind speed (m/s), temperature (°C) and wind direction (°).
  2) CSV with the turbine power curve (wind speed m/s, power kW).

Outputs (folder --output):
  - summary.json / report.md : numerical results
  - weibull.png, power_curve.png, wind_rose.png, monthly_energy.png
  - monthly_energy.csv, wind_rose.csv

Usage:
  python windpower.py data.csv curve.csv --output results \
      --measure-height 80 --hub-height 100 --altitude 50
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
RHO_0 = 1.225        # kg/m3, standard air density (15 °C, sea level)
R_AIR = 287.05       # J/(kg K), dry air
HOURS_PER_YEAR = 8760

# Accepted column names (lowercase, no accents, spaces -> underscores).
# Both English and Spanish headers are recognised.
ALIASES = {
    "time": ["date", "datetime", "date_time", "timestamp", "time", "fecha", "fecha_hora", "tiempo", "hora"],
    "v": ["wind_speed", "wind_speed_ms", "wind_speed_m/s", "windspeed", "speed", "ws", "v", "wind",
          "velocidad", "velocidad_ms", "velocidad_m/s", "velocidad_viento", "viento", "vel"],
    "t": ["temperature", "temperature_c", "temp", "t", "ta", "air_temperature",
          "temperatura", "temperatura_c", "t_aire"],
    "dir": ["wind_direction", "wind_dir", "wind_dir_deg", "direction", "dir", "wd", "winddirection",
            "direccion", "direccion_deg", "direccion_viento", "dir_viento"],
}
POWER_CURVE_V = ["wind_speed", "wind_speed_ms", "speed", "v", "ws", "velocidad", "viento"]
POWER_CURVE_P = ["power", "power_kw", "p", "kw", "potencia", "potencia_kw"]


def _normalize(s: str) -> str:
    s = str(s).strip().lower()
    for a, b in zip("áéíóúñ", "aeioun"):
        s = s.replace(a, b)
    return s.replace(" ", "_")


# ----------------------------------------------------------------------------
# Reading and cleaning
# ----------------------------------------------------------------------------
def _read_csv(path, sep=None, needs_speed=False, col_v=None, min_cols=1) -> pd.DataFrame:
    """Read a CSV and detect its separator. If automatic detection gives an unusable
    result (no speed column, or too few columns), try , ; tab and |."""
    def attempt(s):
        if hasattr(path, "seek"):
            path.seek(0)
        return pd.read_csv(path, sep=s, engine="python", encoding_errors="replace")

    def valid(df):
        if df.shape[1] < min_cols:
            return False
        if needs_speed:
            names = {_normalize(c) for c in df.columns}
            return (col_v in df.columns) if col_v else bool(names & set(ALIASES["v"]))
        return True

    candidates = [sep] if sep else [None, ",", ";", "\t", "|"]
    first = None
    for s in candidates:
        try:
            df = attempt(s)
        except Exception:
            continue
        if first is None:
            first = df
        if valid(df):
            return df
    if first is None:
        raise ValueError("Could not read the file as CSV.")
    return first


def read_data(path, col_v=None, col_t=None, col_dir=None, col_time=None, sep=None) -> pd.DataFrame:
    """Read the weather CSV. Returns a DataFrame with columns v, t, dir (time index if present)."""
    df = _read_csv(path, sep, needs_speed=True, col_v=col_v)
    col_map = {_normalize(c): c for c in df.columns}

    def find(key, explicit):
        if explicit:
            if explicit not in df.columns:
                raise ValueError(f"Column '{explicit}' does not exist. Columns: {list(df.columns)}")
            return explicit
        for alias in ALIASES[key]:
            if alias in col_map:
                return col_map[alias]
        return None

    c_v = find("v", col_v)
    c_t = find("t", col_t)
    c_d = find("dir", col_dir)
    c_f = find("time", col_time)
    if c_v is None:
        raise ValueError(f"Wind speed column not found. Columns: {list(df.columns)}. "
                         "Use --col-speed to name it.")

    out = pd.DataFrame()
    out["v"] = pd.to_numeric(df[c_v], errors="coerce")
    out["t"] = pd.to_numeric(df[c_t], errors="coerce") if c_t else np.nan
    out["dir"] = pd.to_numeric(df[c_d], errors="coerce") if c_d else np.nan
    if c_f:
        ts = pd.to_datetime(df[c_f], errors="coerce", dayfirst=False)
        if ts.notna().mean() > 0.9:
            out.index = ts
            out.index.name = "time"
    return out


def clean(df: pd.DataFrame):
    """Remove impossible values. Returns (clean_df, info)."""
    n0 = len(df)
    bad_v = df["v"].isna() | (df["v"] < 0) | (df["v"] > 80)
    d = df.copy()
    d.loc[(d["t"] < -60) | (d["t"] > 60), "t"] = np.nan
    d.loc[(d["dir"] < 0) | (d["dir"] > 360), "dir"] = np.nan
    d = d[~bad_v]
    info = {"total_records": int(n0), "valid_records": int(len(d)),
            "discarded": int(n0 - len(d)),
            "availability_pct": round(100 * len(d) / n0, 2) if n0 else 0.0}
    return d, info


# ----------------------------------------------------------------------------
# Height extrapolation (power law)
# ----------------------------------------------------------------------------
def extrapolate_height(v, h_meas, h_hub, alpha=0.143):
    """Power law: v_h = v_ref * (h / h_ref)^alpha. alpha = 1/7 is typical."""
    return v * (h_hub / h_meas) ** alpha


# ----------------------------------------------------------------------------
# Air density
# ----------------------------------------------------------------------------
def air_density(temp_c, altitude_m=0.0, pressure_pa=None):
    """Density (kg/m3). Pressure from the standard atmosphere if not given."""
    if pressure_pa is None:
        pressure_pa = 101325.0 * (1 - 2.25577e-5 * altitude_m) ** 5.25588
    t_k = np.asarray(temp_c, dtype=float) + 273.15
    return pressure_pa / (R_AIR * t_k)


# ----------------------------------------------------------------------------
# Weibull
# ----------------------------------------------------------------------------
def weibull_mle(v):
    """Maximum likelihood on v>0. Returns (k, c)."""
    x = np.asarray(v, dtype=float)
    x = x[x > 0]
    ln = np.log(x)

    def f(k):
        xk = x ** k
        return (xk * ln).sum() / xk.sum() - 1.0 / k - ln.mean()

    k = optimize.brentq(f, 0.2, 20.0)
    c = ((x ** k).mean()) ** (1.0 / k)
    return float(k), float(c)


def weibull_moments(v):
    """Justus empirical method: k=(σ/μ)^-1.086, c=μ/Γ(1+1/k)."""
    x = np.asarray(v, dtype=float)
    mu, sd = x.mean(), x.std(ddof=1)
    k = (sd / mu) ** -1.086
    c = mu / special.gamma(1 + 1 / k)
    return float(k), float(c)


def weibull_energy_pattern(v):
    """Energy pattern factor method (Akdağ and Dinler)."""
    x = np.asarray(v, dtype=float)
    mu = x.mean()
    epf = (x ** 3).mean() / mu ** 3
    k = 1 + 3.69 / epf ** 2
    c = mu / special.gamma(1 + 1 / k)
    return float(k), float(c)


def weibull_pdf(v, k, c):
    v = np.asarray(v, dtype=float)
    return (k / c) * (v / c) ** (k - 1) * np.exp(-((v / c) ** k))


def weibull_cdf(v, k, c):
    return 1 - np.exp(-((np.asarray(v, dtype=float) / c) ** k))


def goodness_of_fit(v, k, c):
    """R², RMSE (histogram vs pdf) and Kolmogorov-Smirnov statistic."""
    x = np.asarray(v, dtype=float)
    x = x[x > 0]
    bins = np.arange(0, np.ceil(x.max()) + 1, 1.0)
    h, edges = np.histogram(x, bins=bins, density=True)
    mid = 0.5 * (edges[:-1] + edges[1:])
    p = weibull_pdf(mid, k, c)
    ss_res = ((h - p) ** 2).sum()
    ss_tot = ((h - h.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = math.sqrt(((h - p) ** 2).mean())
    ks = stats.kstest(x, lambda z: weibull_cdf(z, k, c))
    return {"R2": round(float(r2), 4), "RMSE": round(float(rmse), 5),
            "KS_statistic": round(float(ks.statistic), 4)}


# ----------------------------------------------------------------------------
# Wind resource
# ----------------------------------------------------------------------------
def power_density(v, rho):
    """Mean wind power density (W/m2) = 0.5 * rho * <v^3>."""
    v = np.asarray(v, dtype=float)
    rho = np.asarray(rho, dtype=float)
    return float((0.5 * rho * v ** 3).mean())


def wind_class(pd_w_m2):
    """Approximate NREL wind power class (defined at 50 m); indicative only."""
    limits = [(100, "1 (poor)"), (150, "2 (marginal)"), (200, "3 (fair)"),
              (250, "4 (good)"), (300, "5 (excellent)"), (400, "6 (outstanding)")]
    for lim, name in limits:
        if pd_w_m2 < lim:
            return name
    return "7 (superb)"


# ----------------------------------------------------------------------------
# Power curve
# ----------------------------------------------------------------------------
class PowerCurve:
    def __init__(self, v, p_kw, rho_ref=RHO_0, name="turbine"):
        o = np.argsort(v)
        self.v = np.asarray(v, dtype=float)[o]
        self.p = np.asarray(p_kw, dtype=float)[o]
        self.rho_ref = rho_ref
        self.name = name
        self.rated_power = float(self.p.max())
        # cut-in: first speed with P>0 ; cut-out: last speed with P>0
        pos = self.v[self.p > 0]
        self.cut_in = float(pos.min()) if len(pos) else float("nan")
        self.cut_out = float(pos.max()) if len(pos) else float("nan")
        # rated speed: first speed with P >= 0.99 * rated
        self.rated_speed = float(self.v[self.p >= 0.99 * self.rated_power].min())

    @classmethod
    def from_csv(cls, path, rho_ref=RHO_0, col_v=None, col_p=None, unit="kW"):
        df = _read_csv(path, None, min_cols=2)
        cols = {_normalize(c): c for c in df.columns}
        cv = col_v or next((cols[a] for a in POWER_CURVE_V if a in cols), df.columns[0])
        cp = col_p or next((cols[a] for a in POWER_CURVE_P if a in cols), df.columns[1])
        v = pd.to_numeric(df[cv], errors="coerce")
        p = pd.to_numeric(df[cp], errors="coerce")
        ok = v.notna() & p.notna()
        p = p[ok].values
        if unit.upper() == "W":
            p = p / 1000.0
        elif unit.upper() == "MW":
            p = p * 1000.0
        name = Path(getattr(path, "name", None) or str(path)).stem
        return cls(v[ok].values, p, rho_ref=rho_ref, name=name)

    def power(self, v, rho=None):
        """Power (kW) by linear interpolation; 0 outside [cut-in, cut-out].
        If rho is given, speed is corrected: v_eq = v*(rho/rho_ref)^(1/3) (IEC 61400-12-1)."""
        v = np.asarray(v, dtype=float)
        if rho is not None:
            v = v * (np.asarray(rho, dtype=float) / self.rho_ref) ** (1.0 / 3.0)
        return np.interp(v, self.v, self.p, left=0.0, right=0.0)


# ----------------------------------------------------------------------------
# Energy
# ----------------------------------------------------------------------------
def energy_series(curve: PowerCurve, v, rho, dt_hours=1.0):
    """Energy from the time series. Returns power series (kW) and total energy (kWh)."""
    p = curve.power(v, rho)
    return p, float(p.sum() * dt_hours)


def energy_weibull(curve: PowerCurve, k, c, mean_rho, hours=HOURS_PER_YEAR, step=0.1):
    """Expected annual energy by integrating P(v)·f(v) with the fitted Weibull."""
    vv = np.arange(0, 40 + step, step)
    f = weibull_pdf(vv, k, c)
    f[0] = 0.0 if k > 1 else f[0]
    p = curve.power(vv, mean_rho)
    mean_p = float(integrate.trapezoid(p * f, vv))
    return mean_p, mean_p * hours


def wind_rose(dirs, v, n_sectors=16):
    """Frequency, mean speed and energy share by sector. Returns a DataFrame."""
    d = pd.DataFrame({"dir": dirs, "v": v}).dropna()
    if d.empty:
        return None
    width = 360 / n_sectors
    sec = (((d["dir"] + width / 2) % 360) // width).astype(int)
    g = d.groupby(sec)
    r = pd.DataFrame({"frequency_pct": 100 * g.size() / len(d),
                      "mean_speed": g["v"].mean(),
                      "energy_share_pct": 100 * g["v"].apply(lambda x: (x ** 3).sum()) / (d["v"] ** 3).sum()})
    r = r.reindex(range(n_sectors)).fillna(0.0)
    r.index = [f"{i * width:.1f}" for i in range(n_sectors)]
    r.index.name = "sector_deg"
    return r


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_weibull(v, fits, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = v[v > 0]
    ax.hist(x, bins=np.arange(0, x.max() + 1, 1), density=True, alpha=0.45,
            color="#7aa6c2", edgecolor="white", label="Data")
    vv = np.linspace(0.01, x.max() + 1, 400)
    for name, (k, c) in fits.items():
        ax.plot(vv, weibull_pdf(vv, k, c), lw=2, label=f"{name}: k={k:.2f}, c={c:.2f}")
    ax.set_xlabel("Wind speed (m/s)")
    ax.set_ylabel("Probability density")
    ax.set_title("Weibull distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_curve(curve, k, c, rho, path):
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    vv = np.linspace(0, 30, 600)
    ax1.plot(vv, curve.power(vv), color="#c0392b", lw=2, label="Curve (reference rho)")
    ax1.plot(vv, curve.power(vv, rho), color="#e67e22", lw=2, ls="--",
             label=f"Corrected to rho={rho:.3f} kg/m³")
    ax1.set_xlabel("Wind speed (m/s)")
    ax1.set_ylabel("Power (kW)")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.fill_between(vv, weibull_pdf(vv, k, c), alpha=0.2, color="#2980b9", label="Weibull")
    ax2.set_ylabel("Probability density")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    ax1.set_title(f"Power curve vs wind distribution – {curve.name}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_rose(rose, path):
    n = len(rose)
    th = np.deg2rad(np.arange(n) * 360 / n)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(th, rose["frequency_pct"], width=2 * np.pi / n * 0.9, color="#16a085", alpha=0.85,
           edgecolor="white")
    ax.set_title("Wind rose – frequency (%)", pad=20)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_monthly(monthly, path):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(monthly.index.astype(str), monthly["energy_MWh"], color="#8e44ad", alpha=0.85)
    ax.set_ylabel("Energy (MWh)")
    ax.set_xlabel("Month")
    ax.set_title("Estimated monthly energy")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------------
METHOD_NAMES = {"mle": "MLE", "moments": "Moments", "energy": "Energy pattern"}


def compute(raw: pd.DataFrame, curve: "PowerCurve", args):
    """Calculation core, writes no files (used by the CLI and the dashboard).

    `raw`: DataFrame from read_data(). `args`: object with the parameters
    (measure_height, hub_height, alpha, altitude, pressure, method, losses_pct,
    dt_hours, sectors). Returns (results: dict, ctx: dict with series and tables).
    """
    df, info = clean(raw)
    if len(df) < 100:
        raise ValueError(f"Only {len(df)} valid records; more data is needed to fit the Weibull.")

    # Extrapolate to hub height
    v_meas = df["v"].values
    if args.measure_height and args.hub_height and args.measure_height != args.hub_height:
        v = extrapolate_height(v_meas, args.measure_height, args.hub_height, args.alpha)
    else:
        v = v_meas.copy()

    # Air density (hourly temperature; if missing, use the series mean or 15 °C)
    t = df["t"].fillna(df["t"].mean() if df["t"].notna().any() else 15.0).values
    rho = air_density(t, args.altitude, args.pressure)
    mean_rho = float(np.mean(rho))

    # Weibull
    fits = {
        "MLE": weibull_mle(v),
        "Moments": weibull_moments(v),
        "Energy pattern": weibull_energy_pattern(v),
    }
    method = METHOD_NAMES[args.method]
    k, c = fits[method]
    fit_stats = {n: goodness_of_fit(v, *p) for n, p in fits.items()}

    # Resource
    pdens = power_density(v, rho)
    pdens_weibull = 0.5 * mean_rho * c ** 3 * special.gamma(1 + 3 / k)

    # Turbine
    dt = args.dt_hours
    p_series, e_series_kwh = energy_series(curve, v, rho, dt)
    data_hours = len(v) * dt
    annual_factor = HOURS_PER_YEAR / data_hours
    e_annual_series_kwh = e_series_kwh * annual_factor
    _, e_w_kwh = energy_weibull(curve, k, c, mean_rho)

    loss_factor = 1 - (args.losses_pct / 100.0)
    e_net_series = e_annual_series_kwh * loss_factor
    e_net_weib = e_w_kwh * loss_factor

    pn = curve.rated_power
    cf_series = e_net_series / (pn * HOURS_PER_YEAR)
    cf_weib = e_net_weib / (pn * HOURS_PER_YEAR)

    # Hours by operating regime
    h_op = float(((v >= curve.cut_in) & (v < curve.cut_out)).mean() * HOURS_PER_YEAR)
    h_rated = float((curve.power(v, rho) >= 0.99 * pn).mean() * HOURS_PER_YEAR)
    h_low = float((v < curve.cut_in).mean() * HOURS_PER_YEAR)
    h_high = float((v >= curve.cut_out).mean() * HOURS_PER_YEAR)

    # Monthly energy (if timestamps exist)
    monthly = None
    if isinstance(df.index, pd.DatetimeIndex):
        s = pd.Series(p_series * dt * loss_factor, index=df.index)
        monthly = (s.groupby([s.index.year, s.index.month]).sum() / 1000.0).to_frame("energy_MWh")
        monthly.index = [f"{a}-{m:02d}" for a, m in monthly.index]

    # Wind rose
    rose = wind_rose(df["dir"].values, v, args.sectors)

    res = {
        "data": info,
        "wind": {
            "mean_speed_m_s": round(float(v.mean()), 3),
            "std_dev_m_s": round(float(v.std(ddof=1)), 3),
            "max_speed_m_s": round(float(v.max()), 2),
            "measurement_height_m": args.measure_height,
            "hub_height_m": args.hub_height,
            "shear_alpha": args.alpha,
        },
        "air": {"mean_temperature_C": round(float(np.mean(t)), 2),
                "mean_density_kg_m3": round(mean_rho, 4),
                "altitude_m": args.altitude},
        "weibull": {
            "method_used": method,
            "k": round(k, 3), "c_m_s": round(c, 3),
            "all_methods": {n: {"k": round(p[0], 3), "c": round(p[1], 3)} for n, p in fits.items()},
            "goodness_of_fit": fit_stats,
        },
        "resource": {
            "power_density_series_W_m2": round(pdens, 1),
            "power_density_weibull_W_m2": round(pdens_weibull, 1),
            "reference_wind_class": wind_class(pdens),
        },
        "turbine": {
            "name": curve.name,
            "rated_power_kW": pn,
            "cut_in_m_s": curve.cut_in,
            "rated_speed_m_s": curve.rated_speed,
            "cut_out_m_s": curve.cut_out,
        },
        "energy": {
            "losses_applied_pct": args.losses_pct,
            "annual_series_MWh": round(e_net_series / 1000, 1),
            "annual_weibull_MWh": round(e_net_weib / 1000, 1),
            "weibull_vs_series_diff_pct": round(100 * (e_net_weib - e_net_series) / e_net_series, 2) if e_net_series else None,
            "capacity_factor_series_pct": round(100 * cf_series, 2),
            "capacity_factor_weibull_pct": round(100 * cf_weib, 2),
            "equivalent_full_load_hours_series": round(e_net_series / pn, 0),
            "mean_power_kW_series": round(float(p_series.mean()) * loss_factor, 1),
        },
        "annual_hours_by_regime": {
            "stopped_low_wind": round(h_low, 0),
            "partial_load": round(h_op - h_rated, 0),
            "rated_power": round(h_rated, 0),
            "stopped_high_wind": round(h_high, 0),
        },
    }
    if rose is not None:
        res["prevailing_direction"] = {
            "sector_deg": rose["frequency_pct"].idxmax(),
            "frequency_pct": round(float(rose["frequency_pct"].max()), 1),
        }

    ctx = {"df": df, "v": v, "t": t, "rho": rho, "mean_rho": mean_rho, "p_series": p_series,
           "fits": fits, "k": k, "c": c, "curve": curve, "rose": rose, "monthly": monthly,
           "dt": dt, "loss_factor": loss_factor}
    return res, ctx


def run_cli(args) -> dict:
    """Command-line flow: read files, compute, write results."""
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    raw = read_data(args.data, args.col_speed, args.col_temp, args.col_dir, args.col_time, args.sep)
    curve = PowerCurve.from_csv(args.curve, rho_ref=args.rho_ref, unit=args.power_unit)
    res, ctx = compute(raw, curve, args)

    if ctx["monthly"] is not None:
        ctx["monthly"].to_csv(out / "monthly_energy.csv")
        plot_monthly(ctx["monthly"], out / "monthly_energy.png")
    if ctx["rose"] is not None:
        ctx["rose"].round(3).to_csv(out / "wind_rose.csv")
        plot_rose(ctx["rose"], out / "wind_rose.png")
    plot_weibull(ctx["v"], ctx["fits"], out / "weibull.png")
    plot_curve(curve, ctx["k"], ctx["c"], ctx["mean_rho"], out / "power_curve.png")

    (out / "summary.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(build_report(res), encoding="utf-8")
    return res


def build_report(r: dict) -> str:
    w, e, p, a, v = r["weibull"], r["energy"], r["resource"], r["turbine"], r["wind"]
    lines = [
        "# Wind resource assessment", "",
        f"**Turbine:** {a['name']} ({a['rated_power_kW']:.0f} kW)  ",
        f"**Valid records:** {r['data']['valid_records']} of {r['data']['total_records']} "
        f"({r['data']['availability_pct']} %)", "",
        "## Wind", "",
        f"- Mean speed at {v['hub_height_m']} m: **{v['mean_speed_m_s']} m/s** (σ = {v['std_dev_m_s']} m/s)",
        f"- Mean air density: {r['air']['mean_density_kg_m3']} kg/m³ (mean T {r['air']['mean_temperature_C']} °C)",
        f"- Power density: {p['power_density_series_W_m2']} W/m² (class {p['reference_wind_class']})", "",
        "## Weibull", "",
        f"- Method: {w['method_used']}  →  **k = {w['k']}**, **c = {w['c_m_s']} m/s**", "",
        "| Method | k | c (m/s) | R² | RMSE | KS |", "|---|---|---|---|---|---|",
    ]
    for n, pr in w["all_methods"].items():
        b = w["goodness_of_fit"][n]
        lines.append(f"| {n} | {pr['k']} | {pr['c']} | {b['R2']} | {b['RMSE']} | {b['KS_statistic']} |")
    lines += [
        "", "## Annual energy (with " + str(e["losses_applied_pct"]) + " % losses)", "",
        "| | Time series | Weibull |", "|---|---|---|",
        f"| Energy (MWh/year) | **{e['annual_series_MWh']}** | {e['annual_weibull_MWh']} |",
        f"| Capacity factor (%) | **{e['capacity_factor_series_pct']}** | {e['capacity_factor_weibull_pct']} |",
        "", f"Equivalent full-load hours: {e['equivalent_full_load_hours_series']:.0f} h  ",
        f"Weibull vs time series difference: {e['weibull_vs_series_diff_pct']} %", "",
        "## Annual hours by operating regime", "",
    ]
    for k_, v_ in r["annual_hours_by_regime"].items():
        lines.append(f"- {k_.replace('_', ' ')}: {v_:.0f} h")
    if "prevailing_direction" in r:
        d = r["prevailing_direction"]
        lines += ["", f"**Prevailing direction:** sector {d['sector_deg']}° ({d['frequency_pct']} % of the time)"]
    lines += ["", "Plots: `weibull.png`, `power_curve.png`, `wind_rose.png`, `monthly_energy.png`"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Wind resource: Weibull + power curve + energy yield")
    ap.add_argument("data", help="Hourly CSV: wind speed (m/s), temperature (°C), direction (°)")
    ap.add_argument("curve", help="Power curve CSV: wind speed (m/s), power (kW)")
    ap.add_argument("--output", default="results")
    ap.add_argument("--sep", default=None, help="separator (auto-detected by default)")
    ap.add_argument("--col-speed"); ap.add_argument("--col-temp"); ap.add_argument("--col-dir"); ap.add_argument("--col-time")
    ap.add_argument("--measure-height", type=float, default=None, help="anemometer height (m)")
    ap.add_argument("--hub-height", type=float, default=None, help="hub height (m)")
    ap.add_argument("--alpha", type=float, default=0.143, help="wind shear exponent (default 1/7)")
    ap.add_argument("--altitude", type=float, default=0.0, help="site altitude (m a.s.l.)")
    ap.add_argument("--pressure", type=float, default=None, help="mean pressure (Pa); default: standard atmosphere")
    ap.add_argument("--rho-ref", type=float, default=RHO_0, help="air density of the manufacturer's curve")
    ap.add_argument("--power-unit", default="kW", choices=["W", "kW", "MW"])
    ap.add_argument("--method", default="mle", choices=["mle", "moments", "energy"])
    ap.add_argument("--losses-pct", type=float, default=0.0,
                    help="total losses (wake, availability, electrical...) in %%")
    ap.add_argument("--dt-hours", type=float, default=1.0, help="time step of the data")
    ap.add_argument("--sectors", type=int, default=16)
    args = ap.parse_args(argv)

    try:
        r = run_cli(args)
    except Exception as ex:  # clear message for the user
        print(f"Error: {ex}", file=sys.stderr)
        return 1
    print((Path(args.output) / "report.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
