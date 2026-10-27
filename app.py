"""
Wind resource dashboard (Streamlit).

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.integrate import trapezoid

import windpower as wp

HERE = Path(__file__).parent
st.set_page_config(page_title="Wind resource", page_icon="🌬️", layout="wide")

COLOR_DATA = "#2a7fb8"
COLOR_CURVE = "#c0392b"
COLOR_ENERGY = "#7d3c98"


# ----------------------------------------------------------------------------
# Loading (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_weather(content: bytes, name: str, col_v, col_t, col_dir, col_time) -> pd.DataFrame:
    f = io.BytesIO(content)
    f.name = name
    return wp.read_data(f, col_v or None, col_t or None, col_dir or None, col_time or None)


@st.cache_data(show_spinner=False)
def csv_columns(content: bytes) -> list[str]:
    return list(pd.read_csv(io.BytesIO(content), sep=None, engine="python", nrows=5).columns)


def load_curve(content: bytes, name: str, rho_ref: float, unit: str) -> wp.PowerCurve:
    f = io.BytesIO(content)
    f.name = name
    return wp.PowerCurve.from_csv(f, rho_ref=rho_ref, unit=unit)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🌬️ Wind resource")

st.sidebar.header("1. Weather data")
f_weather = st.sidebar.file_uploader("Hourly CSV: wind speed (m/s), temperature (°C), direction (°)",
                                     type=["csv", "txt"], key="weather")

st.sidebar.header("2. Wind turbine(s)")
f_curves = st.sidebar.file_uploader("Power curve CSV: wind speed (m/s), power",
                                    type=["csv", "txt"], accept_multiple_files=True, key="curves",
                                    help="Upload several curves to compare them.")
power_unit = st.sidebar.selectbox("Power unit in the curve file", ["kW", "W", "MW"])
rho_ref = st.sidebar.number_input("Air density of the manufacturer's curve (kg/m³)", 0.9, 1.4, 1.225, 0.001,
                                  format="%.3f")

use_example = False
if not f_weather and not f_curves:
    use_example = st.sidebar.checkbox("Use example files", value=True)

st.sidebar.header("3. Site and parameters")
c1, c2 = st.sidebar.columns(2)
h_meas = c1.number_input("Anemometer height (m)", 1.0, 200.0, 50.0, 1.0)
h_hub = c2.number_input("Hub height (m)", 1.0, 250.0, 80.0, 1.0)
alpha = st.sidebar.slider("Wind shear exponent α", 0.05, 0.40, 0.143, 0.001,
                          help="0.143 (1/7) is typical for open terrain.")
altitude = st.sidebar.number_input("Site altitude (m a.s.l.)", -100.0, 5000.0, 40.0, 10.0)
losses = st.sidebar.slider("Total losses (%)", 0.0, 40.0, 12.0, 0.5,
                           help="Wake, availability, electrical, etc.")
method = st.sidebar.selectbox("Weibull fitting method", ["mle", "moments", "energy"],
                              format_func={"mle": "Maximum likelihood", "moments": "Moments (Justus)",
                                           "energy": "Energy pattern factor"}.get)
dt_h = st.sidebar.number_input("Data time step (h)", 0.1, 24.0, 1.0, 0.1)
sectors = st.sidebar.select_slider("Wind rose sectors", [8, 12, 16, 24, 36], value=16)

AUTO = "(automatic)"
with st.sidebar.expander("Column names (if not detected automatically)"):
    available = csv_columns(f_weather.getvalue()) if f_weather else []
    options = [AUTO] + available
    sel_v = st.selectbox("Wind speed", options)
    sel_t = st.selectbox("Temperature", options)
    sel_d = st.selectbox("Direction", options)
    sel_f = st.selectbox("Date/time", options)

# ----------------------------------------------------------------------------
# Data input
# ----------------------------------------------------------------------------
st.title("Wind resource assessment")

if use_example:
    weather_bytes, weather_name = (HERE / "example_data.csv").read_bytes(), "example_data.csv"
    curves_in = [((HERE / "example_power_curve.csv").read_bytes(), "example_power_curve.csv")]
    st.info("Showing the example files (synthetic data and a generic 2 MW curve). "
            "Upload your own files in the sidebar to assess your site.")
else:
    if not f_weather or not f_curves:
        st.markdown(
            "### Getting started\n"
            "1. Upload the **weather CSV** (hourly wind speed, temperature and direction).\n"
            "2. Upload the **power curve CSV** of one or more wind turbines.\n"
            "3. Adjust heights, altitude and losses in the sidebar.\n\n"
            "To just try it out, tick *Use example files*."
        )
        missing = []
        if not f_weather:
            missing.append("weather data")
        if not f_curves:
            missing.append("turbine power curve(s)")
        st.warning("Still missing: " + " and ".join(missing) + ".")
        st.stop()
    weather_bytes, weather_name = f_weather.getvalue(), f_weather.name
    curves_in = [(f.getvalue(), f.name) for f in f_curves]

try:
    raw = load_weather(weather_bytes, weather_name,
                       None if sel_v == AUTO else sel_v, None if sel_t == AUTO else sel_t,
                       None if sel_d == AUTO else sel_d, None if sel_f == AUTO else sel_f)
except Exception as ex:
    st.error(f"Could not read the weather file: {ex}")
    st.stop()

curves: list[wp.PowerCurve] = []
for content, name in curves_in:
    try:
        curves.append(load_curve(content, name, rho_ref, power_unit))
    except Exception as ex:
        st.error(f"Could not read the curve '{name}': {ex}")
if not curves:
    st.stop()

params = SimpleNamespace(measure_height=h_meas, hub_height=h_hub, alpha=alpha, altitude=altitude,
                         pressure=None, method=method, losses_pct=losses, dt_hours=dt_h,
                         sectors=sectors)

results, contexts = {}, {}
for cv in curves:
    try:
        results[cv.name], contexts[cv.name] = wp.compute(raw, cv, params)
    except Exception as ex:
        st.error(f"Error computing with '{cv.name}': {ex}")
if not results:
    st.stop()

names = list(results)
if len(names) > 1:
    chosen = st.selectbox("Turbine for the detailed view", names)
else:
    chosen = names[0]
res, ctx = results[chosen], contexts[chosen]
curve = ctx["curve"]
v, k, c, rho_m = ctx["v"], ctx["k"], ctx["c"], ctx["mean_rho"]

# ----------------------------------------------------------------------------
# KPIs
# ----------------------------------------------------------------------------
d, e, w, p = res["data"], res["energy"], res["weibull"], res["resource"]
st.caption(f"{weather_name} · {d['valid_records']:,} of {d['total_records']:,} valid records "
           f"({d['availability_pct']} %) · {chosen} ({curve.rated_power:,.0f} kW)")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Mean wind speed", f"{res['wind']['mean_speed_m_s']:.2f} m/s",
          help=f"At hub height ({h_hub:.0f} m)")
k2.metric("Weibull k / c", f"{w['k']:.2f} / {w['c_m_s']:.2f} m/s")
k3.metric("Power density", f"{p['power_density_series_W_m2']:.0f} W/m²")
k4.metric("Annual energy", f"{e['annual_series_MWh']:,.0f} MWh",
          help="Time series, losses already applied.")
k5.metric("Capacity factor", f"{e['capacity_factor_series_pct']:.1f} %")

tabs = st.tabs(["Summary", "Weibull", "Turbine", "Wind rose", "Monthly", "Compare", "Data & downloads"])

# ---- Summary ----
with tabs[0]:
    a, b = st.columns(2)
    with a:
        st.subheader("Annual energy")
        table = pd.DataFrame({
            "Time series": [e["annual_series_MWh"], e["capacity_factor_series_pct"],
                            e["equivalent_full_load_hours_series"]],
            "Weibull": [e["annual_weibull_MWh"], e["capacity_factor_weibull_pct"],
                        round(e["annual_weibull_MWh"] * 1000 / curve.rated_power)],
        }, index=["Energy (MWh/year)", "Capacity factor (%)", "Equivalent full-load hours (h)"])
        st.dataframe(table, width="stretch")
        st.caption(f"Weibull vs time series difference: {e['weibull_vs_series_diff_pct']} %. "
                   f"Losses applied: {e['losses_applied_pct']} %.")
        st.subheader("Air")
        st.write(f"Mean temperature **{res['air']['mean_temperature_C']} °C** · "
                 f"mean density **{res['air']['mean_density_kg_m3']} kg/m³** "
                 f"(at {altitude:.0f} m a.s.l.)")
    with b:
        st.subheader("Hours per year by operating regime")
        h = res["annual_hours_by_regime"]
        labels = {"stopped_low_wind": "Low wind (stopped)", "partial_load": "Partial load",
                  "rated_power": "Rated power", "stopped_high_wind": "High wind (stopped)"}
        fig = go.Figure(go.Bar(x=[labels[x] for x in h], y=list(h.values()), marker_color=COLOR_DATA,
                               text=[f"{x:,.0f}" for x in h.values()], textposition="outside"))
        fig.update_layout(height=330, margin=dict(t=10, b=10), yaxis_title="Hours/year")
        st.plotly_chart(fig, width="stretch")
    if "prevailing_direction" in res:
        pdn = res["prevailing_direction"]
        st.write(f"**Prevailing direction:** sector {pdn['sector_deg']}° ({pdn['frequency_pct']} % of the time).")
    st.caption(f"Wind class (NREL reference, defined at 50 m): {p['reference_wind_class']}.")

# ---- Weibull ----
with tabs[1]:
    x = v[v > 0]
    vv = np.linspace(0.01, max(x.max() + 1, 25), 400)
    fig = go.Figure()
    fig.add_histogram(x=x, histnorm="probability density", xbins=dict(start=0, size=1),
                      marker_color=COLOR_DATA, opacity=0.45, name="Data")
    for n, (kk, cc) in ctx["fits"].items():
        fig.add_scatter(x=vv, y=wp.weibull_pdf(vv, kk, cc), mode="lines",
                        name=f"{n}: k={kk:.2f}, c={cc:.2f}",
                        line=dict(width=3 if n == w["method_used"] else 1.5,
                                  dash="solid" if n == w["method_used"] else "dot"))
    fig.update_layout(height=420, xaxis_title="Wind speed (m/s)", yaxis_title="Probability density",
                      legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    fit_table = pd.DataFrame(w["goodness_of_fit"]).T
    fit_table.insert(0, "c (m/s)", [w["all_methods"][i]["c"] for i in fit_table.index])
    fit_table.insert(0, "k", [w["all_methods"][i]["k"] for i in fit_table.index])
    st.dataframe(fit_table, width="stretch")
    st.caption("Goodness of fit on the 1 m/s histogram (R², RMSE) and Kolmogorov-Smirnov statistic.")

# ---- Turbine ----
with tabs[2]:
    vv = np.linspace(0, 30, 600)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=vv, y=curve.power(vv), name=f"Curve (ρ={curve.rho_ref:.3f})",
                    line=dict(color=COLOR_CURVE, width=3), secondary_y=False)
    fig.add_scatter(x=vv, y=curve.power(vv, rho_m), name=f"Corrected to ρ={rho_m:.3f}",
                    line=dict(color="#e67e22", width=3, dash="dash"), secondary_y=False)
    fig.add_scatter(x=vv, y=wp.weibull_pdf(vv, k, c), name="Weibull", fill="tozeroy",
                    line=dict(color=COLOR_DATA, width=1), opacity=0.35, secondary_y=True)
    fig.update_xaxes(title="Wind speed (m/s)")
    fig.update_yaxes(title="Power (kW)", secondary_y=False)
    fig.update_yaxes(title="Probability density", secondary_y=True)
    fig.update_layout(height=430, legend=dict(orientation="h", y=-0.2), margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")

    # Contribution to energy by wind speed
    contrib = curve.power(vv, rho_m) * wp.weibull_pdf(vv, k, c)
    area = trapezoid(contrib, vv)
    contrib = contrib / area * 100 if area > 0 else contrib
    fig2 = go.Figure(go.Scatter(x=vv, y=contrib, fill="tozeroy", line=dict(color=COLOR_ENERGY)))
    fig2.update_layout(height=280, title="At which wind speeds is the energy produced? (% per m/s)",
                       xaxis_title="Wind speed (m/s)", yaxis_title="% of energy", margin=dict(t=50))
    st.plotly_chart(fig2, width="stretch")

    t_ = res["turbine"]
    st.write(f"Cut-in **{t_['cut_in_m_s']:.1f} m/s** · rated **{t_['rated_speed_m_s']:.1f} m/s** · "
             f"cut-out **{t_['cut_out_m_s']:.1f} m/s** · rated power **{t_['rated_power_kW']:,.0f} kW**.")

# ---- Wind rose ----
with tabs[3]:
    rose = ctx["rose"]
    if rose is None:
        st.info("The file has no wind direction column.")
    else:
        a, b = st.columns(2)
        angles = [float(i) for i in rose.index]
        width = 360 / len(rose)
        for col, field, title, color in [(a, "frequency_pct", "Frequency (%)", COLOR_DATA),
                                         (b, "energy_share_pct", "Wind energy share by sector (%)", COLOR_ENERGY)]:
            fig = go.Figure(go.Barpolar(r=rose[field], theta=angles, width=[width * 0.9] * len(rose),
                                        marker_color=color))
            fig.update_layout(title=title, height=420, margin=dict(t=60),
                              polar=dict(angularaxis=dict(direction="clockwise", rotation=90)))
            col.plotly_chart(fig, width="stretch")
        st.dataframe(rose.round(2), width="stretch")

# ---- Monthly ----
with tabs[4]:
    m = ctx["monthly"]
    if m is None:
        st.info("To see monthly energy the CSV needs a date column (e.g. `date`).")
    else:
        fig = go.Figure(go.Bar(x=m.index, y=m["energy_MWh"], marker_color=COLOR_ENERGY,
                               text=m["energy_MWh"].round(0), textposition="outside"))
        fig.update_layout(height=400, yaxis_title="Energy (MWh)", margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        df_ = ctx["df"].copy()
        df_["v"] = v
        vm = df_["v"].groupby([df_.index.year, df_.index.month]).mean()
        vm.index = [f"{a}-{b:02d}" for a, b in vm.index]
        fig = go.Figure(go.Scatter(x=vm.index, y=vm.values, mode="lines+markers", line=dict(color=COLOR_DATA)))
        fig.update_layout(height=280, title="Monthly mean wind speed (m/s)", margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")

# ---- Compare ----
with tabs[5]:
    rows = []
    for n, r in results.items():
        rows.append({"Turbine": n, "Rated power (kW)": r["turbine"]["rated_power_kW"],
                     "Energy, series (MWh/year)": r["energy"]["annual_series_MWh"],
                     "Energy, Weibull (MWh/year)": r["energy"]["annual_weibull_MWh"],
                     "Capacity factor (%)": r["energy"]["capacity_factor_series_pct"],
                     "Equivalent full-load hours": r["energy"]["equivalent_full_load_hours_series"]})
    comp = pd.DataFrame(rows).set_index("Turbine")
    if len(comp) == 1:
        st.info("Upload more than one turbine curve in the sidebar to compare them.")
    st.dataframe(comp, width="stretch")
    if len(comp) > 1:
        fig = go.Figure(go.Bar(x=comp.index, y=comp["Energy, series (MWh/year)"], marker_color=COLOR_ENERGY,
                               text=comp["Energy, series (MWh/year)"].round(0), textposition="outside"))
        fig.update_layout(height=360, yaxis_title="MWh/year", margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
        vv = np.linspace(0, 30, 300)
        fig = go.Figure()
        for n, ctx_n in contexts.items():
            fig.add_scatter(x=vv, y=ctx_n["curve"].power(vv, ctx_n["mean_rho"]), name=n, mode="lines")
        fig.update_layout(height=360, xaxis_title="Wind speed (m/s)", yaxis_title="Power (kW)",
                          title="Power curves (density-corrected)", margin=dict(t=50))
        st.plotly_chart(fig, width="stretch")

# ---- Data & downloads ----
with tabs[6]:
    st.subheader("Downloads")
    b1, b2, b3, b4 = st.columns(4)
    b1.download_button("Report (.md)", wp.build_report(res), f"report_{chosen}.md", "text/markdown")
    b2.download_button("Summary (.json)", json.dumps(res, indent=2, ensure_ascii=False),
                       f"summary_{chosen}.json", "application/json")
    if ctx["monthly"] is not None:
        b3.download_button("Monthly energy (.csv)", ctx["monthly"].to_csv(), f"monthly_energy_{chosen}.csv", "text/csv")
    series = pd.DataFrame({"hub_wind_speed_m_s": v, "air_density_kg_m3": ctx["rho"],
                           "power_kW": ctx["p_series"] * ctx["loss_factor"]}, index=ctx["df"].index)
    b4.download_button("Computed hourly series (.csv)", series.to_csv(), f"series_{chosen}.csv", "text/csv")
    st.subheader("Data preview")
    st.dataframe(raw.head(200), width="stretch")
    st.subheader("Report")
    st.markdown(wp.build_report(res))
