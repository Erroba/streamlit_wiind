# Wind resource assessment (Weibull + power curve)

Takes an hourly CSV of wind data and computes: Weibull distribution, power density,
annual energy yield of a wind turbine (from the time series and from the Weibull fit),
capacity factor, wind rose and monthly energy.

## Installation

    pip install -r requirements.txt

## Dashboard (recommended)

    streamlit run app.py

It opens in your browser (http://localhost:8501). In the sidebar:

1. Upload the **weather** CSV (wind speed, temperature, direction).
2. Upload the **power curve** CSV of one or more turbines (to compare them).
3. Adjust heights, altitude, shear exponent, losses and fitting method.

Tabs: Summary, Weibull, Turbine, Wind rose, Monthly, Compare, Data & downloads
(report .md, summary .json, monthly energy .csv and computed hourly series .csv).
With no files uploaded, tick *Use example files* to try it.

## Command line

    python windpower.py data.csv curve.csv --output results \
        --measure-height 50 --hub-height 80 --altitude 40 --losses-pct 12

Quick test with synthetic data:

    python generate_example_data.py
    python windpower.py example_data.csv example_power_curve.csv

## CSV formats

**Weather data** (one row per hour). Column names are detected automatically, in English or Spanish
(wind_speed / velocidad / ws, temperature / temperatura / temp, wind_direction / direccion / dir,
date / fecha / datetime). For other names use `--col-speed`, `--col-temp`, `--col-dir`, `--col-time`
(or the column selectors in the dashboard). Temperature and direction are optional
(without temperature the standard density at the given altitude is used).

    date,wind_speed_ms,temperature_C,wind_dir_deg
    2025-01-01 00:00:00,5.2,18.4,310

**Power curve**: wind speed (m/s) and power (kW). Use `--power-unit W|MW` if it is not in kW.
`example_power_curve.csv` is a generic 2 MW curve (illustrative, not from a manufacturer):
replace it with the real curve of your turbine.

## Main options

| Option | Description |
|---|---|
| `--measure-height`, `--hub-height` | Extrapolate wind to hub height with the power law (`--alpha`, default 0.143) |
| `--altitude`, `--pressure` | Air density (together with hourly temperature) |
| `--rho-ref` | Air density the curve is given at (default 1.225 kg/m³) |
| `--method` | Weibull fit used: `mle` (default), `moments`, `energy` |
| `--losses-pct` | Overall losses (wake, availability, electrical) |
| `--dt-hours` | Time step of the data (default 1) |

## Method

- **Weibull:** f(v) = (k/c)(v/c)^(k-1) exp(-(v/c)^k). Three fitting methods with R², RMSE and KS.
- **Air density:** ρ = P/(R·T), with P from the standard atmosphere at the site altitude.
- **Curve correction:** v_eq = v·(ρ/ρ_ref)^(1/3) (IEC 61400-12-1 approach).
- **Energy from the series:** sum of P(v_hour) × Δt, scaled to 8760 h if the record is not a full year.
- **Energy from the Weibull:** integral of P(v)·f(v) × 8760 h.
- **Capacity factor:** net energy / (rated power × 8760).

## Outputs (folder `results/`)

`report.md`, `summary.json`, `weibull.png`, `power_curve.png`, `wind_rose.png`,
`monthly_energy.png`, `monthly_energy.csv`, `wind_rose.csv`.

## Notes

- The "wind class" uses NREL power density thresholds, defined at 50 m; at other heights it is indicative only.
- A bankable assessment needs at least one full year of data, long-term correlation (MCP) and computed wake losses.
