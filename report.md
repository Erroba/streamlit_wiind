# Wind resource assessment

**Turbine:** example_power_curve (2000 kW)  
**Valid records:** 8760 of 8760 (100.0 %)

## Wind

- Mean speed at 80.0 m: **6.522 m/s** (σ = 3.486 m/s)
- Mean air density: 1.1829 kg/m³ (mean T 24.0 °C)
- Power density: 322.1 W/m² (class 6 (outstanding))

## Weibull

- Method: MLE  →  **k = 1.965**, **c = 7.366 m/s**

| Method | k | c (m/s) | R² | RMSE | KS |
|---|---|---|---|---|---|
| MLE | 1.965 | 7.366 | 0.9915 | 0.00395 | 0.0195 |
| Moments | 1.975 | 7.358 | 0.9916 | 0.00391 | 0.0192 |
| Energy pattern | 1.951 | 7.356 | 0.9915 | 0.00394 | 0.0179 |

## Annual energy (with 12.0 % losses)

| | Time series | Weibull |
|---|---|---|
| Energy (MWh/year) | **4217.5** | 4266.9 |
| Capacity factor (%) | **24.07** | 24.35 |

Equivalent full-load hours: 2109 h  
Weibull vs time series difference: 1.17 %

## Annual hours by operating regime

- stopped low wind: 1773 h
- partial load: 6568 h
- rated power: 419 h
- stopped high wind: 0 h

**Prevailing direction:** sector 315.0° (20.5 % of the time)

Plots: `weibull.png`, `power_curve.png`, `wind_rose.png`, `monthly_energy.png`
