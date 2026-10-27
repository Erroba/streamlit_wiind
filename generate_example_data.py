"""Generate a synthetic one-year hourly CSV (wind speed, temperature, direction) to test the system."""
import numpy as np
import pandas as pd
from scipy import stats as st

rng = np.random.default_rng(42)
idx = pd.date_range("2025-01-01", periods=8760, freq="h")
n = len(idx)

# Wind: Weibull k=2.1, c=7.0 with diurnal and seasonal cycles + persistence (AR1)
k_true, c_true = 2.1, 7.0
z = np.zeros(n)
for i in range(1, n):
    z[i] = 0.93 * z[i - 1] + np.sqrt(1 - 0.93 ** 2) * rng.standard_normal()
u = st.norm.cdf(z)
v = c_true * (-np.log(1 - u)) ** (1 / k_true)
hour = idx.hour.values
doy = idx.dayofyear.values
v *= 1 + 0.15 * np.cos(2 * np.pi * (hour - 15) / 24) + 0.10 * np.cos(2 * np.pi * (doy - 60) / 365)
v = np.clip(v, 0, None).round(2)

# Temperature: annual + diurnal cycle (warm climate)
temp = 24 + 8 * np.sin(2 * np.pi * (doy - 110) / 365) + 5 * np.sin(2 * np.pi * (hour - 9) / 24) \
       + rng.normal(0, 1.2, n)

# Direction: prevailing from the NW (~315°) with scatter
wdir = (315 + rng.normal(0, 40, n) + 25 * np.sin(2 * np.pi * doy / 365)) % 360

pd.DataFrame({"date": idx, "wind_speed_ms": v, "temperature_C": temp.round(1),
              "wind_dir_deg": wdir.round(0)}).to_csv("example_data.csv", index=False)
print("example_data.csv generated:", n, "rows")
