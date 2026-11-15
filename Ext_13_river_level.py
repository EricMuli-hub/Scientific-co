"""Extension of Task 13: river water level recorded every 3 hours (metres).
Estimate the level at 04:30 and 10:00 and compare linear and cubic-spline methods."""
import os
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
hours = np.arange(0, 25, 3, dtype=float)
level = np.array([1.20, 1.15, 1.30, 1.85, 2.40, 2.30, 1.90, 1.55, 1.35])
lin = interp1d(hours, level); cub = CubicSpline(hours, level)
for t in (4.5, 10.0):
    print(f"{int(t):02d}:{int(round((t % 1) * 60)):02d}  linear={float(lin(t)):.3f} m  cubic={float(cub(t)):.3f} m")
t_f = np.linspace(0, 24, 241)
print("Maximum of cubic spline:", round(float(cub(t_f).max()), 3), "m at", round(float(t_f[np.argmax(cub(t_f))]), 1), "h")
plt.scatter(hours, level, color="k", zorder=3, label="3-hourly readings")
plt.plot(t_f, lin(t_f), "k-", label="Linear"); plt.plot(t_f, cub(t_f), "k--", label="Cubic spline")
plt.xlabel("Hour of day"); plt.ylabel("Water level (m)"); plt.title("River level interpolation")
plt.legend(); plt.grid(True)
plt.savefig("figures/ext13_river.png", dpi=150, bbox_inches="tight"); plt.show()
