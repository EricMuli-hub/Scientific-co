"""Extension of Task 2: vaccine cold-room monitoring.
Assumptions: 24 hourly readings from one IoT probe; safe range is 2-8 C."""
import numpy as np
temps_c = np.array([4.2, 4.0, 3.8, 3.9, 4.1, 4.5, 5.2, 6.1, 7.0, 7.8, 8.4, 8.9,
                    9.1, 8.6, 7.7, 6.9, 6.0, 5.4, 4.9, 4.6, 4.4, 4.3, 4.2, 4.1])
out_of_range = (temps_c < 2) | (temps_c > 8)
print(f"Mean {temps_c.mean():.2f} C, min {temps_c.min():.1f} C, max {temps_c.max():.1f} C")
print("Kelvin (first 3):", np.round(temps_c[:3] + 273.15, 2))
print("Hours outside 2-8 C:", out_of_range.sum(), "at hours", np.where(out_of_range)[0].tolist())
print(f"Share of day out of range: {out_of_range.mean():.1%}")
