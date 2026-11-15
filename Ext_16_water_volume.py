"""Extension of Task 16: total water pumped from a borehole.
Flow rate (cubic metres per hour) is logged hourly for 8 hours (9 readings)."""
import numpy as np
from scipy.integrate import simpson
hours = np.arange(0, 9, dtype=float)
flow = np.array([0.0, 3.2, 5.1, 6.0, 6.2, 5.8, 4.7, 3.1, 1.0])
trap = np.trapezoid(flow, hours); simp = simpson(flow, x=hours)
print(f"Trapezoidal volume: {trap:.3f} m3")
print(f"Simpson volume    : {simp:.3f} m3")
print(f"Difference        : {abs(trap - simp):.3f} m3")
print(f"Average flow rate : {trap/8:.3f} m3/h; litres pumped = {simp*1000:,.0f}")
