"""Extension of Task 15: acceleration of a car from velocity measured every
2 seconds (m/s). np.gradient uses central differences inside and one-sided at the ends."""
import os
import numpy as np
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
t = np.arange(0, 21, 2, dtype=float)
v = np.array([0, 3.1, 6.0, 8.8, 11.1, 12.9, 14.0, 14.6, 14.9, 15.0, 15.0])
a = np.gradient(v, t)
for ti, vi, ai in zip(t, v, a):
    print(f"t={ti:4.0f} s  v={vi:5.1f} m/s  a={ai:5.2f} m/s^2")
print("Peak acceleration:", round(a.max(), 3), "m/s^2 at t =", t[a.argmax()], "s")
plt.plot(t, a, "k-o"); plt.xlabel("Time (s)"); plt.ylabel("Acceleration (m/s$^2$)")
plt.title("Acceleration estimated from velocity"); plt.grid(True)
plt.savefig("figures/ext15_acceleration.png", dpi=150, bbox_inches="tight"); plt.show()
