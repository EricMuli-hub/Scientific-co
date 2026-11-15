"""Extension of Task 14: exponential growth of a bacterial colony.
Model y = a*exp(b*t). Data: colony counts at hourly intervals."""
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
t = np.arange(0, 7, dtype=float)
count = np.array([100, 148, 220, 330, 490, 730, 1080], dtype=float)
model = lambda t, a, b: a*np.exp(b*t)
(p, cov) = curve_fit(model, t, count, p0=[100, 0.4])
a, b = p
pred = model(t, a, b)
rmse = np.sqrt(np.mean((count - pred)**2))
print(f"Fitted model: count = {a:.2f} * exp({b:.4f} t)")
print(f"Parameter standard errors: {np.round(np.sqrt(np.diag(cov)), 4)}")
print(f"RMSE: {rmse:.2f} colonies; doubling time = {np.log(2)/b:.3f} h")
print(f"Prediction at 8 h: {model(8, a, b):.0f}")
tp = np.linspace(0, 8, 200)
plt.scatter(t, count, color="k", zorder=3, label="Counts"); plt.plot(tp, model(tp, a, b), "k--", label="Exponential fit")
plt.xlabel("Time (h)"); plt.ylabel("Colony count"); plt.title("Bacterial growth fit"); plt.legend(); plt.grid(True)
plt.savefig("figures/ext14_bacteria.png", dpi=150, bbox_inches="tight"); plt.show()
