"""Extension of Task 17: logistic growth of fish in a pond using Euler's method.
dP/dt = r P (1 - P/K); r = 0.8 per month, K = 5000 fish, P(0) = 200."""
import os
import numpy as np
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
r, K, P0, end = 0.8, 5000.0, 200.0, 12.0
def euler(h):
    t = np.arange(0, end + h/2, h); P = np.zeros_like(t); P[0] = P0
    for i in range(len(t) - 1):
        P[i+1] = P[i] + h * r * P[i] * (1 - P[i]/K)
    return t, P
exact = lambda t: K / (1 + (K/P0 - 1) * np.exp(-r*t))
for h in (1.0, 0.5, 0.1):
    t, P = euler(h)
    print(f"h={h}: Euler P(12)={P[-1]:.2f}, exact={exact(12.0):.2f}, error={abs(P[-1]-exact(12.0)):.2f}")
t, P = euler(1.0); t2, P2 = euler(0.1)
print("Month when population first exceeds 2500 (h=0.1):", round(t2[np.argmax(P2 > 2500)], 1))
plt.plot(t, P, "k--", marker="o", label="Euler h=1"); plt.plot(t2, P2, "k:", label="Euler h=0.1")
tt = np.linspace(0, end, 300); plt.plot(tt, exact(tt), "k-", label="Exact")
plt.xlabel("Month"); plt.ylabel("Fish population"); plt.title("Logistic growth in a fish pond")
plt.legend(); plt.grid(True)
plt.savefig("figures/ext17_fish.png", dpi=150, bbox_inches="tight"); plt.show()
