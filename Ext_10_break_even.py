"""Extension of Task 10: break-even production quantity of a bakery.
Revenue = 50 q; cost = 2000 + 20 q + 0.01 q^2; profit(q) = revenue - cost.
Assumptions: q is units per month; price 50; fixed cost 2000; unit cost rises slightly with volume."""
import os
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
profit = lambda q: 50*q - (2000 + 20*q + 0.01*q**2)
lo = root_scalar(profit, bracket=[1, 500], method="bisect", xtol=1e-8)
hi = root_scalar(profit, bracket=[1000, 5000], method="bisect", xtol=1e-8)
g = lambda q: (2000 + 0.01*q**2) / 30      # fixed-point form of profit(q)=0
q = 100.0; n = 0
for n in range(1, 200):
    qn = g(q)
    if abs(qn - q) < 1e-8: q = qn; break
    q = qn
print(f"Lower break-even (bisection): {lo.root:.4f} units ({lo.iterations} iterations)")
print(f"Upper break-even (bisection): {hi.root:.4f} units")
print(f"Lower break-even (fixed point): {q:.4f} units ({n} iterations)")
qs = np.linspace(0, 3200, 400)
best = qs[np.argmax(profit(qs))]
print(f"Maximum profit near q = {best:.0f}: {profit(best):.2f}")
plt.plot(qs, profit(qs), "k-"); plt.axhline(0, color="k", lw=0.8)
plt.axvline(lo.root, color="k", ls="--"); plt.axvline(hi.root, color="k", ls=":")
plt.xlabel("Units per month"); plt.ylabel("Profit"); plt.title("Bakery break-even points"); plt.grid(True)
plt.savefig("figures/ext10_break_even.png", dpi=150, bbox_inches="tight"); plt.show()
