"""Task 20: Random numbers and Monte Carlo (server overload probability)."""
import numpy as np
import os
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)

rng = np.random.default_rng(42)
capacity = 1100.0
mean_demand = 850.0
std_demand = 80.0
N = 1000

# Random demand scenarios
demand = rng.normal(mean_demand, std_demand, N)
overloaded = demand > capacity
overload_probability = overloaded.mean()

print(f"Estimated overload probability: {overload_probability:.4%}")
print(f"Number of overload cases: {overloaded.sum()} out of {N}")

# Running probability to show convergence
running_probability = np.cumsum(overloaded) / np.arange(1, N + 1)
indices = np.unique(
    np.logspace(1, np.log10(N), 200).astype(int)
) - 1

plt.semilogx(indices + 1, running_probability[indices], "k-")
plt.xlabel("Number of simulations")
plt.ylabel("Estimated overload probability")
plt.title("Monte Carlo estimate of server-overload risk")
plt.grid(True)
plt.savefig("figures/task20_monte_carlo.png", dpi=150, bbox_inches="tight")
plt.show()
