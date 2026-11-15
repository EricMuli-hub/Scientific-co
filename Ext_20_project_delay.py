"""Extension of Task 20: probability that a three-stage software project
finishes after a 30-day deadline. Durations are triangular (min, most likely, max) in days:
Design (5, 7, 12), Build (10, 14, 24), Testing (4, 6, 12)."""
import os
import numpy as np
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
rng = np.random.default_rng(2024)
N = 200_000
total = (rng.triangular(5, 7, 12, N) + rng.triangular(10, 14, 24, N) + rng.triangular(4, 6, 12, N))
late = total > 30
p = late.mean(); se = np.sqrt(p*(1-p)/N)
print(f"Mean duration: {total.mean():.2f} days, std: {total.std():.2f} days")
print(f"Probability of finishing after 30 days: {p:.4%} (standard error {se:.4%})")
print(f"95% interval for p: {p-1.96*se:.4%} to {p+1.96*se:.4%}")
print(f"90th percentile duration: {np.percentile(total, 90):.2f} days")
plt.hist(total, bins=60, color="white", edgecolor="k")
plt.axvline(30, color="k", ls="--", label="Deadline (30 days)")
plt.xlabel("Total duration (days)"); plt.ylabel("Frequency"); plt.title("Monte Carlo project duration"); plt.legend()
plt.savefig("figures/ext20_project.png", dpi=150, bbox_inches="tight"); plt.show()
