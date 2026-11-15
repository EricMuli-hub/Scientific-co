"""Extension of Task 4: accumulated round-off in a banking ledger.
Assumption: 1,000,000 deposits of exactly KES 0.10 each; the true total is 100,000.00."""
import math
import numpy as np
n = 1_000_000
naive = 0.0
for _ in range(n):
    naive += 0.1
exact_total = 100_000.0
print(f"Naive running sum : {naive:.10f}")
print(f"math.fsum         : {math.fsum([0.1]*n):.10f}")
print(f"Absolute error    : {abs(naive - exact_total):.3e}")
print(f"Relative error    : {abs(naive - exact_total)/exact_total:.3e}")
print("naive == 100000.0 ?", naive == exact_total)
print("np.isclose(naive, 100000.0, rtol=1e-9)?", np.isclose(naive, exact_total, rtol=1e-9))
# Cancellation: (1 - cos x) / x^2 should be close to 0.5 for small x
for x in (1e-4, 1e-6, 1e-8):
    unstable = (1 - np.cos(x)) / x**2
    stable = 2 * np.sin(x / 2)**2 / x**2
    print(f"x={x:.0e}: unstable={unstable:.10f} stable={stable:.10f}")
