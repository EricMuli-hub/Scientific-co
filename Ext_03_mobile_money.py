"""Extension of Task 3: vectorized currency conversion and fee on 1 million
mobile-money transactions. Assumptions: amounts in KES uniformly between 50 and
70,000; rate 129 KES per USD; 1.5% service fee."""
import numpy as np, timeit
rng = np.random.default_rng(7)
amount_kes = rng.uniform(50, 70_000, 1_000_000)
RATE, FEE = 129.0, 0.015
def loop():
    return [(a / RATE) * (1 + FEE) for a in amount_kes]
def vec():
    return amount_kes / RATE * (1 + FEE)
tl = timeit.timeit(loop, number=1)
tv = timeit.timeit(vec, number=3) / 3
res = vec()
print("First three USD totals:", np.round(res[:3], 2))
print(f"Total value (USD): {res.sum():,.2f}")
print(f"Loop {tl:.4f}s, vectorized {tv:.5f}s, speed-up {tl/tv:.0f}x")
