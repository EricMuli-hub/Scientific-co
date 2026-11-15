"""Extension of Task 11: effective interest rate of a loan.
Assumptions: loan KES 500,000 repaid by 36 equal monthly instalments of KES 17,500.
Solve 500000 = 17500 * (1 - (1+r)^-36) / r for the monthly rate r."""
from scipy.optimize import root_scalar
P, pmt, n = 500_000.0, 17_500.0, 36
f = lambda r: pmt * (1 - (1 + r)**(-n)) / r - P
def df(r):
    return pmt * (n*(1 + r)**(-n - 1) * r - (1 - (1 + r)**(-n))) / r**2
newton = root_scalar(f, fprime=df, x0=0.02, method="newton", xtol=1e-14)
secant = root_scalar(f, x0=0.01, x1=0.03, method="secant", xtol=1e-14)
for name, s in (("Newton", newton), ("Secant", secant)):
    print(f"{name:7s} monthly rate = {s.root:.6%}, iterations = {s.iterations}, "
          f"nominal annual = {12*s.root:.4%}, effective annual = {(1+s.root)**12 - 1:.4%}")
print("Total repaid:", pmt * n, " Total interest:", pmt * n - P)
print("Residual f(r):", f(newton.root))
