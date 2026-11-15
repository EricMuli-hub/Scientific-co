"""Instruction C: change at least one input/parameter for every task and
observe how the result changes. Baseline values are the ones in the original
task programs. Run: python code/parameter_changes.py"""
import timeit
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import root_scalar, root, curve_fit, minimize
from scipy.interpolate import interp1d, CubicSpline
from scipy.integrate import simpson, solve_ivp
from scipy.stats import norm


def header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)

# ---------------------------------------------------------------- Task 2
header("TASK 2: add 3 C to every reading (hot day)")
temps = np.array([18.5, 18.0, 17.8, 17.5, 17.9, 19.0, 21.5, 23.8, 25.6, 27.1,
                  28.3, 29.0, 29.4, 29.1, 28.5, 27.6, 26.2, 24.8, 23.4, 22.1,
                  21.0, 20.2, 19.6, 19.0])
for label, t in (("baseline", temps), ("+3 C", temps + 3.0)):
    print(f"{label:9s} mean={t.mean():.2f} min={t.min():.2f} max={t.max():.2f} "
          f"first F reading={1.8*t[0]+32:.2f}")

# ---------------------------------------------------------------- Task 3
header("TASK 3: change number of measurements")
for n in (100_000, 1_000_000, 3_000_000):
    rng = np.random.default_rng(42)
    lat = rng.uniform(5, 250, n)
    def loop():
        out = []
        for v in lat:
            out.append((v / 1000.0) ** 2)
        return out
    def vec():
        return (lat / 1000.0) ** 2
    tl = timeit.timeit(loop, number=1)
    tv = timeit.timeit(vec, number=3) / 3
    print(f"N={n:>9,d}  loop={tl:.4f}s  vectorized={tv:.5f}s  speed-up={tl/tv:.1f}x")

# ---------------------------------------------------------------- Task 4
header("TASK 4: change the approximation and the size of x")
true = np.sqrt(2)
for approx in (1.414, 1.4142, 1.41421356):
    ae = abs(true - approx)
    print(f"approx={approx:<10} absolute error={ae:.3e} relative error={ae/true:.6%}")
for x in (1e-12, 1e-14, 1e-16):
    direct = (np.sqrt(1 + x) - 1) / x
    stable = 1 / (np.sqrt(1 + x) + 1)
    print(f"x={x:.0e}: direct={direct:.6f} stable={stable:.6f}")

# ---------------------------------------------------------------- Task 5
header("TASK 5: change rotation angle and scale factors")
pts = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
for ang, sx, sy in ((45, 2.0, 1.5), (90, 3.0, 3.0), (30, 1.0, 1.0)):
    th = np.deg2rad(ang)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    S = np.diag([sx, sy])
    M = R @ S
    print(f"angle={ang} deg, scale=({sx},{sy}) det={np.linalg.det(M):.4f}")
    print(np.round(pts @ M.T, 4))

# ---------------------------------------------------------------- Task 6
header("TASK 6: change observed totals; then make the servers identical in ratio")
A = np.array([[4., 2.], [8., 12.]])
for b in (np.array([20., 72.]), np.array([26., 76.])):
    s = np.linalg.solve(A, b)
    print("totals", b, "-> Type A, Type B =", np.round(s, 4))
A_sing = np.array([[4., 2.], [8., 4.]])  # Type B now proportional to Type A
print("det of singular matrix:", np.linalg.det(A_sing))
try:
    np.linalg.solve(A_sing, np.array([20., 72.]))
except np.linalg.LinAlgError as e:
    print("np.linalg.solve raised LinAlgError:", e)

# ---------------------------------------------------------------- Task 7
header("TASK 7: strengthen the coupling between unknowns (condition number)")
b1 = np.array([15., 10., 10.])
for c in (-1.0, -1.9):
    A = np.array([[4, c, 0], [c, 4, c], [0, c, 3.0]])
    lu, piv = lu_factor(A)
    x = lu_solve((lu, piv), b1)
    print(f"coupling={c}: solution={np.round(x, 4)} cond(A)={np.linalg.cond(A):.2f}")

# ---------------------------------------------------------------- Task 8
header("TASK 8: change right boundary and tolerance")
A = (np.diag([2.] * 5) + np.diag([-1.] * 4, 1) + np.diag([-1.] * 4, -1))
def jac(A, b, tol):
    D = np.diag(A); R = A - np.diagflat(D); x = np.zeros_like(b); n = 0
    while n < 10000:
        xn = (b - R @ x) / D; n += 1
        if np.linalg.norm(xn - x, np.inf) < tol:
            return xn, n
        x = xn
    return x, n
def gs(A, b, tol):
    x = np.zeros_like(b); n = 0
    while n < 10000:
        old = x.copy(); n += 1
        for i in range(len(b)):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ old[i+1:]) / A[i, i]
        if np.linalg.norm(x - old, np.inf) < tol:
            return x, n
    return x, n
for right, tol in ((20., 1e-8), (80., 1e-8), (20., 1e-3), (20., 1e-12)):
    b = np.array([100., 0, 0, 0, right])
    xj, nj = jac(A, b, tol); xg, ng = gs(A, b, tol)
    print(f"right={right:>5}, tol={tol:.0e}: Jacobi {nj} it, Gauss-Seidel {ng} it, "
          f"temperatures={np.round(xg, 3)}")

# ---------------------------------------------------------------- Task 9
header("TASK 9: add a link between nodes A and D")
def centrality(M):
    w, v = np.linalg.eig(M)
    i = np.argmax(w.real)
    vec = np.abs(v[:, i].real)
    return w[i].real, vec / vec.sum()
base = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0.]])
mod = base.copy(); mod[0, 3] = mod[3, 0] = 1
for name, M in (("baseline", base), ("A-D linked", mod)):
    lam, c = centrality(M)
    print(f"{name:10s} dominant eigenvalue={lam:.4f} importance={np.round(c, 4)}")

# ---------------------------------------------------------------- Task 10
header("TASK 10: change amplitude and tolerance of the heat-balance model")
for amp, tol in ((10, 1e-10), (10, 1e-3), (30, 1e-10)):
    f = lambda T, amp=amp: T - 25 - amp*np.exp(-T/20)
    r = root_scalar(f, bracket=[20, 60], method="bisect", xtol=tol)
    T = 30.0; n = 0
    for n in range(1, 200):
        Tn = 25 + amp*np.exp(-T/20)
        if abs(Tn - T) < tol:
            T = Tn; break
        T = Tn
    print(f"amp={amp}, tol={tol:.0e}: bisection root={r.root:.6f} ({r.iterations} it), "
          f"fixed-point={T:.6f} ({n} it)")

# ---------------------------------------------------------------- Task 11
header("TASK 11: change the last cash inflow and the starting guess")
def make(cf):
    npv = lambda r: sum(c/(1+r)**t for t, c in enumerate(cf))
    dn = lambda r: sum(-t*c/(1+r)**(t+1) for t, c in enumerate(cf) if t > 0)
    return npv, dn
for cf in ([-10000, 3000, 3500, 4000, 2500], [-10000, 3000, 3500, 4000, 4500]):
    npv, dn = make(cf)
    nw = root_scalar(npv, fprime=dn, x0=0.10, method="newton", xtol=1e-12)
    sc = root_scalar(npv, x0=0.05, x1=0.20, method="secant", xtol=1e-12)
    print(f"cash flows {cf}: IRR Newton={nw.root:.4%} ({nw.iterations} it), "
          f"Secant={sc.root:.4%} ({sc.iterations} it)")
npv, dn = make([-10000, 3000, 3500, 4000, 2500])
for x0 in (0.10, 0.50, 0.90):
    r = root_scalar(npv, fprime=dn, x0=x0, method="newton", xtol=1e-12)
    print(f"Newton start x0={x0}: root={r.root:.6%} in {r.iterations} iterations")

# ---------------------------------------------------------------- Task 12
header("TASK 12: change radius and the initial guesses")
for rad in (2.0, 3.0):
    eq = lambda z, rad=rad: [z[0]**2 + z[1]**2 - rad**2, z[1] - (z[0]**2 - 1)]
    for g in ([1, 1], [-1, 1], [0, -2], [2, -2]):
        s = root(eq, g)
        print(f"radius={rad}, guess={g}: solution={np.round(s.x, 5)}, converged={s.success}")

# ---------------------------------------------------------------- Task 13
header("TASK 13: replace the 10:00 reading with an outlier (22.5 -> 26.0)")
hours = np.arange(8, 15, dtype=float)
for label, temp in (("baseline", [18., 20., 22.5, 24., 25., 25.5, 25.]),
                    ("outlier", [18., 20., 26.0, 24., 25., 25.5, 25.])):
    lin = interp1d(hours, temp); cub = CubicSpline(hours, temp)
    print(f"{label:9s} linear(10:30)={float(lin(10.5)):.2f}  cubic(10:30)={float(cub(10.5)):.2f}"
          f"  linear(9:30)={float(lin(9.5)):.2f}  cubic(9:30)={float(cub(9.5)):.2f}")

# ---------------------------------------------------------------- Task 14
header("TASK 14: change the model degree")
users = np.array([50, 100, 150, 200, 250, 300, 350.])
resp = np.array([82, 95, 115, 145, 185, 238, 305.])
for deg in (1, 2, 3):
    p = np.polyfit(users, resp, deg)
    rmse = np.sqrt(np.mean((resp - np.polyval(p, users))**2))
    print(f"degree {deg}: RMSE={rmse:.2f} ms, prediction at 400 users={np.polyval(p, 400):.1f} ms")

# ---------------------------------------------------------------- Task 15
header("TASK 15: change step size h on f(x)=x^3 at x=2 (true derivative = 12)")
f = lambda x: x**3
for h in (1.0, 0.1, 0.01, 0.001):
    fwd = (f(2 + h) - f(2)) / h
    cen = (f(2 + h) - f(2 - h)) / (2*h)
    print(f"h={h:<6}: forward={fwd:.6f} (error {abs(fwd-12):.2e})  central={cen:.6f} (error {abs(cen-12):.2e})")
t = np.arange(7.); pos = np.array([0, 4.8, 10.2, 16.1, 22.7, 30.0, 38.0])
print("GPS data using every 2nd sample (dt=2 s) at t=2:",
      (pos[4] - pos[0]) / (t[4] - t[0]), "m/s  versus dt=1 s:",
      (pos[3] - pos[1]) / (t[3] - t[1]), "m/s")

# ---------------------------------------------------------------- Task 16
header("TASK 16: change the peak reading (1.8 -> 3.0 kW)")
th = np.arange(0, 13, 2.)
for label, p in (("baseline", [0.8, 1.0, 1.4, 1.8, 1.6, 1.2, 0.9]),
                 ("peak 3.0", [0.8, 1.0, 1.4, 3.0, 1.6, 1.2, 0.9])):
    p = np.array(p)
    print(f"{label:9s} trapezoid={np.trapezoid(p, th):.3f} kWh  simpson={simpson(p, x=th):.3f} kWh")

# ---------------------------------------------------------------- Task 17
header("TASK 17: change step size h and rate constant k")
for k, h in ((0.12, 1.0), (0.12, 0.1), (0.12, 5.0), (0.40, 1.0), (0.40, 5.0)):
    time = np.arange(0, 30 + h/2, h)
    T = np.zeros_like(time); T[0] = 90.
    for i in range(len(time) - 1):
        T[i+1] = T[i] + h * (-k * (T[i] - 25))
    exact = 25 + 65*np.exp(-k*time[-1])
    print(f"k={k}, h={h}: Euler={T[-1]:.3f} exact={exact:.3f} error={abs(T[-1]-exact):.4f}")

# ---------------------------------------------------------------- Task 18
header("TASK 18: change R and C (time constant tau = R*C)")
for R, C in ((1000., 0.001), (2000., 0.001), (1000., 0.0005)):
    sol = solve_ivp(lambda t, V: [(5 - V[0])/(R*C)], (0, 5), [0.0],
                    t_eval=np.linspace(0, 5, 200), rtol=1e-9, atol=1e-11)
    tau = R*C
    t63 = sol.t[np.argmax(sol.y[0] >= 0.632*5)]
    print(f"R={R:.0f}, C={C}: tau={tau} s, V(5 s)={sol.y[0, -1]:.4f} V, reaches 63% of 5 V at about {t63:.2f} s")

# ---------------------------------------------------------------- Task 19
header("TASK 19: change cost of Type B and the capacity requirement")
def solve(costA, costB, need):
    res = minimize(lambda z: costA*z[0] + costB*z[1], x0=[5, 3], method="SLSQP",
                   bounds=[(0, None)]*2,
                   constraints={"type": "ineq", "fun": lambda z: 100*z[0] + 170*z[1] - need})
    best = None
    for a in range(0, 40):
        for b in range(0, 40):
            if 100*a + 170*b >= need:
                c = costA*a + costB*b
                if best is None or c < best[0]:
                    best = (c, a, b, 100*a + 170*b)
    return res, best
for cA, cB, need in ((12, 18, 1000), (12, 25, 1000), (12, 18, 1500)):
    res, best = solve(cA, cB, need)
    print(f"costA={cA}, costB={cB}, need={need}: A={res.x[0]:.3f}, B={res.x[1]:.3f}, "
          f"cost={res.fun:.2f}; integer best (cost,A,B,capacity)={best}")

# ---------------------------------------------------------------- Task 20
header("TASK 20: change N, standard deviation and capacity")
for N, mean, std, cap in ((100_000, 850, 120, 1000), (1_000, 850, 120, 1000),
                          (100_000, 850, 80, 1000), (100_000, 850, 120, 1100)):
    rng = np.random.default_rng(42)
    d = rng.normal(mean, std, N)
    p = (d > cap).mean()
    exact = 1 - norm.cdf(cap, mean, std)
    print(f"N={N:>7,d}, std={std}, capacity={cap}: simulated={p:.4%}, exact normal tail={exact:.4%}")
