"""Extension of Task 18: SIR epidemic model solved with solve_ivp.
S' = -beta S I / N; I' = beta S I / N - gamma I; R' = gamma I.
N = 1000, beta = 0.30 per day, gamma = 0.10 per day, I(0) = 1."""
import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
N, beta, gamma = 1000.0, 0.30, 0.10
def sir(t, y, beta=beta):
    S, I, R = y
    return [-beta*S*I/N, beta*S*I/N - gamma*I, gamma*I]
t_eval = np.linspace(0, 160, 321)
sol = solve_ivp(sir, (0, 160), [N - 1, 1, 0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
S, I, R = sol.y
print("Basic reproduction number R0 = beta/gamma =", beta/gamma)
print(f"Peak infections: {I.max():.1f} people on day {sol.t[I.argmax()]:.1f}")
print(f"Final recovered (total ever infected): {R[-1]:.1f} of {N:.0f}")
print(f"Conservation check max |S+I+R-N| = {np.max(np.abs(S + I + R - N)):.2e}")
sol2 = solve_ivp(lambda t, y: sir(t, y, beta=0.15), (0, 160), [N - 1, 1, 0], t_eval=t_eval, rtol=1e-8, atol=1e-10)
print(f"With beta=0.15 (distancing): peak {sol2.y[1].max():.1f} on day {sol2.t[sol2.y[1].argmax()]:.1f}")
plt.plot(sol.t, S, "k-", label="Susceptible"); plt.plot(sol.t, I, "k--", label="Infected"); plt.plot(sol.t, R, "k:", label="Recovered")
plt.xlabel("Day"); plt.ylabel("People"); plt.title("SIR epidemic model"); plt.legend(); plt.grid(True)
plt.savefig("figures/ext18_sir.png", dpi=150, bbox_inches="tight"); plt.show()
