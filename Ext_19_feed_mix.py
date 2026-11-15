"""Extension of Task 19: cheapest daily feed mix for dairy cows.
Feed A: 30 KES/kg, 0.2 kg protein/kg, 3 MJ/kg. Feed B: 45 KES/kg, 0.4 kg protein/kg, 2.5 MJ/kg.
Requirements per day: protein >= 4 kg and energy >= 50 MJ."""
import numpy as np
from scipy.optimize import minimize, linprog
cost = lambda z: 30*z[0] + 45*z[1]
cons = [{"type": "ineq", "fun": lambda z: 0.2*z[0] + 0.4*z[1] - 4},
        {"type": "ineq", "fun": lambda z: 3*z[0] + 2.5*z[1] - 50}]
res = minimize(cost, x0=[15, 5], method="SLSQP", bounds=[(0, None)]*2, constraints=cons)
x, y = res.x
print(f"SLSQP : Feed A={x:.3f} kg, Feed B={y:.3f} kg, cost={res.fun:.2f} KES, success={res.success}")
lp = linprog([30, 45], A_ub=[[-0.2, -0.4], [-3, -2.5]], b_ub=[-4, -50], bounds=[(0, None)]*2)
print(f"linprog: Feed A={lp.x[0]:.3f} kg, Feed B={lp.x[1]:.3f} kg, cost={lp.fun:.2f} KES")
print(f"Protein supplied: {0.2*x + 0.4*y:.3f} kg; energy supplied: {3*x + 2.5*y:.3f} MJ")
