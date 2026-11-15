"""Extension of Task 7: three-storey building under several wind loads.
Stiffness matrix K (kN/mm) is fixed; load vectors f (kN) change per scenario.
LU factorisation is done once. K u = f gives floor displacements u (mm)."""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
k = 50.0
K = k * np.array([[2., -1., 0.], [-1., 2., -1.], [0., -1., 1.]])
loads = {"calm": [5, 5, 5], "moderate wind": [10, 20, 30], "storm": [20, 40, 80]}
lu, piv = lu_factor(K)
for name, f in loads.items():
    f = np.array(f, dtype=float)
    u = lu_solve((lu, piv), f)
    print(f"{name:14s} displacement (mm) = {np.round(u, 4)}  residual = {np.linalg.norm(f - K @ u):.1e}")
    drift = np.diff(np.concatenate([[0.0], u]))
    print(f"{'':14s} inter-storey drift (mm) = {np.round(drift, 4)}")
