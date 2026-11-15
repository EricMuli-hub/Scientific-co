"""Extension of Task 8: steady heat conduction in a metal rod with 10 interior
points. Assumptions: left end 80 C, right end 30 C, no internal heat source."""
import os
import numpy as np
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
n = 10
A = 2*np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
b = np.zeros(n); b[0] = 80.0; b[-1] = 30.0

def jacobi(A, b, tol=1e-8, max_iter=5000):
    D = np.diag(A); R = A - np.diagflat(D); x = np.zeros_like(b); err = []
    for _ in range(max_iter):
        xn = (b - R @ x) / D
        e = np.linalg.norm(xn - x, np.inf); err.append(e)
        if e < tol: return xn, err
        x = xn
    return x, err

def gauss_seidel(A, b, tol=1e-8, max_iter=5000):
    x = np.zeros_like(b); err = []
    for _ in range(max_iter):
        old = x.copy()
        for i in range(len(b)):
            x[i] = (b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ old[i+1:]) / A[i, i]
        e = np.linalg.norm(x - old, np.inf); err.append(e)
        if e < tol: return x, err
    return x, err

xj, ej = jacobi(A, b); xg, eg = gauss_seidel(A, b); xd = np.linalg.solve(A, b)
print("Jacobi iterations:", len(ej), " Gauss-Seidel iterations:", len(eg))
print("Gauss-Seidel temperatures:", np.round(xg, 3))
print("Max difference from direct solve:", np.max(np.abs(xg - xd)))
plt.semilogy(ej, "k-", label="Jacobi"); plt.semilogy(eg, "k--", label="Gauss-Seidel")
plt.xlabel("Iteration"); plt.ylabel("Maximum change"); plt.grid(True); plt.legend()
plt.title("Rod heat conduction: convergence")
plt.savefig("figures/ext08_rod.png", dpi=150, bbox_inches="tight"); plt.show()
