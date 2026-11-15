"""Extension of Task 9: influence of devices in a campus Wi-Fi network.
Nodes: Router, Switch, Lab1, Lab2, Printer. Links are undirected.
Router-Switch, Router-Lab1, Switch-Lab1, Switch-Lab2, Switch-Printer."""
import numpy as np
names = ["Router", "Switch", "Lab1", "Lab2", "Printer"]
A = np.zeros((5, 5))
for i, j in [(0, 1), (0, 2), (1, 2), (1, 3), (1, 4)]:
    A[i, j] = A[j, i] = 1
w, v = np.linalg.eig(A)
i = np.argmax(w.real)
vec = np.abs(v[:, i].real); c = vec / vec.sum()
print("Dominant eigenvalue:", round(w[i].real, 4))
for n, s in sorted(zip(names, c), key=lambda t: -t[1]):
    print(f"{n:8s} {s:.4f}")
print("Degree of each node:", A.sum(axis=1).astype(int).tolist())
print("Verification error:", np.linalg.norm(A @ v[:, i].real - w[i].real * v[:, i].real))
