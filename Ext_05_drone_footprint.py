"""Extension of Task 5: transforming a triangular drone footprint using
homogeneous coordinates (scale 0.5, rotate 90 degrees, translate by (4, 2))."""
import os
import numpy as np
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)
tri = np.array([[0, 0], [4, 0], [2, 3.]])
homog = np.hstack([tri, np.ones((3, 1))])
th = np.deg2rad(90)
S = np.diag([0.5, 0.5, 1.0])
R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
T = np.array([[1, 0, 4], [0, 1, 2], [0, 0, 1.]])
M = T @ R @ S
new = (homog @ M.T)[:, :2]
print("Combined matrix:\n", np.round(M, 4))
print("Transformed vertices:\n", np.round(new, 4))
print("Area before:", 0.5 * abs(np.linalg.det(np.array([tri[1]-tri[0], tri[2]-tri[0]]))))
print("Area after :", 0.5 * abs(np.linalg.det(np.array([new[1]-new[0], new[2]-new[0]]))))
cl = lambda p: np.vstack([p, p[0]])
plt.plot(*cl(tri).T, "k-o", label="Original")
plt.plot(*cl(new).T, "k--s", label="Transformed")
plt.gca().set_aspect("equal"); plt.grid(True); plt.legend()
plt.title("Drone footprint transformation"); plt.xlabel("x (m)"); plt.ylabel("y (m)")
plt.savefig("figures/ext05_drone.png", dpi=150, bbox_inches="tight"); plt.show()
