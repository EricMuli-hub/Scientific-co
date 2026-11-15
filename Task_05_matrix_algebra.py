"""Task 5: Matrix algebra and matrix operations (2-D graphics transform)."""
import numpy as np
import os
import matplotlib.pyplot as plt
os.makedirs("figures", exist_ok=True)

# Four corners of a square: each row is [x, y]
points = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0]
])

angle = np.deg2rad(45)
rotation = np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle),  np.cos(angle)]
])
scale = np.array([
    [2.0, 0.0],
    [0.0, 1.5]
])

# Combined transformation
transformation = rotation @ scale

# Row-vector form requires multiplication by transpose
transformed_points = points @ transformation.T

print("Original points:\n", points)
print("Transformation matrix:\n", transformation)
print("Transformed points:\n", transformed_points)
print("Determinant:", np.linalg.det(transformation))

# Plot (added to help interpretation)
def closed(p):
    return np.vstack([p, p[0]])
plt.plot(*closed(points).T, "k-", marker="o", label="Original")
plt.plot(*closed(transformed_points).T, "k--", marker="s", label="Transformed")
plt.gca().set_aspect("equal")
plt.xlabel("x"); plt.ylabel("y"); plt.title("Scaling then rotating a square")
plt.legend(); plt.grid(True)
plt.savefig("figures/task05_transform.png", dpi=150, bbox_inches="tight")
plt.show()
