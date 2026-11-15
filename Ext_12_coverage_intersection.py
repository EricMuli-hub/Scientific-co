"""Extension of Task 12: where does a road (parabola y = 0.5 x^2 - 2) cross the
edge of a Wi-Fi coverage circle x^2 + y^2 = 9 (radius 3 km)? Four initial guesses."""
import numpy as np
from scipy.optimize import root
def eq(z):
    x, y = z
    return [x**2 + y**2 - 9, y - (0.5*x**2 - 2)]
seen = []
for g in ([2, 1], [-2, 1], [2, -3], [0.5, 3]):
    s = root(eq, g)
    print(f"guess={g}: solution={np.round(s.x, 5)} converged={s.success} residual={np.round(eq(s.x), 10)}")
# Algebraic check: substituting x^2 = 2(y+2) gives y^2 + 2y - 5 = 0
ys = np.roots([1, 2, -5]); print("Roots for y from the polynomial y^2+2y-5=0:", np.round(ys, 5))
print("Only y >= -2 is valid on the parabola; x = +/-", round(float(np.sqrt(2*(ys.max() + 2))), 5))
