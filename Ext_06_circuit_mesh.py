"""Extension of Task 6: mesh-current analysis of a three-loop DC circuit.
Assumptions: ideal sources, resistances in ohms, currents in amperes.
Loop 1: (10+20) I1 - 20 I2 = 12
Loop 2: -20 I1 + (20+30+40) I2 - 40 I3 = 0
Loop 3: -40 I2 + (40+50) I3 = -6   (source opposes the assumed direction)"""
import numpy as np
A = np.array([[30., -20., 0.], [-20., 90., -40.], [0., -40., 90.]])
b = np.array([12., 0., -6.])
I = np.linalg.solve(A, b)
print("Mesh currents (A):", np.round(I, 5))
print("Residual norm:", np.linalg.norm(b - A @ I))
print("Current in the shared 20-ohm resistor (A):", round(I[0] - I[1], 5))
print("Power in that resistor (W):", round(20 * (I[0] - I[1])**2, 5))
