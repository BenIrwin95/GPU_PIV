import numpy as np
import matplotlib.pyplot as plt
# slightly modified example from a chatgpt example
# basically the key things that had been missing is the special handling of end points
# you basically ignore the usual formula and just set it equal to 1


# -----------------------------------
# Step 1: Cox-de Boor recursive basis
# -----------------------------------
def N(i, k, u, knots):
    # special case
    if(u==knots[-1] and i==len(knots)-k-2):
        return 1
    if k == 0:
        if (knots[i] <= u < knots[i + 1]):
            return 1.0
        else:
            return 0.0

    denom1 = knots[i + k] - knots[i]
    denom2 = knots[i + k + 1] - knots[i + 1]

    term1 = 0.0
    term2 = 0.0

    if denom1 != 0:
        term1 = (u - knots[i]) / denom1 * N(i, k - 1, u, knots)

    if denom2 != 0:
        term2 = (knots[i + k + 1] - u) / denom2 * N(i + 1, k - 1, u, knots)

    return term1 + term2

# --------------------------
# Step 2: Define B-spline
# --------------------------
def bspline_curve(control_points, knots, degree, u_vals):
    n = len(control_points) - 1
    curve = []

    for u in u_vals:
        point = np.zeros(2)
        for i in range(n + 1):
            coeff = N(i, degree, u, knots)
            point += coeff * control_points[i]
        curve.append(point)

    return np.array(curve)

# ----------------------------
# Step 3: Setup: Points & Knots
# ----------------------------
control_points = np.array([
    [0.5, 0.5],
    [1.0, 2.0],
    [3.0, 3.0],
    [4.0, 0.0]
])

degree = 2  # quadratic B-spline
n = len(control_points) - 1

# Clamped knot vector: [0, 0, 0, 1, 2, 2, 2]
knots = np.array([0, 0, 0, 1, 2, 2, 2], dtype=float)

# --------------------------
# Step 4: Evaluate the curve
# --------------------------
u_vals = np.linspace(knots[degree], knots[-degree-1], 100)
curve = bspline_curve(control_points, knots, degree, u_vals)

# --------------------------
# Step 5: Plot
# --------------------------
plt.plot(curve[:, 0], curve[:, 1], label='B-spline Curve', color='blue')
plt.plot(control_points[:, 0], control_points[:, 1], 'o--', label='Control Points', color='orange')
plt.title('B-spline Curve (Manual Basis Function Calculation)')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()
