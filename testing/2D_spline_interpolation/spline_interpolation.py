import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


# knot vector creation (clamped)
def create_clamped_knot_vector(n,degree):
    m = n + degree + 1
    t=np.linspace(0,1,m-2*degree)
    t = np.concatenate((np.zeros(degree),t,np.ones(degree)))
    return t

def create_uniform_knot_vector(n,degree):
    m = n + degree + 1
    t=np.linspace(0,1,m)
    return t


def Bspline_coeffs(u, i, k, knots):
    #u: the value within the knot vector we are evaluating at
    #i: the ith Bspline coefficient are evaluating
    #k: the degree of coefficient we want
    #knots: the knot vector

    # handle special case at end points
    if(u==knots[-1] and i==len(knots)-k-2):
        return 1
    # handle the first row of coeffs
    if(k==0):
        if(knots[i] <= u < knots[i + 1]):
            return 1
        else:
            return 0
    else:
        denom1 = knots[i + k] - knots[i]
        denom2 = knots[i + k + 1] - knots[i + 1]

        term1 = 0.0
        term2 = 0.0
        # we recursively have the function call itself
        # when k==0, the recursion stops
        if denom1 != 0:
            term1 = (u - knots[i]) / denom1 * Bspline_coeffs(u,i, k - 1, knots)

        if denom2 != 0:
            term2 = (knots[i + k + 1] - u) / denom2 * Bspline_coeffs(u,i + 1, k - 1, knots)

        return term1 + term2


def Bspline_curve(u_vals,C,degree):
    #u the value within the knot vector range we wish to evaluate at
    #C: control points
    #degree: 2 == quadratic

    # generate clamped knot vector
    knots = create_clamped_knot_vector(len(control_points),degree)

    #evaluate spline at all specified points
    out = np.zeros((len(u_vals), np.shape(control_points)[1]))
    for j in range(0,len(u_vals)):
    #for j in range(0,1):
        u = u_vals[j]
        for i in range(0,n):
            B = Bspline_coeffs(u, i, degree, knots)
            out[j] += B * control_points[i]
    return out



# the points we want to pass through
ref_points = np.array([
    [0.5, 0.5],
    [1.0, 2.0],
    [3.0, 3.0],
    [4.0, 0.0]
])

degree = 3  # quadratic B-spline
order=degree+1
n = len(ref_points)

knots = create_clamped_knot_vector(n,degree)
#knots = create_uniform_knot_vector(n,degree) # seems to break
print(knots)
control_points=np.zeros(np.shape(ref_points))

# basis matrix
A=np.zeros((n,n))
b = np.zeros((n,1))
for i in range(0,n):
    u_eval = i/(n-1)
    for j in range(0,n):
        A[i,j] = Bspline_coeffs(u_eval, j, degree, knots)

print(A)

control_points[:,0] = solve(A, ref_points[:,0])
control_points[:,1] = solve(A, ref_points[:,1])


u_vals = np.linspace(0,1,100)
out = Bspline_curve(u_vals,control_points,degree)


plt.figure()
plt.plot(control_points[:, 0], control_points[:, 1], 'o--', label='Control Points', color='orange')
plt.plot(ref_points[:, 0], ref_points[:, 1], 'o', label='Re Points')
plt.plot(out[:,0],out[:,1])
plt.show()

