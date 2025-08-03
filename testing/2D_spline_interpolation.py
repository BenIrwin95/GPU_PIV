import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve


# knot vector creation (clamped)
def create_clamped_knot_vector(n,degree):
    m = n + degree + 1
    t=np.linspace(0,1,m-2*degree)
    t = np.concatenate((np.zeros(degree),t,np.ones(degree)))
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





degree = 2  # quadratic B-spline
order=degree+1
nx = 4
ny = 4

x_ref = np.linspace(1,2,nx)
y_ref = np.linspace(1,2,ny)
X_ref, Y_ref = np.meshgrid(x_ref, y_ref)

Z_ref = X_ref**2 + Y_ref**2


# construct the matrix
u_knots = create_clamped_knot_vector(nx,degree)
v_knots = create_clamped_knot_vector(ny,degree)
N=nx*ny
A=np.zeros((N,N))
for i in range(0,ny):
    for j in range(0,nx):
        rowIdx = i*nx + j
        u_ref=j/(nx-1)
        v_ref=i/(ny-1)
        for ii in range(0,ny):
            for jj in range(0,nx):
                colIdx = ii*nx + jj
                A[rowIdx,colIdx] = Bspline_coeffs(u_ref, jj, degree, u_knots)*Bspline_coeffs(v_ref, ii, degree, v_knots)

X_control = X_ref
Y_control = Y_ref

Z_control = solve(A, Z_ref.flatten())
Z_control=np.reshape(Z_control,np.shape(X_control))

n_interp=50
u_interp = np.linspace(0,1,n_interp)
v_interp = np.linspace(0,1,n_interp)
U_interp, V_interp = np.meshgrid(u_interp, v_interp)
z_interp = np.zeros((n_interp,n_interp))



u_knots = create_clamped_knot_vector(nx,degree)
v_knots = create_clamped_knot_vector(ny,degree)
for i in range(0,n_interp):
    for j in range(0,n_interp):
        u = u_interp[j]
        v = v_interp[i]
        for ii in range(0,ny):
            for jj in range(0,nx):
                z_interp[i,j] += Z_control[ii,jj] * Bspline_coeffs(u, jj, degree, u_knots)*Bspline_coeffs(v, ii, degree, v_knots)


plt.figure()
plt.contourf(X_ref,Y_ref,Z_ref)

plt.figure()
plt.contourf(U_interp,V_interp,z_interp)
plt.show()

