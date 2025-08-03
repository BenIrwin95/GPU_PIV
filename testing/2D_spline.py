import numpy as np
import matplotlib.pyplot as plt


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
nx = 6
ny = 6

x_control = np.linspace(1,2,nx)
y_control = np.linspace(1,2,ny)
X_control, Y_control = np.meshgrid(x_control, y_control)

z_control = X_control**2 + Y_control**2


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
                z_interp[i,j] += z_control[ii,jj] * Bspline_coeffs(u, jj, degree, u_knots)*Bspline_coeffs(v, ii, degree, v_knots)


plt.figure()
plt.contourf(X_control,Y_control,z_control)

plt.figure()
plt.contourf(U_interp,V_interp,z_interp)
plt.show()
