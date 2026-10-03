# SPDX-License-Identifier: MIT
"""
Numerical checks accompanying Supplementary Material I and II, revision v3.
(A) Determinants for selected orders 2--6 at one parameter configuration.
    These checks illustrate the unforced harmonic systems; they do not
    prove nonvanishing for all orders or parameters. The general uniqueness
    proof is given in Supplementary Material I, section 6.
(B) A radial finite-volume solve with the order-one angular dependence
    already imposed. It verifies that modal problem on a truncated domain.
    The amplitude error converges quadratically with mesh refinement,
    while f(s)/s is uniform in the cavity to rounding precision.
    Mesh convergence does not remove the separate far-boundary effect.
No classroom effectiveness or unrestricted 2-D numerical validation is claimed.
"""

import numpy as np

mu_r = 1000.0
a, b = 1.0, 2.0
H0 = 1.0
R_far = 5.5          # radius of the circular far boundary (truncated problem)

# =====================================================================
# (A) determinants of the homogeneous boundary-condition systems
# =====================================================================
print("=" * 72)
print("(A) Harmonic-uniqueness check: |det| of the homogeneous BC system")
print("    (nonzero determinant  =>  that harmonic is NOT excited)")
print("=" * 72)


def det_cylinder(nu, mu_r, a, b):
    M = np.array([
        [a**nu,            -a**nu,             -a**(-nu),               0.0],
        [a**(nu - 1), -mu_r * a**(nu - 1),  mu_r * a**(-nu - 1),        0.0],
        [0.0,               b**nu,              b**(-nu),        -b**(-nu)],
        [0.0,          mu_r * b**(nu - 1), -mu_r * b**(-nu - 1),  b**(-nu - 1)],
    ])
    M = M / np.abs(M).max(axis=1, keepdims=True)
    return np.linalg.det(M)


def det_sphere(l, mu_r, a, b):
    M = np.array([
        [a**l,             -a**l,                    -a**(-(l + 1)),                 0.0],
        [l * a**(l - 1), -mu_r * l * a**(l - 1),  mu_r * (l + 1) * a**(-(l + 2)),    0.0],
        [0.0,               b**l,                     b**(-(l + 1)),         -b**(-(l + 1))],
        [0.0,          mu_r * l * b**(l - 1), -mu_r * (l + 1) * b**(-(l + 2)),
         (l + 1) * b**(-(l + 2))],
    ])
    M = M / np.abs(M).max(axis=1, keepdims=True)
    return np.linalg.det(M)


print(f"{'harmonic':>9s} {'|det| cylinder':>16s} {'|det| sphere':>16s}")
for k in range(2, 7):
    print(f"{k:>9d} {abs(det_cylinder(k, mu_r, a, b)):>16.4e} "
          f"{abs(det_sphere(k, mu_r, a, b)):>16.4e}")
print("-> sampled determinants are nonzero for orders 2--6 in this case.")
print("   General uniformity follows from the analytical uniqueness proof.\n")

# =====================================================================
# exact solution of the truncated problem (uniform Dirichlet at R_far)
# unknowns u = (H_i, B2, C2, E3, D3):
#   phi1 = -H_i s cos th ; phi2 = (B2 s + C2/s) cos th ;
#   phi3 = (E3 s + D3/s) cos th ; phi3(R_far) = -H0 R_far cos th
# =====================================================================
M5 = np.array([
    [-a, -a, -1.0 / a, 0.0, 0.0],                    # phi continuous at a
    [1.0, mu_r, -mu_r / a**2, 0.0, 0.0],             # B_n continuous at a
    [0.0, b, 1.0 / b, -b, -1.0 / b],                 # phi continuous at b
    [0.0, mu_r, -mu_r / b**2, -1.0, 1.0 / b**2],     # B_n continuous at b
    [0.0, 0.0, 0.0, R_far, 1.0 / R_far],             # Dirichlet at R_far
])
rhs = np.array([0.0, 0.0, 0.0, 0.0, -H0 * R_far])
H_int_trunc = np.linalg.solve(M5, rhs)[0]

denom = (mu_r + 1) ** 2 * b**2 - (mu_r - 1) ** 2 * a**2
H_int_inf = 4 * mu_r * b**2 * H0 / denom

# =====================================================================
# (B) conservative finite-volume solve of (s mu f')' - mu f / s = 0
# =====================================================================
print("=" * 72)
print("(B) Independent radial solve of the nu = 1 mode (finite volumes)")
print("=" * 72)
print(f"exact H_int, infinite domain        : {H_int_inf:.8e}")
print(f"exact H_int, truncated at R = {R_far} m : {H_int_trunc:.8e}   "
      f"(finite-boundary effect: {abs(H_int_trunc - H_int_inf) / H_int_inf:.1%})")
print()


def solve_radial(h):
    """Cell-centered FV grid on (0, R_far]; a, b and R_far are exact
    multiples of h, so every material interface lies on a cell face."""
    Ncell = int(round(R_far / h))
    faces = h * np.arange(Ncell + 1)            # s = 0 ... R_far
    centers = 0.5 * (faces[:-1] + faces[1:])

    mu_cell = np.ones(Ncell)
    mu_cell[(centers > a) & (centers < b)] = mu_r
    # face permeabilities: harmonic mean of the neighbouring cells
    mu_face = np.empty(Ncell + 1)
    mu_face[1:-1] = 2 * mu_cell[:-1] * mu_cell[1:] / (mu_cell[:-1] + mu_cell[1:])
    mu_face[0], mu_face[-1] = mu_cell[0], mu_cell[-1]

    # tridiagonal system A f = r for f at cell centers
    lo = np.zeros(Ncell)      # sub-diagonal
    di = np.zeros(Ncell)      # diagonal
    up = np.zeros(Ncell)      # super-diagonal
    r = np.zeros(Ncell)

    wE = faces[1:] * mu_face[1:] / h            # east-face conductances
    wW = faces[:-1] * mu_face[:-1] / h          # west face (zero at s = 0)
    sink = mu_cell * h / centers                # from -mu f / s term

    di[:] = -(wE + wW) - sink
    up[:-1] = wE[:-1]
    lo[1:] = wW[1:]
    # outer Dirichlet f(R_far) = -H0 R_far via ghost value at the last face
    di[-1] -= wE[-1]                            # ghost: f_ghost = 2 f_bc - f_N
    r[-1] = -2.0 * wE[-1] * (-H0 * R_far)

    # Thomas algorithm
    for j in range(1, Ncell):
        m = lo[j] / di[j - 1]
        di[j] -= m * up[j - 1]
        r[j] -= m * r[j - 1]
    f = np.empty(Ncell)
    f[-1] = r[-1] / di[-1]
    for j in range(Ncell - 2, -1, -1):
        f[j] = (r[j] - up[j] * f[j + 1]) / di[j]

    core = centers < 0.9 * a
    g = f[core] / centers[core]                 # = -H_int if f is linear
    H_int_fd = -g.mean()
    lin_dev = np.max(np.abs(g / g.mean() - 1.0))
    return H_int_fd, lin_dev


print(f"{'h (m)':>9s} {'H_int (FD)':>14s} {'vs truncated exact':>19s} "
      f"{'max dev. of f(s)/s':>19s}")
res = []
for h in (0.01, 0.005, 0.0025):
    H_fd, lin_dev = solve_radial(h)
    res.append((h, H_fd, lin_dev))
    print(f"{h:>9.4f} {H_fd:>14.8e} "
          f"{abs(H_fd - H_int_trunc) / H_int_trunc:>19.2e} {lin_dev:>19.2e}")

err = [abs(H - H_int_trunc) / H_int_trunc for _, H, _ in res]
print(f"\nrefinement ratios of the H_int error: {err[0] / err[1]:.2f}, "
      f"{err[1] / err[2]:.2f}  (~4 = clean second-order convergence)")
print("note: the deviation of f(s)/s from a constant is already at MACHINE")
print("PRECISION (~1e-13) at every resolution -- the computed cavity field")
print("is uniform to rounding error, independent of h.")
print("-> f(s) is a straight line through the origin inside the cavity:")
print("   the interior field is UNIFORM and the streamlines are STRAIGHT.")
print("   This solver assumes the order-one angular dependence.")
print("   It verifies the modal problem; the general proof is analytical.")
