import numpy as np
import matplotlib.pyplot as plt

H = np.array([
    0, 8.76, 17.52, 26.28, 35.04, 43.8,
    52.56, 61.32, 70.08, 78.84, 87.6
])

mu = np.array([
    5590.87, 5232.43, 4885.76, 4550.87, 4227.75,
    3916.41, 3616.83, 3329.03, 3053.01, 2788.75, 2536.27
])

E = np.full(len(H), 2.1e11)

Ifa = np.array([
    2.925428571, 2.546761905, 2.206047619, 1.900619048,
    1.628, 1.385761905, 1.171571429, 0.983142857,
    0.818333333, 0.675142857, 0.55152381
])

def stiff_mat(E, I, L):

    E = np.atleast_1d(E).astype(float)
    I = np.atleast_1d(I).astype(float)
    L = np.atleast_1d(L).astype(float)

    if E.size == 1:
        E = np.full(L.size, E[0])
    if I.size == 1:
        I = np.full(L.size, I[0])

    n_elem = L.size
    dof_per_node = 2
    n_nodes = n_elem + 1

    Ke = np.zeros((n_elem, 4, 4))

    for e in range(n_elem):
        Le = L[e]
        Ke[e] = (E[e] * I[e] / Le**3) * np.array([
            [ 12,     6*Le,   -12,     6*Le],
            [ 6*Le, 4*Le**2, -6*Le, 2*Le**2],
            [-12,    -6*Le,    12,    -6*Le],
            [ 6*Le, 2*Le**2, -6*Le, 4*Le**2]
        ])

    Kg = np.zeros((n_nodes*dof_per_node, n_nodes*dof_per_node))

    for e in range(n_elem):
        idx = slice(2*e, 2*e + 4)
        Kg[idx, idx] += Ke[e]

    return Kg, Ke


def cantilever_transverse_defln(K, F_inp):
    F_inp = np.asarray(F_inp, dtype=float)

    # If only transverse forces are supplied
    if F_inp.size == K.shape[0] // 2:
        F = np.zeros(K.shape[0])
        F[0::2] = F_inp
    else:
        F = F_inp

    # Apply fixed boundary conditions (w1 = 0, θ1 = 0)
    K_red = K[2:, 2:]
    F_red = F[2:]

    u = np.linalg.solve(K_red, F_red)

    n_nodes = K.shape[0] // 2
    defln = np.zeros(n_nodes)
    slope = np.zeros(n_nodes)

    defln[1:] = u[0::2]
    slope[1:] = u[1::2]

    return defln, slope

L = H[1:] - H[:-1]

E_elem = E[1:]
I_elem = Ifa[1:]

K, _ = stiff_mat(E_elem, I_elem, L)

Load = np.zeros(len(H))

rho = 1.225         
R = 61.5          
WSP = 10          
a = 0.5

Thrust = 2 * rho * np.pi * R**2 * WSP**2 * (1 - a)
Load[-1] = Thrust

defln, slope = cantilever_transverse_defln(K, Load)

plt.figure(figsize=(6, 8))
plt.plot(np.zeros_like(H), H, 'k--', label='Undeformed')
plt.plot(defln, H, 'b-o', label='Deformed')
plt.xlabel('Transverse Deflection (m)')
plt.ylabel('Tower Height (m)')
plt.title(f'Tower Deflection under Thrust = {Thrust:.0f} N')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
