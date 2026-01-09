import numpy as np
from scipy.linalg import eigh
from FiniteElementMethod import mu, E, Ifa as I, L, stiff_mat

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

H = np.array([
    0, 8.76, 17.52, 26.28, 35.04, 43.8,
    52.56, 61.32, 70.08, 78.84, 87.6
])

mu = np.array([
    5590.87, 5232.43, 4885.76, 4550.87, 4227.75,
    3916.41, 3616.83, 3329.03, 3053.01, 2788.75, 2536.27
])

E = 2.1e11

I = np.array([
    2.925428571, 2.546761905, 2.206047619, 1.900619048,
    1.628, 1.385761905, 1.171571429, 0.983142857,
    0.818333333, 0.675142857, 0.55152381
])

L = H[1:] - H[:-1]
n_elem = len(L)

def stiff_mat(E, I, L):
    K = np.zeros((2*(len(L)+1), 2*(len(L)+1)))

    for i in range(len(L)):
        Le = L[i]
        Ke = (E*I[i]/Le**3) * np.array([
            [12, 6*Le, -12, 6*Le],
            [6*Le, 4*Le**2, -6*Le, 2*Le**2],
            [-12, -6*Le, 12, -6*Le],
            [6*Le, 2*Le**2, -6*Le, 4*Le**2]
        ])

        idx = slice(2*i, 2*i+4)
        K[idx, idx] += Ke

    return K

def mass_mat(m, L):
    M = np.zeros((2*(len(L)+1), 2*(len(L)+1)))

    for i in range(len(L)):
        Le = L[i]
        me = m[i]

        Me = (me*Le/420) * np.array([
            [156, 22*Le, 54, -13*Le],
            [22*Le, 4*Le**2, 13*Le, -3*Le**2],
            [54, 13*Le, 156, -22*Le],
            [-13*Le, -3*Le**2, -22*Le, 4*Le**2]
        ])

        idx = slice(2*i, 2*i+4)
        M[idx, idx] += Me

    return M

K = stiff_mat(E, I[1:], L)
M = mass_mat(mu[1:], L)

K = K[2:, 2:]
M = M[2:, 2:]

eigvals, eigvecs = eigh(K, M)

eigvals = eigvals[eigvals > 1e-6]  # remove numerical noise
omega = np.sqrt(eigvals)
freq = omega / (2*np.pi)

print("Natural Frequencies (Hz):")
for i, f in enumerate(freq[:5], 1):
    print(f"Mode {i}: {f:.3f} Hz")

modes = eigvecs[:, :5]
trans_dofs = np.arange(0, modes.shape[0], 2)
mode_shapes = modes[trans_dofs, :]

plt.figure()
for i in range(3):
    plt.plot(H, np.r_[0, mode_shapes[:, i]], label=f"{freq[i]:.2f} Hz")

plt.xlabel("Tower Height (m)")
plt.ylabel("Mode Shape")
plt.title("Tower Mode Shapes (FA)")
plt.legend()
plt.grid()
plt.show()

mode = 2
t = np.linspace(0, 2*np.pi, 300)

plt.figure()
line, = plt.plot(H, np.zeros_like(H), '-o')
plt.ylim([-0.02, 0.02])
plt.grid()

for ti in t:
    y = np.r_[0, mode_shapes[:, mode]] * np.sin(ti)
    line.set_ydata(y)
    plt.title(f"Mode {mode+1} Animation")
    plt.pause(0.01)

plt.show()
