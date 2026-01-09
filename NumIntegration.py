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

E = np.array([2.1e11] * 11)

Ifa = np.array([
    2.925428571, 2.546761905, 2.206047619, 1.900619048,
    1.628, 1.385761905, 1.171571429, 0.983142857,
    0.818333333, 0.675142857, 0.55152381
])

# wind speed
WSP = 10

plt.figure()
plt.plot(H, mu, '-*')
plt.grid(True)
plt.xlabel('Tower height (m)')
plt.ylabel('Mass per unit length (Kg/m)')
plt.show()

def trapz(x, y):
    integral = 0.0
    for i in range(len(x) - 1):
        dx = x[i + 1] - x[i]
        integral += 0.5 * (y[i] + y[i + 1]) * dx
    return integral

mass = trapz(H, mu)

print(f"Total Mass of the tower = {round(mass)} Kg")
