from math import e, exp
import numpy as np
import matplotlib.pyplot as plt

'''
Given:
x**2 * y'' + (x + 1) * y' - y = 0
y(1) = 2 + e
y'(1) = 1

Converted:
y' = g(x, y, z) = z
z' = f(x, y, z) = (y - (x + 1) * y') / x**2
y(1) = 2 + e
z(1) = 1

Exact solution:
y = x + 1 + x * e**(1/x)
'''

def f(x, y, z):
    return (y - (x + 1) * z) / x**2

def g(x, y, z):
    return z

def exact_solution(x):
    return x + 1 + x * exp(1/x)

def euler_method(f, g, y0, z0, interval, h):
    l, r = interval
    x = [i for i in np.arange(l, r + h, h)]
    y = [y0]
    z = z0
    for i in range(len(x) - 1):
        z += h * f(x[i], y[i], z)
        y.append(y[i] + h * g(x[i], y[i], z))
    return x, y

def runge_kutta_method(f, g, y0, z0, interval, h, return_z=False):
    l, r = interval
    x = [i for i in np.arange(l, r + h, h)]
    y = [y0]
    z = [z0]
    for i in range(len(x) - 1):
        K1 = h * g(x[i], y[i], z[i])
        L1 = h * f(x[i], y[i], z[i])
        K2 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K1, z[i] + 0.5 * L1)
        L2 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K1, z[i] + 0.5 * L1)
        K3 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * K2, z[i] + 0.5 * L2)
        L3 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * K2, z[i] + 0.5 * L2)
        K4 = h * g(x[i] + h, y[i] + K3, z[i] + L3)
        L4 = h * f(x[i] + h, y[i] + K3, z[i] + L3)
        delta_y = (K1 + 2 * K2 + 2 * K3 + K4) / 6
        delta_z = (L1 + 2 * L2 + 2 * L3 + L4) / 6
        y.append(y[i] + delta_y)
        z.append(z[i] + delta_z)

    if not return_z:
        return x, y
    else:
        return x, y, z

def adams_method(f, g, y0, z0, interval, h):
    x_runge, y_runge, z_runge = runge_kutta_method(f, g, y0, z0, interval, h, return_z=True)
    x = x_runge
    y = y_runge[:4]
    z = z_runge[:4]
    for i in range(3, len(x_runge) - 1):
        z_i = z[i] + h * (55 * f(x[i], y[i], z[i]) -
                          59 * f(x[i - 1], y[i - 1], z[i - 1]) +
                          37 * f(x[i - 2], y[i - 2], z[i - 2]) -
                          9 * f(x[i - 3], y[i - 3], z[i - 3])) / 24
        z.append(z_i)
        y_i = y[i] + h * (55 * g(x[i], y[i], z[i]) -
                          59 * g(x[i - 1], y[i - 1], z[i - 1]) +
                          37 * g(x[i - 2], y[i - 2], z[i - 2]) -
                          9 * g(x[i - 3], y[i - 3], z[i - 3])) / 24
        y.append(y_i)
    return x, y

def runge_romberg_method(h1, h2, y1, y2, p):
    assert h1 == h2 * 2
    norm = 0
    for i in range(len(y1)):
        norm += (y1[i] - y2[i * 2]) ** 2
    return norm ** 0.5 / (2**p - 1)

def main():
    with open('input.txt', 'r') as file:
        data = [list(map(float, line.split())) for line in file.readlines()]
    y0 = data[0][0] + e
    dy0 = data[1][0]
    interval = (data[2][0], data[2][1])
    h = data[3][0]

    x_euler, y_euler = euler_method(f, g, y0, dy0, interval, h)
    _, y_euler2 = euler_method(f, g, y0, dy0, interval, h/2)
    
    x_runge, y_runge = runge_kutta_method(f, g, y0, dy0, interval, h)
    _, y_runge2 = runge_kutta_method(f, g, y0, dy0, interval, h/2)
    
    x_adams, y_adams = adams_method(f, g, y0, dy0, interval, h)
    _, y_adams2 = adams_method(f, g, y0, dy0, interval, h/2)
    
    x_exact = [i for i in np.arange(interval[0], interval[1] + h, h)]
    y_exact = [exact_solution(x_i) for x_i in x_exact]


    plt.plot(x_euler, y_euler, label=f'Euler, step={h}')
    plt.plot(x_runge, y_runge, label=f'Runge-Kutta, step={h}')
    plt.plot(x_adams, y_adams, label=f'Adams, step={h}')
    plt.plot(x_exact, y_exact, label='Exact solution')
    plt.legend()
    plt.show()

    with open('output.txt', 'w') as file:
        file.write(f"Given:\nx**2 * y'' + (x + 1) * y' - y = 0\ny(1) = 2 + e\ny\'(1) = 1\n")
        file.write(f"Converted:\ny' = g(x, y, z) = z\nz' = f(x, y, z) = (y - (x + 1) * y') / x**2\ny(1) = 2 + e\nz(1) = 1\n")
        file.write(f"Exact solution:\ny = x + 1 + x * e**(1/x)\n\n")

        file.write(f"Euler method:\n")
        file.write(f"Solution (step = {h}):\n{[float(y) for y in y_euler]}\n")
        file.write(f"Error rate (runge_romberg) = {runge_romberg_method(h, h/2, y_euler, y_euler2, 1)}\n\n")

        file.write(f"Runge-Kutta method:\n")
        file.write(f"Solution (step = {h}):\n{[float(y) for y in y_runge]}\n")
        file.write(f"Error rate (runge_romberg) = {runge_romberg_method(h, h/2, y_runge, y_runge2, 4)}\n\n")

        file.write(f"Adams method:\n")
        file.write(f"Solution (step = {h}):\n{[float(y) for y in y_adams]}\n")
        file.write(f"Error rate (runge_romberg) = {runge_romberg_method(h, h/2, y_adams, y_adams2, 4)}\n\n")

        file.write(f"Exact solution:\n{[float(y) for y in y_exact]}")

if __name__ == '__main__':
    main()