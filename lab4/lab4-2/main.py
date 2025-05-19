from math import atan, pi
import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_method(f, g, y0, z0, interval, h):
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

    return x, y, z

def tridiagonal_solve(A, b):
    n = len(A)
    # Step 1. Forward
    v = [0 for _ in range(n)]
    u = [0 for _ in range(n)]
    v[0] = A[0][1] / -A[0][0]
    u[0] = b[0] / A[0][0]
    for i in range(1, n-1):
        v[i] = A[i][i+1] / (-A[i][i] - A[i][i-1] * v[i-1])
        u[i] = (A[i][i-1] * u[i-1] - b[i]) / (-A[i][i] - A[i][i-1] * v[i-1])
    v[n-1] = 0
    u[n-1] = (A[n-1][n-2] * u[n-2] - b[n-1]) / (-A[n-1][n-1] - A[n-1][n-2] * v[n-2])

    # Step 2. Backward
    x = [0 for _ in range(n)]
    x[n-1] = u[n-1]
    for i in range(n-1, 0, -1):
        x[i-1] = v[i-1] * x[i] + u[i-1]
    return x

def f(x, y, z):
    return 2 * y / (x**2 + 1)

def g(x, y, z):
    return z

# y'' + p_fd(x)y' + q_fd(x)y = f_fd(x)

def p_fd(x):
    return 0

def q_fd(x):
    return -2 / (x**2 + 1)

def f_fd(x):
    return 0

def exact_solution(x):
    return x**2 + x + 1 + (x**2 + 1) * atan(x)

def shooting_method(f, g, dy0, y1_target, interval, h, eps=1e-6, max_iter=100):
    n_prev = 0.0
    n = 1.0
    for iter_count in range(max_iter):
        x, y1, z1 = runge_kutta_method(f, g, n_prev, dy0, interval, h)
        val1 = y1[-1]  # y(1)

        x, y2, z2 = runge_kutta_method(f, g, n, dy0, interval, h)
        val2 = y2[-1]

        if abs(val1 - y1_target) < eps:
            return x, y1, z1, n_prev, iter_count
        if abs(val2 - y1_target) < eps:
            return x, y2, z2, n, iter_count

        slope = (val2 - val1) / (n - n_prev)
        n_new = n - (val2 - y1_target) / slope
 
        n_prev, n = n, n_new
        val1, val2 = val2, val1

    x, y, z = runge_kutta_method(f, g, n, dy0, interval, h)
    return x, y, z, n, max_iter

def finite_difference_method(p, q, f, y0, yn, interval, h):
    A = []
    B = []
    rows = []
    a, b = interval
    x = np.arange(a, b + h, h)
    n = len(x)

    # Creating tridiagonal matrix
    for i in range(n):
        if i == 0:
            rows.append(1)
        else:
            rows.append(0)
    A.append(rows)
    B.append(y0)

    for i in range(1, n - 1):
        rows = []
        B.append(f(x[i]))
        for j in range(n):
            if j == i - 1:
                rows.append(1 / h ** 2 - p(x[i]) / (2 * h))
            elif j == i:
                rows.append(-2 / h ** 2 + q(x[i]))
            elif j == i + 1:
                rows.append(1 / h ** 2 + p(x[i]) / (2 * h))
            else:
                rows.append(0)
        A.append(rows)

    rows = []
    B.append(yn)
    for i in range(n):
        if i == n - 1:
            rows.append(1)
        else:
            rows.append(0)

    A.append(rows)
    y = tridiagonal_solve(A, B)
    return x, y

def runge_romberg_method(h1, h2, y1, y2, p):
    assert h1 == h2 * 2
    norm = 0
    for i in range(len(y1)):
        norm += (y1[i] - y2[i * 2]) ** 2
    return norm ** 0.5 / (2 ** p - 1)

def main():
    with open('input.txt', 'r') as file:
        data = [list(map(float, line.split())) for line in file.readlines()]
    y0 = data[0][0]
    y1 = data[1][0] + pi / 2
    interval = (data[2][0], data[2][1])
    h = data[3][0]

    x_shooting, y_shooting, z_shooting, y01, iter_count1 = shooting_method(f, g, y0, y1, interval, h)
    plt.plot(x_shooting, y_shooting, label=f'shooting method, step={h}')
    x_shooting2, y_shooting2, z_shooting2, y02, iter_count2 = shooting_method(f, g, y0, y1, interval, h / 2)

    x_fd, y_fd = finite_difference_method(p_fd, q_fd, f_fd, y01, y1, interval, h)
    plt.plot(x_fd, y_fd, label=f'finite difference method, step={h}')
    x_fd2, y_fd2 = finite_difference_method(p_fd, q_fd, f_fd, y02, y1, interval, h / 2)

    x_exact = [i for i in np.arange(interval[0], interval[1] + h, h)]
    y_exact = [exact_solution(x_i) for x_i in x_exact]
    plt.plot(x_exact, y_exact, label='exact solution')

    with open('output.txt', 'w') as file:
        file.write(f"Given:\n(x**2 + 1) * y'' -2 * y = 0\ny\'(0) = 2\ny(1) = 3 + pi/2\n")
        file.write(f"Converted:\ny' = g(x, y, z) = z\nz' = f(x, y, z) = 2 * y / (x**2 + 1) / x**2\ny(1) = 3 + pi/2\nz(0) = 2\n")
        file.write(f"Exact solution:\nx**2 + x + 1 + (x**2 + 1) * arctg(x)\n\n")

        file.write(f"Shooting:\n")
        file.write(f"X values (step = {h}):\n{[float(x) for x in x_shooting]}\n")
        file.write(f"Solution (step = {h}):\n{[float(y) for y in y_shooting]}\n")
        file.write(f"Iter_count: {iter_count1}\n")
        file.write(f"Error rate (runge_romberg) = {runge_romberg_method(h, h/2, y_shooting, y_shooting2, 4)}\n\n")

        file.write(f"Finite Difference method:\n")
        file.write(f"X values (step = {h}):\n{[float(x) for x in x_fd]}\n")
        file.write(f"Solution (step = {h}):\n{[float(y) for y in y_fd]}\n")
        file.write(f"Error rate (runge_romberg) = {runge_romberg_method(h, h/2, y_fd, y_fd2, 2)}\n\n")

        file.write(f"Exact solution:\n{[float(y) for y in y_exact]}")

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()