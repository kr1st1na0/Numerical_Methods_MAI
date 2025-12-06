import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

plt.rcParams['figure.figsize'] = [8, 7]
a = 1

def phi1(y, t, mu1, mu2):  # x = 0
    return np.cos(mu2 * y) * np.exp(-(mu1**2 + mu2**2) * a * t)

def phi2(y, t, mu1, mu2):  # x = Lx
    Lx = mu1 * np.pi / 2
    return np.cos(mu1 * Lx) * np.cos(mu2 * y) * np.exp(-(mu1**2 + mu2**2) * a * t)

def phi3(x, t, mu1, mu2):  # y = 0
    return np.cos(mu1 * x) * np.exp(-(mu1**2 + mu2**2) * a * t)

def phi4(x, t, mu1, mu2):  # y = Ly
    Ly = mu2 * np.pi / 2
    return np.cos(mu1 * x) * np.cos(mu2 * Ly) * np.exp(-(mu1**2 + mu2**2) * a * t)

def psi(x, y, mu1, mu2):
    return np.cos(mu1 * x) * np.cos(mu2 * y)

def U(x, y, t, mu1, mu2):
    return np.cos(mu1 * x) * np.cos(mu2 * y) * np.exp(-(mu1**2 + mu2**2) * a * t)


def Check(A):
    if A.shape[0] != A.shape[1]:
        return False
    n = A.shape[0]
    for i in range(n):
        diag = abs(A[i, i])
        off_diag = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diag < off_diag:
            return False
    return True


def solve(A, b):
    if Check(A):
        n = len(b)
        p = np.zeros(n)
        q = np.zeros(n)
        p[0] = -A[0, 1] / A[0, 0]
        q[0] = b[0] / A[0, 0]
        for i in range(1, n - 1):
            denom = A[i, i] + A[i, i - 1] * p[i - 1]
            p[i] = -A[i, i + 1] / denom
            q[i] = (b[i] - A[i, i - 1] * q[i - 1]) / denom
        denom = A[n - 1, n - 1] + A[n - 1, n - 2] * p[n - 2]
        q[n - 1] = (b[n - 1] - A[n - 1, n - 2] * q[n - 2]) / denom
        x = np.zeros(n)
        x[-1] = q[-1]
        for i in range(n - 2, -1, -1):
            x[i] = p[i] * x[i + 1] + q[i]
        return x
    else:
        return np.linalg.solve(A, b)


def VariableDirectionMethod(Nt, Nx, Ny, tau, hx, hy, mu1, mu2):
    Lx = mu1 * np.pi / 2
    Ly = mu2 * np.pi / 2
    u = np.zeros((Nt + 1, Nx + 1, Ny + 1))
    x_grid = np.linspace(0, Lx, Nx + 1)
    y_grid = np.linspace(0, Ly, Ny + 1)

    # Начальное условие
    for i in range(Nx + 1):
        for j in range(Ny + 1):
            u[0, i, j] = psi(x_grid[i], y_grid[j], mu1, mu2)  # ИСПРАВЛЕНО: порядок аргументов

    # Граничные условия
    for k in range(Nt + 1):
        t = k * tau
        u[k, :, 0] = phi3(x_grid, t, mu1, mu2)   # y = 0
        u[k, :, -1] = phi4(x_grid, t, mu1, mu2)  # y = Ly
        u[k, 0, :] = phi1(y_grid, t, mu1, mu2)   # x = 0
        u[k, -1, :] = phi2(y_grid, t, mu1, mu2)  # x = Lx

    for k in range(Nt):
        tmp = deepcopy(u[k])
        t_half = (k + 0.5) * tau
        t_next = (k + 1) * tau

        # Первый дробный шаг (по x)
        for j in range(1, Ny):
            A = np.zeros((Nx - 1, Nx - 1))
            d = np.zeros(Nx - 1)
            sigma_x = a * tau / (2 * hx * hx)
            sigma_y = a * tau / (2 * hy * hy)

            # Граничные узлы
            A[0, 0] = 1 + 2 * sigma_x
            if Nx > 2:
                A[0, 1] = -sigma_x
            d[0] = (u[k, 1, j] +
                    sigma_y * (u[k, 1, j - 1] - 2 * u[k, 1, j] + u[k, 1, j + 1]) +
                    sigma_x * phi1(y_grid[j], t_half, mu1, mu2))

            for i in range(1, Nx - 2):
                A[i, i - 1] = -sigma_x
                A[i, i] = 1 + 2 * sigma_x
                A[i, i + 1] = -sigma_x
                d[i] = (u[k, i + 1, j] +
                        sigma_y * (u[k, i + 1, j - 1] - 2 * u[k, i + 1, j] + u[k, i + 1, j + 1]))

            if Nx > 2:
                A[-1, -2] = -sigma_x
                A[-1, -1] = 1 + 2 * sigma_x
                d[-1] = (u[k, -2, j] +
                         sigma_y * (u[k, -2, j - 1] - 2 * u[k, -2, j] + u[k, -2, j + 1]) +
                         sigma_x * phi2(y_grid[j], t_half, mu1, mu2))

            sol = solve(A, d)
            tmp[1:-1, j] = sol

        # Обновим границы tmp
        tmp[:, 0] = phi3(x_grid, t_half, mu1, mu2)
        tmp[:, -1] = phi4(x_grid, t_half, mu1, mu2)
        tmp[0, :] = phi1(y_grid, t_half, mu1, mu2)
        tmp[-1, :] = phi2(y_grid, t_half, mu1, mu2)

        # Второй дробный шаг (по y)
        for i in range(1, Nx):
            A = np.zeros((Ny - 1, Ny - 1))
            d = np.zeros(Ny - 1)
            sigma_x = a * tau / (2 * hx * hx)
            sigma_y = a * tau / (2 * hy * hy)

            A[0, 0] = 1 + 2 * sigma_y
            if Ny > 2:
                A[0, 1] = -sigma_y
            d[0] = (tmp[i, 1] +
                    sigma_x * (tmp[i - 1, 1] - 2 * tmp[i, 1] + tmp[i + 1, 1]) +
                    sigma_y * phi3(x_grid[i], t_next, mu1, mu2))

            for j in range(1, Ny - 2):
                A[j, j - 1] = -sigma_y
                A[j, j] = 1 + 2 * sigma_y
                A[j, j + 1] = -sigma_y
                d[j] = (tmp[i, j + 1] +
                        sigma_x * (tmp[i - 1, j + 1] - 2 * tmp[i, j + 1] + tmp[i + 1, j + 1]))

            if Ny > 2:
                A[-1, -2] = -sigma_y
                A[-1, -1] = 1 + 2 * sigma_y
                d[-1] = (tmp[i, -2] +
                         sigma_x * (tmp[i - 1, -2] - 2 * tmp[i, -2] + tmp[i + 1, -2]) +
                         sigma_y * phi4(x_grid[i], t_next, mu1, mu2))

            sol = solve(A, d)
            u[k + 1, i, 1:-1] = sol

        # Обновим границы u[k+1]
        u[k + 1, :, 0] = phi3(x_grid, t_next, mu1, mu2)
        u[k + 1, :, -1] = phi4(x_grid, t_next, mu1, mu2)
        u[k + 1, 0, :] = phi1(y_grid, t_next, mu1, mu2)
        u[k + 1, -1, :] = phi2(y_grid, t_next, mu1, mu2)

    return u


def FractionalStepsMethod(Nt, Nx, Ny, tau, hx, hy, mu1, mu2):
    Lx = mu1 * np.pi / 2
    Ly = mu2 * np.pi / 2
    u = np.zeros((Nt + 1, Nx + 1, Ny + 1))
    x_grid = np.linspace(0, Lx, Nx + 1)
    y_grid = np.linspace(0, Ly, Ny + 1)

    for i in range(Nx + 1):
        for j in range(Ny + 1):
            u[0, i, j] = psi(x_grid[i], y_grid[j], mu1, mu2)

    for k in range(Nt + 1):
        t = k * tau
        u[k, :, 0] = phi3(x_grid, t, mu1, mu2)
        u[k, :, -1] = phi4(x_grid, t, mu1, mu2)
        u[k, 0, :] = phi1(y_grid, t, mu1, mu2)
        u[k, -1, :] = phi2(y_grid, t, mu1, mu2)

    for k in range(Nt):
        tmp = deepcopy(u[k])
        t_half = (k + 0.5) * tau
        t_next = (k + 1) * tau

        # Первый шаг (по x)
        r = a * tau / (hx * hx)
        for j in range(1, Ny):
            A = np.zeros((Nx - 1, Nx - 1))
            d = np.zeros(Nx - 1)
            A[0, 0] = 1 + 2 * r
            if Nx > 2:
                A[0, 1] = -r
            d[0] = u[k, 1, j] + r * phi1(y_grid[j], t_half, mu1, mu2)

            for i in range(1, Nx - 2):
                A[i, i - 1] = -r
                A[i, i] = 1 + 2 * r
                A[i, i + 1] = -r
                d[i] = u[k, i + 1, j]

            if Nx > 2:
                A[-1, -2] = -r
                A[-1, -1] = 1 + 2 * r
                d[-1] = u[k, -2, j] + r * phi2(y_grid[j], t_half, mu1, mu2)

            sol = solve(A, d)
            tmp[1:-1, j] = sol

        tmp[:, 0] = phi3(x_grid, t_half, mu1, mu2)
        tmp[:, -1] = phi4(x_grid, t_half, mu1, mu2)
        tmp[0, :] = phi1(y_grid, t_half, mu1, mu2)
        tmp[-1, :] = phi2(y_grid, t_half, mu1, mu2)

        # Второй шаг (по y)
        r = a * tau / (hy * hy)
        for i in range(1, Nx):
            A = np.zeros((Ny - 1, Ny - 1))
            d = np.zeros(Ny - 1)
            A[0, 0] = 1 + 2 * r
            if Ny > 2:
                A[0, 1] = -r
            d[0] = tmp[i, 1] + r * phi3(x_grid[i], t_next, mu1, mu2)

            for j in range(1, Ny - 2):
                A[j, j - 1] = -r
                A[j, j] = 1 + 2 * r
                A[j, j + 1] = -r
                d[j] = tmp[i, j + 1]

            if Ny > 2:
                A[-1, -2] = -r
                A[-1, -1] = 1 + 2 * r
                d[-1] = tmp[i, -2] + r * phi4(x_grid[i], t_next, mu1, mu2)

            sol = solve(A, d)
            u[k + 1, i, 1:-1] = sol

        u[k + 1, :, 0] = phi3(x_grid, t_next, mu1, mu2)
        u[k + 1, :, -1] = phi4(x_grid, t_next, mu1, mu2)
        u[k + 1, 0, :] = phi1(y_grid, t_next, mu1, mu2)
        u[k + 1, -1, :] = phi2(y_grid, t_next, mu1, mu2)

    return u


def show_solution(Nx, Ny, Nt, hx, hy, tau, u_adi, u_fs, mu1, mu2):
    Lx = mu1 * np.pi / 2
    Ly = mu2 * np.pi / 2
    x_array = np.linspace(0, Lx, Nx + 1)
    y_array = np.linspace(0, Ly, Ny + 1)
    
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    t_indices = [int(Nt * 0.05), int(Nt * 0.4), int(Nt * 0.7)]
    x_fix = Nx // 2
    y_fix = Ny // 4

    for col, k in enumerate(t_indices):
        t_val = k * tau

        # === Сечение по x (фиксированный y) ===
        u_exact_x = np.array([U(x, y_array[y_fix], t_val, mu1, mu2) for x in x_array])
        u_adi_x = u_adi[k, :, y_fix]
        u_fs_x = u_fs[k, :, y_fix]

        ax[0, col].plot(x_array, u_exact_x, 'k-', label='Analytical')
        ax[0, col].plot(x_array, u_adi_x, 'b--', label='VariableDirections')
        ax[0, col].plot(x_array, u_fs_x, 'r-.', label='FractionalSteps')
        ax[0, col].set_xlabel('x')
        ax[0, col].set_ylabel('U(x, y, t)')
        ax[0, col].set_title(f'x (y={y_array[y_fix]:.2f})\n t = {t_val:.3f}')
        ax[0, col].grid(True)
        ax[0, col].legend()

        # === Сечение по y (фиксированный x) ===
        u_exact_y = np.array([U(x_array[x_fix], y, t_val, mu1, mu2) for y in y_array])
        u_adi_y = u_adi[k, x_fix, :]
        u_fs_y = u_fs[k, x_fix, :]

        ax[1, col].plot(y_array, u_exact_y, 'k-', label='Analytical')
        ax[1, col].plot(y_array, u_adi_y, 'b--', label='VariableDirections')
        ax[1, col].plot(y_array, u_fs_y, 'r-.', label='FractionalSteps')
        ax[1, col].set_xlabel('y')
        ax[1, col].set_ylabel('U(x, y, t)')
        ax[1, col].set_title(f'y (x={x_array[x_fix]:.2f})\n t = {t_val:.3f}')
        ax[1, col].grid(True)
        ax[1, col].legend()

    plt.tight_layout()
    plt.show()


def error(Nt, Lx, Ly, tau, mu1, mu2):
    N_array = [10, 20, 40]
    errors1x = []
    errors2x = []
    for N in N_array:
        hx = Lx / N
        hy = Ly / N
        # Методы
        u1 = VariableDirectionMethod(Nt, N, N, tau, hx, hy, mu1, mu2)
        u2 = FractionalStepsMethod(Nt, N, N, tau, hx, hy, mu1, mu2)
        # Точка для ошибки
        t = tau * Nt / 2
        x_mid = Lx / 2
        y_mid = Ly / 2
        # Сечение по x
        ux_exact = np.array([U(x_i * hx, y_mid, t, mu1, mu2) for x_i in range(N + 1)])
        u1x = u1[int(Nt / 2), :, int(N / 2)]
        u2x = u2[int(Nt / 2), :, int(N / 2)]
        errors1x.append(np.max(np.abs(ux_exact - u1x)))
        errors2x.append(np.max(np.abs(ux_exact - u2x)))
    return N_array, np.array(errors1x), np.array(errors2x)


def show_errors(Nt, Lx, Ly, tau, mu1, mu2):
    N_array, errors1x, errors2x = error(Nt, Lx, Ly, tau, mu1, mu2)
    delta_x = np.array([Lx / N for N in N_array])
    plt.figure()
    plt.plot(delta_x, errors1x, 'b-', label='VariableDirections')
    plt.plot(delta_x, errors2x, 'r--', label='FractionalSteps')
    plt.xlabel('delta X')
    plt.ylabel('Epsilon')
    plt.title('Error')
    plt.grid()
    plt.legend()
    plt.show()


def plot_3d_solution(x_grid, y_grid, u_adi, u_fs, mu1, mu2, T, case_name):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

    # Аналитическое решение в момент времени T
    U_anal = np.array([[U(x, y, T, mu1, mu2) for y in y_grid] for x in x_grid])
    U_adi = u_adi[-1]  # Последний временной слой
    U_fs = u_fs[-1]

    fig = plt.figure(figsize=(18, 5))

    # Аналитическое
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, U_anal, cmap='viridis', alpha=0.9)
    ax1.set_title(f'{case_name}\nAnalytical')
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('u')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    # Метод переменных направлений (ADI)
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, U_adi, cmap='viridis', alpha=0.9)
    ax2.set_title(f'{case_name}\nVariableDirections')
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('u')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)

    # Метод дробных шагов (Fractional Steps)
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, U_fs, cmap='viridis', alpha=0.9)
    ax3.set_title(f'{case_name}\nFractionalSteps')
    ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('u')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)

    plt.tight_layout()
    plt.show()


def compute_error_at_time(u_num, x_grid, y_grid, t_val, mu1, mu2):
    """Вычисляет max-норму ошибки между численным и аналитическим решением в момент t_val."""
    Nx = len(x_grid) - 1
    Ny = len(y_grid) - 1
    u_exact = np.zeros_like(u_num)
    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            u_exact[i, j] = U(x, y, t_val, mu1, mu2)
    return np.max(np.abs(u_exact - u_num))


def error_h_dependence(mu1, mu2, T, tau_fixed, N_array):
    """Зависимость ошибки от h = hx = hy при фиксированном tau."""
    Lx = mu1 * np.pi / 2
    Ly = mu2 * np.pi / 2
    errors_adi = []
    errors_fs = []
    h_array = []

    for N in N_array:
        hx = Lx / N
        hy = Ly / N
        Nt = int(T / tau_fixed)
        if Nt == 0:
            Nt = 1
        tau = T / Nt  # корректируем tau, чтобы точно попасть в T

        x_grid = np.linspace(0, Lx, N + 1)
        y_grid = np.linspace(0, Ly, N + 1)

        u_adi = VariableDirectionMethod(Nt, N, N, tau, hx, hy, mu1, mu2)
        u_fs = FractionalStepsMethod(Nt, N, N, tau, hx, hy, mu1, mu2)

        err_adi = compute_error_at_time(u_adi[-1], x_grid, y_grid, T, mu1, mu2)
        err_fs = compute_error_at_time(u_fs[-1], x_grid, y_grid, T, mu1, mu2)

        errors_adi.append(err_adi)
        errors_fs.append(err_fs)
        h_array.append(hx)  # hx = hy

    return np.array(h_array), np.array(errors_adi), np.array(errors_fs)


def error_tau_dependence(mu1, mu2, T, h_fixed, tau_array):
    """Зависимость ошибки от tau при фиксированном h."""
    Lx = mu1 * np.pi / 2
    Ly = mu2 * np.pi / 2
    Nx = int(Lx / h_fixed)
    Ny = int(Ly / h_fixed)
    if Nx == 0: Nx = 1
    if Ny == 0: Ny = 1
    hx = Lx / Nx
    hy = Ly / Ny

    errors_adi = []
    errors_fs = []

    for tau in tau_array:
        Nt = int(np.ceil(T / tau))
        tau = T / Nt  # точное деление

        x_grid = np.linspace(0, Lx, Nx + 1)
        y_grid = np.linspace(0, Ly, Ny + 1)

        u_adi = VariableDirectionMethod(Nt, Nx, Ny, tau, hx, hy, mu1, mu2)
        u_fs = FractionalStepsMethod(Nt, Nx, Ny, tau, hx, hy, mu1, mu2)

        err_adi = compute_error_at_time(u_adi[-1], x_grid, y_grid, T, mu1, mu2)
        err_fs = compute_error_at_time(u_fs[-1], x_grid, y_grid, T, mu1, mu2)

        errors_adi.append(err_adi)
        errors_fs.append(err_fs)

    return np.array(tau_array[:len(errors_adi)]), np.array(errors_adi), np.array(errors_fs)


def plot_error_vs_h(mu1, mu2, T, tau_fixed, N_array, case_name):
    h_vals, err_adi, err_fs = error_h_dependence(mu1, mu2, T, tau_fixed, N_array)
    plt.figure(figsize=(8, 6))
    plt.loglog(h_vals, err_adi, 'bo-', label='МПН (ADI)')
    plt.loglog(h_vals, err_fs, 'rs-', label='Дробные шаги')
    plt.xlabel(r'$h = h_x = h_y$')
    plt.ylabel(r'Макс. погрешность $\varepsilon$')
    plt.title(f'Зависимость погрешности от $h$ ({case_name})\n(τ = {tau_fixed})')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()


def plot_error_vs_tau(mu1, mu2, T, h_fixed, tau_array, case_name):
    tau_vals, err_adi, err_fs = error_tau_dependence(mu1, mu2, T, h_fixed, tau_array)
    plt.figure(figsize=(8, 6))
    plt.loglog(tau_vals, err_adi, 'bo-', label='МПН (ADI)')
    plt.loglog(tau_vals, err_fs, 'rs-', label='Дробные шаги')
    plt.xlabel(r'Шаг по времени $\tau$')
    plt.ylabel(r'Макс. погрешность $\varepsilon$')
    plt.title(f'Зависимость погрешности от $\\tau$ ({case_name})\n(h = {h_fixed:.4f})')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()


def main():
    cases = [
        (1, 1, "μ₁=1, μ₂=1"),
        (2, 1, "μ₁=2, μ₂=1"),
        (1, 2, "μ₁=1, μ₂=2")
    ]
    
    Nx = Ny = 51
    Nt = 50
    T = 0.1

    # Параметры для графиков сходимости
    N_array_h = [10, 20, 40]
    tau_fixed = 0.002  # фиксированный шаг по времени

    tau_array = [0.01, 0.005, 0.002, 0.001]
    h_fixed = 0.1  # фксированный шаг по пространству

    for mu1, mu2, case_name in cases:
        print(f"Случай: {case_name}\n")
        
        Lx = mu1 * np.pi / 2
        Ly = mu2 * np.pi / 2
        hx = Lx / Nx
        hy = Ly / Ny
        tau = T / Nt

        u_adi = VariableDirectionMethod(Nt, Nx, Ny, tau, hx, hy, mu1, mu2)
        u_fs = FractionalStepsMethod(Nt, Nx, Ny, tau, hx, hy, mu1, mu2)
        
        show_solution(Nx, Ny, Nt, hx, hy, tau, u_adi, u_fs, mu1, mu2)

        show_errors(Nt, Lx, Ly, tau, mu1, mu2)

        x_grid = np.linspace(0, Lx, Nx + 1)
        y_grid = np.linspace(0, Ly, Ny + 1)
        plot_3d_solution(x_grid, y_grid, u_adi, u_fs, mu1, mu2, T, case_name)

        # Исследование сходимости
        plot_error_vs_h(mu1, mu2, T, tau_fixed, N_array_h, case_name)
        plot_error_vs_tau(mu1, mu2, T, h_fixed, tau_array, case_name)


if __name__ == "__main__":
    main()