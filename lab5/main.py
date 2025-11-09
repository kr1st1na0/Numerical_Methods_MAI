import numpy as np
import matplotlib.pyplot as plt

a = 1.0

# Граничные условия
def phi0(t):
    return 0.0

def phi1(t):
    return 1.0

# Начальное условие
def init_cond(x):
    return x + np.sin(np.pi * x)

# Аналитическое решение
def analytical_solution(x, t, a=1.0):
    return x + np.exp(-np.pi**2 * a * t) * np.sin(np.pi * x)

# Метод прогонки (для трёхдиагональной системы)
def tma(a_diag, b_diag, c_diag, d_vec):
    n = len(b_diag)
    p = np.zeros(n)
    q = np.zeros(n)
    
    # Прямой ход
    p[0] = -c_diag[0] / b_diag[0]
    q[0] = d_vec[0] / b_diag[0]
    
    for i in range(1, n):
        denom = b_diag[i] + a_diag[i] * p[i - 1]
        p[i] = -c_diag[i] / denom
        q[i] = (d_vec[i] - a_diag[i] * q[i - 1]) / denom

    # Обратный ход
    x = np.zeros(n)
    x[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
        
    return x

def explicit(a, h, tau, t_range, x_range):
    t_start, t_end = t_range
    N = int(round((x_range[1] - x_range[0]) / h)) + 1
    time_steps = int(np.ceil((t_end - t_start) / tau)) + 1

    x = np.linspace(x_range[0], x_range[1], N)
    u = np.zeros((time_steps, N))
    u[0, :] = init_cond(x)

    sigma = a * tau / h**2
    if sigma > 0.5:
        print(f"Внимание: явная схема неустойчива! sigma = {sigma:.3f} > 0.5")

    for k in range(1, time_steps):
        for j in range(1, N - 1):
            u[k][j] = sigma * (u[k - 1][j + 1] + u[k - 1][j - 1]) + (1 - 2 * sigma) * u[k - 1][j]
        
        t_k = t_start + k * tau
        u[k][0] = phi0(t_k)
        u[k][-1] = phi1(t_k)

    return u, x


def implicit(a, h, tau, t_range, x_range):
    t_start, t_end = t_range
    N = int(round((x_range[1] - x_range[0]) / h)) + 1
    time_steps = int(np.ceil((t_end - t_start) / tau)) + 1

    x = np.linspace(x_range[0], x_range[1], N)
    u = np.zeros((time_steps, N))
    u[0, :] = init_cond(x)

    sigma = a * tau / h**2

    for k in range(1, time_steps):
        a_diag = np.zeros(N)
        b_diag = np.zeros(N)
        c_diag = np.zeros(N)
        d_vec  = np.zeros(N)

        # Внутренние узлы
        for j in range(1, N - 1):
            a_diag[j] = -sigma
            b_diag[j] = 1.0 + 2.0 * sigma
            c_diag[j] = -sigma
            d_vec[j]  = u[k - 1, j]

        # Граничные условия
        t_k = t_start + k * tau
        b_diag[0] = 1.0
        c_diag[0] = 0.0
        d_vec[0]  = phi0(t_k)

        a_diag[-1] = 0.0
        b_diag[-1] = 1.0
        d_vec[-1] = phi1(t_k)

        u[k, :] = tma(a_diag, b_diag, c_diag, d_vec)

    return u, x


def crank_nicolson(a, h, tau, t_range, x_range):
    t_start, t_end = t_range
    N = int(round((x_range[1] - x_range[0]) / h)) + 1
    time_steps = int(np.ceil((t_end - t_start) / tau)) + 1

    x = np.linspace(x_range[0], x_range[1], N)
    u = np.zeros((time_steps, N))
    u[0, :] = init_cond(x)

    sigma = a * tau / h**2

    for k in range(1, time_steps):
        a_diag = np.zeros(N)
        b_diag = np.zeros(N)
        c_diag = np.zeros(N)
        d_vec  = np.zeros(N)

        for j in range(1, N - 1):
            a_diag[j] = -sigma / 2.0
            b_diag[j] = 1.0 + sigma
            c_diag[j] = -sigma / 2.0
            d_vec[j] = (1.0 - sigma) * u[k - 1, j] + (sigma / 2.0) * (u[k - 1, j - 1] + u[k - 1, j + 1])

        t_k = t_start + k * tau
        b_diag[0] = 1.0
        c_diag[0] = 0.0
        d_vec[0]  = phi0(t_k)

        a_diag[-1] = 0.0
        b_diag[-1] = 1.0
        d_vec[-1] = phi1(t_k)

        u[k, :] = tma(a_diag, b_diag, c_diag, d_vec)

    return u, x

def plot_solutions(x, t_target, t_range, h, sigma, a, **solutions):
    tau = sigma * h**2 / a
    t_grid = np.arange(t_range[0], t_range[1] + tau, tau)
    k_target = np.argmin(np.abs(t_grid - t_target))
    
    plt.figure(figsize=(12, 6))
    for name, sol in solutions.items():
        plt.plot(x, sol[k_target, :], 'o', label=name)
    
    u_analytical = analytical_solution(x, t_target, a)
    plt.plot(x, u_analytical, 'k--', label='Analytical')
    
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title(f'Solutions at t = {t_target:.3f}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_vs_time(x, t_grid, a, **solutions):
    plt.figure(figsize=(10, 6))
    
    for name, u_num in solutions.items():
        errors = []
        for k, t in enumerate(t_grid):
            u_true = analytical_solution(x, t, a)
            err = np.max(np.abs(u_num[k, :] - u_true))
            errors.append(err)
        plt.plot(t_grid, errors, label=name, linewidth=2)
    
    plt.xlabel('Time $t$')
    plt.ylabel('Max absolute error')
    plt.title('Max error vs time')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

def main():
    a = 1.0
    x_range = [0.0, 1.0]
    t_range = [0.0, 0.5]

    N, K = 10, 125
    tau = (t_range[1] - t_range[0]) / K
    h = (x_range[1] - x_range[0]) / N

    # Проверка устойчивости явной схемы
    sigma = a * tau / h**2
    if sigma > 0.5:
        print('Явная схема неустойчива!')
        return 1

    # Все три схемы с одинаковыми h, tau
    u_exp, x = explicit(a, h, tau, t_range, x_range)
    u_imp, _ = implicit(a, h, tau, t_range, x_range)
    u_cn,  _ = crank_nicolson(a, h, tau, t_range, x_range)

    # Согласуем количество временных шагов
    min_steps = min(u_exp.shape[0], u_imp.shape[0], u_cn.shape[0])
    u_exp = u_exp[:min_steps]
    u_imp = u_imp[:min_steps]
    u_cn  = u_cn[:min_steps]

    t_grid = np.linspace(t_range[0], t_range[0] + (min_steps - 1) * tau, min_steps)

    # Графики решений
    for t_target in [0.05, 0.1, 0.25]:
        plot_solutions(
            x, t_target, t_range, h, sigma, a,
            Explicit=u_exp,
            Implicit=u_imp,
            Crank_Nicolson=u_cn
        )

    # График ошибок
    plot_error_vs_time(
        x, t_grid, a,
        Explicit=u_exp,
        Implicit=u_imp,
        Crank_Nicolson=u_cn
    )

'''
def compute_error_at_final_time(u_num, x, t_final, a):
    u_true = analytical_solution(x, t_final, a)
    return np.max(np.abs(u_num[-1, :] - u_true))


def convergence_in_time(a, h_fixed, t_range, tau_values, x_range):
    methods = {
        'Explicit': explicit,
        'Implicit': implicit,
        'Crank-Nicolson': crank_nicolson
    }
    errors = {name: [] for name in methods}
    valid_taus = []

    t_final = t_range[1]
    
    for tau in tau_values:
        # Проверка устойчивости явной схемы
        sigma = a * tau / h_fixed**2
        valid_taus.append(tau)
        
        for name, solver in methods.items():
            if name == 'Explicit' and sigma > 0.5:
                errors[name].append(np.nan)
                continue
            
            try:
                u, x = solver(a, h_fixed, tau, t_range, x_range)
                err = compute_error_at_final_time(u, x, t_final, a)
                errors[name].append(err)
            except Exception as e:
                print(f"Ошибка в {name} при tau={tau:.2e}: {e}")
                errors[name].append(np.nan)
    
    return np.array(valid_taus), {k: np.array(v) for k, v in errors.items()}


def convergence_in_space(a, sigma_fixed, t_range, h_values, x_range):
    methods = {
        'Explicit': explicit,
        'Implicit': implicit,
        'Crank-Nicolson': crank_nicolson
    }
    errors = {name: [] for name in methods}
    valid_hs = []

    t_final = t_range[1]
    
    for h in h_values:
        tau = sigma_fixed * h**2 / a
        sigma = a * tau / h**2
        valid_hs.append(h)
        
        for name, solver in methods.items():
            if name == 'Explicit' and sigma > 0.5:
                errors[name].append(np.nan)
                continue
            
            try:
                u, x = solver(a, h, tau, t_range, x_range)
                err = compute_error_at_final_time(u, x, t_final, a)
                errors[name].append(err)
            except Exception as e:
                print(f"Ошибка в {name} при h={h:.2e}, tau={tau:.2e}: {e}")
                errors[name].append(np.nan)
    
    return np.array(valid_hs), {k: np.array(v) for k, v in errors.items()}

def plot_convergence(taus, err_tau_dict, hs, err_h_dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {
        'Explicit': 'red',
        'Implicit': 'blue',
        'Crank-Nicolson': 'green'
    }
    markers = {
        'Explicit': 'o',
        'Implicit': 's',
        'Crank-Nicolson': '^'
    }

    # Сходимость по времени
    for method in ['Explicit', 'Implicit', 'Crank-Nicolson']:
        err = err_tau_dict[method]
        valid = ~np.isnan(err)
        if np.any(valid):
            ax1.plot(taus[valid], err[valid], 
                      marker=markers[method], color=colors[method], 
                      label=method, markersize=6, linewidth=1.5)

    # Опорные линии
    # O(τ) — для явной и неявной
    valid_imp = ~np.isnan(err_tau_dict['Implicit'])
    if np.any(valid_imp):
        tau_ref = taus[valid_imp]
        err_ref = err_tau_dict['Implicit'][valid_imp]
        C1 = err_ref[0] / tau_ref[0]
        ax1.plot(tau_ref, C1 * tau_ref, 'k--', label=r'$O(\tau)$')

    # O(τ²) — для Кранка–Николсона
    valid_cn = ~np.isnan(err_tau_dict['Crank-Nicolson'])
    if np.any(valid_cn):
        tau_cn = taus[valid_cn]
        err_cn = err_tau_dict['Crank-Nicolson'][valid_cn]
        C2 = err_cn[0] / (tau_cn[0] ** 2)
        ax1.plot(tau_cn, C2 * tau_cn**2, 'k-.', label=r'$O(\tau^2)$')
    
    ax1.set_xlabel(r'Time step $\tau$')
    ax1.set_ylabel('Max error at final time')
    ax1.set_title('Convergence in time (fixed $h$)')
    ax1.grid(True, which="both", ls=":", linewidth=0.5)
    ax1.legend()

    # Сходимость по пространству
    for method in ['Explicit', 'Implicit', 'Crank-Nicolson']:
        err = err_h_dict[method]
        valid = ~np.isnan(err)
        if np.any(valid):
            ax2.plot(hs[valid], err[valid], 
                      marker=markers[method], color=colors[method], 
                      label=method, markersize=6, linewidth=1.5)

    # Опорная линия O(h²)
    valid_cn_h = ~np.isnan(err_h_dict['Crank-Nicolson'])
    if np.any(valid_cn_h):
        h_ref = hs[valid_cn_h]
        err_ref = err_h_dict['Crank-Nicolson'][valid_cn_h]
        C = err_ref[0] / (h_ref[0] ** 2)
        ax2.plot(h_ref, C * h_ref**2, 'k--', label=r'$O(h^2)$')
    
    ax2.set_xlabel(r'Space step $h$')
    ax2.set_ylabel('Max error at final time')
    ax2.set_title('Convergence in space (fixed $tau$)')
    ax2.grid(True, which="both", ls=":", linewidth=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()

def main_convergence():
    a = 1.0
    x_range = [0.0, 1.0]
    t_range = [0.0, 0.5]

    # Зависимость от шага по времени
    N_fixed = 10
    K_values =  np.array([100, 125, 150, 200])
    h_fixed = (x_range[1] - x_range[0]) / N_fixed
    tau_values = (t_range[1] - t_range[0]) / K_values

    taus, err_tau_dict = convergence_in_time(a, h_fixed, t_range, tau_values, x_range)


    # Зависимость от шага по пространству
    K_fixed = 125
    N_values =  np.array([5, 10, 15, 20])
    tau_fixed = (t_range[1] - t_range[0]) / K_fixed
    h_values = (x_range[1] - x_range[0]) / N_values


    hs, err_h_dict = convergence_in_space(a, tau_fixed, t_range, h_values, x_range)

    plot_convergence(taus, err_tau_dict, hs, err_h_dict)
'''

if __name__ == "__main__":
    main()
    # main_convergence()