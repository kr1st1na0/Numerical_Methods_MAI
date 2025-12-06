import numpy as np
import matplotlib.pyplot as plt

a = 1.0

# Граничные условия
u_0 = 0
u_l = 0

# Начальные условия
def init_cond(x):
    return np.sin(x) + np.cos(x) # первое начальное условие

def d_init_cond(x, tau, a=1.0, method='first_order'):
    u0 = init_cond(x)
    ut0 = -a * (np.sin(x) + np.cos(x))  # второе начальное условие
    
    if method == 'first_order':
        # Аппроксимация 1-го порядка: u(x,τ) = u(x,0) + τ * u_t(x,0)
        return u0 + tau * ut0
    elif method == 'second_order':
        # Аппроксимация 2-го порядка: u(x,τ) = u(x,0) + τ * u_t(x,0) + (τ²/2) * u_tt(x,0)
        # u_tt(x,0) = a² * u_xx(x,0)
        uxx0 = -np.sin(x) - np.cos(x)  # вторая производная u(x,0)
        utt0 = a**2 * uxx0
        return u0 + tau * ut0 + 0.5 * tau**2 * utt0

# Аналитическое решение
def analytical_solution(x, t, a=1.0):
    return np.sin(x - a * t) + np.cos(x + a * t)

# Метод прогонки (для трёхдиагональной системы)
def tma(a, b):
    n = len(b)
    p = np.zeros(n)
    q = np.zeros(n)
    # Прямой ход
    p[0] = -a[0][1] / a[0][0]
    q[0] = b[0] / a[0][0]

    for i in range(1, len(p) - 1):
        p[i] = -a[i][i + 1] / (a[i][i] + a[i][i - 1] * p[i - 1])
        q[i] = (b[i] - a[i][i - 1] * q[i - 1]) / (a[i][i] + a[i][i - 1] * p[i - 1])
    i = len(a) - 1
    p[-1] = 0
    q[-1] = (b[-1] - a[-1][-2] * q[-2]) / (a[-1][-1] + a[-1][-2] * p[-2])
    
    # Обратный ход
    x = np.zeros(n)
    x[-1] = q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]
    return x

def apply_boundary_conditions(u_current, h, method='two_point_first_order'):
    N = len(u_current)
    
    if method == 'two_point_first_order':
        # Двухточечная аппроксимация 1-го порядка
        # u_x = (u[1] - u[0])/h = u[0] => u[0] = u[1]/(1 + h)
        u_current[0] = u_current[1] / (1 + h)
        # u_x = (u[N-1] - u[N-2])/h = u[N-1] => u[N-1] = u[N-2]/(1 - h)
        u_current[N-1] = u_current[N-2] / (1 - h)
        
    elif method == 'three_point_second_order':
        # Трехточечная аппроксимация 2-го порядка
        # u_x = (-3u[0] + 4u[1] - u[2])/(2h) = u[0]
        # => u[0] = (4u[1] - u[2])/(3 + 2h)
        u_current[0] = (4 * u_current[1] - u_current[2]) / (3 + 2 * h)
        # u_x = (3u[N-1] - 4u[N-2] + u[N-3])/(2h) = u[N-1]
        # => u[N-1] = (4u[N-2] - u[N-3])/(3 - 2h)
        u_current[N-1] = (4 * u_current[N-2] - u_current[N-3]) / (3 - 2 * h)
        
    elif method == 'two_point_second_order':
        # Двухточечная аппроксимация 2-го порядка
        u_current[0] = u_current[1] / (1 + h)
        u_current[N-1] = u_current[N-2] / (1 - h)
        
    return u_current

def explicit(h, tau, t_range, x_range, boundary_method='two_point_first_order', init_approx='first_order'):
    t_start, t_end = t_range
    N = int(round((x_range[1] - x_range[0]) / h)) + 1
    time_steps = int(np.ceil((t_end - t_start) / tau)) + 1

    x = np.linspace(x_range[0], x_range[1], N)
    u = np.zeros((time_steps, N))
    u[0, :] = init_cond(x)
    
    sigma = a**2 * tau**2 / h**2
    if sigma > 1:
        print(f"Внимание: явная схема неустойчива! sigma = {sigma:.3f} > 1")
    
    # Второй временной слой с разной аппроксимацией начального условия
    for i in range(N):
        u[1][i] = d_init_cond(x[i], tau, a, init_approx)
    
    # Основной цикл по времени
    for k in range(2, time_steps):
        for j in range(1, N-1):
            u[k][j] = sigma * u[k - 1][j - 1] + 2 * (1 - sigma) * u[k - 1][j] + sigma * u[k - 1][j + 1] - u[k - 2][j]
        
        # Применяем граничные условия выбранным методом
        u[k] = apply_boundary_conditions(u[k], h, boundary_method)
        
    return u, x, time_steps

def implicit(h, tau, t_range, x_range, boundary_method='two_point_first_order', init_approx='first_order'):
    t_start, t_end = t_range
    N = int(round((x_range[1] - x_range[0]) / h)) + 1
    time_steps = int(np.ceil((t_end - t_start) / tau)) + 1

    x = np.linspace(x_range[0], x_range[1], N)
    u = np.zeros((time_steps, N))
    u[0, :] = init_cond(x)
    
    sigma = a**2 * tau**2 / h**2
    a_j = sigma
    b_j = -(1 + 2 * sigma)
    c_j = sigma
    
    # Второй временной слой
    for i in range(N):
        u[1][i] = d_init_cond(x[i], tau, a, init_approx)
    
    for k in range(2, time_steps):
        # Создаем матрицу для полной системы
        matrix = np.zeros((N, N))
        d = np.zeros(N)
        
        # Внутренние точки (j = 1 до N-2)
        for j in range(1, N-1):
            matrix[j][j-1] = a_j
            matrix[j][j] = b_j
            matrix[j][j+1] = c_j
            d[j] = -2 * u[k-1][j] + u[k-2][j]
        
        # Граничные условия
        if boundary_method == 'two_point_first_order':
            # Левая граница: u_x(0) = u(0) => (u1 - u0)/h = u0 => u0 = u1/(1+h)
            # В матричной форме: (1+h)u0 - u1 = 0
            matrix[0][0] = 1 + h
            matrix[0][1] = -1
            d[0] = 0
            
            # Правая граница: u_x(π) = u(π) => (u_N-1 - u_N-2)/h = u_N-1 => u_N-1 = u_N-2/(1-h)
            # В матричной форме: -u_N-2 + (1-h)u_N-1 = 0
            matrix[N-1][N-2] = -1
            matrix[N-1][N-1] = 1 - h
            d[N-1] = 0
            
        elif boundary_method == 'three_point_second_order':
            # Левая граница: (-3u0 + 4u1 - u2)/(2h) = u0
            # => -3u0 + 4u1 - u2 = 2h*u0
            # => (-3-2h)u0 + 4u1 - u2 = 0
            matrix[0][0] = -3 - 2*h
            matrix[0][1] = 4
            matrix[0][2] = -1
            d[0] = 0
            
            # Правая граница: (3u_N-1 - 4u_N-2 + u_N-3)/(2h) = u_N-1
            # => 3u_N-1 - 4u_N-2 + u_N-3 = 2h*u_N-1
            # => u_N-3 - 4u_N-2 + (3-2h)u_N-1 = 0
            matrix[N-1][N-3] = 1
            matrix[N-1][N-2] = -4
            matrix[N-1][N-1] = 3 - 2*h
            d[N-1] = 0
            
        elif boundary_method == 'two_point_second_order':
            # Для двухточечной 2-го порядка используем тот же подход, что и для 1-го порядка
            # так как симметричная разность требует фиктивных точек
            matrix[0][0] = 1 + h
            matrix[0][1] = -1
            d[0] = 0
            
            matrix[N-1][N-2] = -1
            matrix[N-1][N-1] = 1 - h
            d[N-1] = 0
        
        # Решаем полную систему
        try:
            u[k] = np.linalg.solve(matrix, d)
        except np.linalg.LinAlgError:
            # Если матрица вырождена, используем псевдообратную
            u[k] = np.linalg.lstsq(matrix, d, rcond=None)[0]
        
    return u, x, time_steps

def plot_solutions(x, t_target, t_grid, title_suffix='', **solutions):
    k_target = np.argmin(np.abs(t_grid - t_target))
    actual_time = t_grid[k_target]
    
    plt.figure(figsize=(12, 6))
    for name, sol in solutions.items():
        plt.plot(x, sol[k_target, :], 'o', label=name, markersize=4, alpha=0.7)
            

    u_analytical = analytical_solution(x, actual_time, a)
    plt.plot(x, u_analytical, 'k-', label='Analytical', linewidth=2)

    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title(f'Solutions at t = {t_target:.3f} {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_vs_time(x, t_grid, a, title_suffix='', **solutions):
    plt.figure(figsize=(10, 6))
    
    for name, u_num in solutions.items():
        errors = []
        for k, t in enumerate(t_grid):
            u_true = analytical_solution(x, t, a)
            err = np.max(np.abs(u_num[k, :] - u_true))
            errors.append(err)
        plt.plot(t_grid[:len(errors)], errors, label=name, linewidth=2)
    
    plt.xlabel('Time $t$')
    plt.ylabel('Max absolute error')
    plt.title(f'Max error vs time {title_suffix}')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.yscale('log')
    plt.show()


# Граничные условия: Двухточечная 1-го порядка
# Второе начальное условие: Аппроксимация 1-го порядка
def main():
    x_range = [0.0, np.pi]
    # t_range = [0.0, 2.0]
    t_range = [0.0, 1.0]
    
    K, N = 50, 40
    tau = (t_range[1] - t_range[0]) / K
    h = (x_range[1] - x_range[0]) / N

    sigma = a**2 * tau**2 / h**2
    # print(f'sigma = {sigma:.3f}')
    if sigma > 1:
        print('Явная схема неустойчива!')
        return 1

    u_exp, x, time_steps_exp = explicit(h, tau, t_range, x_range)
    u_imp, _, time_steps_imp = implicit(h, tau, t_range, x_range)

    # Согласуем количество временных шагов
    min_steps = min(time_steps_exp, time_steps_imp)
    u_exp = u_exp[:min_steps]
    u_imp = u_imp[:min_steps]

    t_grid = np.linspace(t_range[0], t_range[0] + (min_steps - 1) * tau, min_steps)

    # Графики решений
    # for t_target in [1.0, 1.25, 1.5]:
    for t_target in [0.1, 0.25, 0.5]:
        plot_solutions(
            x, t_target, t_grid,
            Explicit=u_exp,
            Implicit=u_imp
        )

    # График ошибок
    plot_error_vs_time(
        x, t_grid, a,
        Explicit=u_exp,
        Implicit=u_imp
    )

# Сравнение методов с разной аппроксимацией граничных условий
def main_approx():
    x_range = [0.0, np.pi]
    # t_range = [0.0, 2.0]
    t_range = [0.0, 1.0]
    
    K, N = 50, 40
    tau = (t_range[1] - t_range[0]) / K
    h = (x_range[1] - x_range[0]) / N

    sigma = a**2 * tau**2 / h**2
    if sigma > 1:
        print('Явная схема неустойчива!')
        return 1

    boundary_methods = {
        'two_point_first_order': 'Двухточечная 1-го порядка (граничных условий)',
        'three_point_second_order': 'Трехточечная 2-го порядка (граничных условий)', 
        'two_point_second_order': 'Двухточечная 2-го порядка (граничных условий)'
    }

    for method, description in boundary_methods.items():        
        u_exp, x, time_steps_exp = explicit(h, tau, t_range, x_range, boundary_method=method)
        u_imp, _, time_steps_imp = implicit(h, tau, t_range, x_range, boundary_method=method)

        min_steps = min(time_steps_exp, time_steps_imp)
        u_exp = u_exp[:min_steps]
        u_imp = u_imp[:min_steps]

        t_grid = np.linspace(t_range[0], t_range[0] + (min_steps - 1) * tau, min_steps)

        plot_solutions(
            # x, 1.5, t_grid, description,
            x, 0.5, t_grid, description,
            Explicit=u_exp,
            Implicit=u_imp
        )

        # plot_error_vs_time(
        #     x, t_grid, a, description,
        #     Explicit=u_exp,
        #     Implicit=u_imp
        # )

# Сравнение методов с разной аппроксимацией второго начального условия
def main_init_approx():
    x_range = [0.0, np.pi]
    # t_range = [0.0, 2.0]
    t_range = [0.0, 1.0]
    
    K, N = 50, 40
    tau = (t_range[1] - t_range[0]) / K
    h = (x_range[1] - x_range[0]) / N

    sigma = a**2 * tau**2 / h**2
    if sigma > 1:
        print('Явная схема неустойчива!')
        return 1

    init_methods = {
        'first_order': 'Аппроксимация 1-го порядка (2-го нач.условия)',
        'second_order': 'Аппроксимация 2-го порядка (2-го нач.условия)'
    }

    for method, description in init_methods.items():        
        u_exp, x, time_steps_exp = explicit(h, tau, t_range, x_range, 
                                           boundary_method='two_point_first_order',
                                           init_approx=method)
        u_imp, _, time_steps_imp = implicit(h, tau, t_range, x_range,
                                           boundary_method='two_point_first_order', 
                                           init_approx=method)

        min_steps = min(time_steps_exp, time_steps_imp)
        u_exp = u_exp[:min_steps]
        u_imp = u_imp[:min_steps]

        t_grid = np.linspace(t_range[0], t_range[0] + (min_steps - 1) * tau, min_steps)

        plot_solutions(
            # x, 1.5, t_grid, description,
            x, 0.5, t_grid, description,
            Explicit=u_exp,
            Implicit=u_imp
        )

        # plot_error_vs_time(
        #     x, t_grid, a, description,
        #     Explicit=u_exp,
        #     Implicit=u_imp
        # )


def compute_error_at_final_time(u_num, x, t_final, a):
    u_true = analytical_solution(x, t_final, a)
    return np.max(np.abs(u_num[-1, :] - u_true))

def convergence_in_time(a, h_fixed, t_range, tau_values, x_range):
    errors_explicit = []
    
    errors_implicit = []
    valid_taus = []
    
    t_final = t_range[1]
    
    for tau in tau_values:
        # Проверка устойчивости явной схемы
        sigma = a**2 * tau**2 / h_fixed**2

        if sigma > 1.0:
            print(f"tau={tau:.6f}: sigma={sigma:.3f} > 1, пропускаем явную схему")
            errors_explicit.append(np.nan)
        else:
            try:
                u_exp, x, _ = explicit(h_fixed, tau, t_range, x_range)
                err_exp = compute_error_at_final_time(u_exp, x, t_final, a)
                errors_explicit.append(err_exp)
            except Exception as e:
                print(f"Ошибка в явной схеме при tau={tau:.6f}: {e}")
                errors_explicit.append(np.nan)
        
        try:
            u_imp, x, _ = implicit(h_fixed, tau, t_range, x_range)
            err_imp = compute_error_at_final_time(u_imp, x, t_final, a)
            errors_implicit.append(err_imp)
        except Exception as e:
            print(f"Ошибка в неявной схеме при tau={tau:.6f}: {e}")
            errors_implicit.append(np.nan)
        
        valid_taus.append(tau)
    
    return np.array(valid_taus), {'Explicit': np.array(errors_explicit), 
                                   'Implicit': np.array(errors_implicit)}

def convergence_in_space(a, tau_fixed, t_range, h_values, x_range):
    errors_explicit = []
    errors_implicit = []
    valid_hs = []
    
    t_final = t_range[1]
    
    for h in h_values:
        # Проверка устойчивости явной схемы
        sigma = a**2 * tau_fixed**2 / h**2
        
        if sigma > 1.0:
            print(f"h={h:.6f}: sigma={sigma:.3f} > 1, пропускаем явную схему")
            errors_explicit.append(np.nan)
        else:
            try:
                u_exp, x, _ = explicit(h, tau_fixed, t_range, x_range)
                err_exp = compute_error_at_final_time(u_exp, x, t_final, a)
                errors_explicit.append(err_exp)
            except Exception as e:
                print(f"Ошибка в явной схеме при h={h:.6f}: {e}")
                errors_explicit.append(np.nan)
        
        try:
            u_imp, x, _ = implicit(h, tau_fixed, t_range, x_range)
            err_imp = compute_error_at_final_time(u_imp, x, t_final, a)
            errors_implicit.append(err_imp)
        except Exception as e:
            print(f"Ошибка в неявной схеме при h={h:.6f}: {e}")
            errors_implicit.append(np.nan)
        
        valid_hs.append(h)
    
    return np.array(valid_hs), {'Explicit': np.array(errors_explicit), 
                                 'Implicit': np.array(errors_implicit)}

def plot_convergence(taus, err_tau_dict, hs, err_h_dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {
        'Explicit': 'red',
        'Implicit': 'blue'
    }
    markers = {
        'Explicit': 'o',
        'Implicit': 's'
    }

    # Сходимость по времени (при фиксированном h)
    for method in ['Explicit', 'Implicit']:
        err = err_tau_dict[method]
        valid = ~np.isnan(err)
        if np.any(valid):
            ax1.plot(taus[valid], err[valid], 
                     marker=markers[method], color=colors[method], 
                     label=method, markersize=6, linewidth=1.5)
    
    # Теоретические линии сходимости
    valid_imp = ~np.isnan(err_tau_dict['Implicit'])
    if np.any(valid_imp):
        tau_ref = taus[valid_imp]
        err_ref = err_tau_dict['Implicit'][valid_imp]
        C1 = err_ref[0] / tau_ref[0]
        ax1.plot(tau_ref, C1 * tau_ref, 'k--', label=r'$O(\tau)$')
    
    ax1.set_xlabel(r'Шаг по времени $\tau$')
    ax1.set_ylabel('Максимальная погрешность в конечный момент')
    ax1.set_title('Сходимость по времени (фиксированный $h$)')
    ax1.grid(True, which="both", ls=":", linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()

    # Сходимость по пространству (при фиксированном tau)
    for method in ['Explicit', 'Implicit']:
        err = err_h_dict[method]
        valid = ~np.isnan(err)
        if np.any(valid):
            ax2.plot(hs[valid], err[valid], 
                     marker=markers[method], color=colors[method], 
                     label=method, markersize=6, linewidth=1.5)
    
    # Теоретическая линия сходимости O(h²)
    valid_exp = ~np.isnan(err_h_dict['Explicit'])
    if np.any(valid_exp):
        h_ref = hs[valid_exp]
        err_ref = err_h_dict['Explicit'][valid_exp]
        C2 = err_ref[0] / (h_ref[0] ** 2)
        ax2.plot(h_ref, C2 * h_ref**2, 'k--', label=r'$O(h^2)$')
    
    ax2.set_xlabel(r'Шаг по пространству $h$')
    ax2.set_ylabel('Максимальная погрешность в конечный момент')
    ax2.set_title('Сходимость по пространству (фиксированный $\tau$)')
    ax2.grid(True, which="both", ls=":", linewidth=0.5)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()

    plt.tight_layout()
    plt.show()
    

def main_convergence():
    a = 1.0
    x_range = [0.0, np.pi]
    t_range = [0.0, 1.0]
    
    # Сходимость по времени (фиксированный h)
    N_fixed = 50
    h_fixed = (x_range[1] - x_range[0]) / N_fixed
    
    # Различные шаги по времени
    K_values = np.array([50, 100, 200, 400, 800])
    tau_values = (t_range[1] - t_range[0]) / K_values
    
    taus, err_tau_dict = convergence_in_time(a, h_fixed, t_range, tau_values, x_range)
    
    
    # Сходимость по пространству (фиксированный tau)
    K_fixed = 500
    tau_fixed = (t_range[1] - t_range[0]) / K_fixed
    
    # Различные шаги по пространству
    N_values = np.array([20, 30, 40, 60, 80, 100])
    h_values = (x_range[1] - x_range[0]) / N_values
    
    hs, err_h_dict = convergence_in_space(a, tau_fixed, t_range, h_values, x_range)
    
    plot_convergence(taus, err_tau_dict, hs, err_h_dict)


if __name__ == "__main__":
    main()
    main_approx()
    main_init_approx()
    main_convergence()