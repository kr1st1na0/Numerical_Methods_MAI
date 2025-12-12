import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['figure.figsize'] = [10, 10]


# Граничные условия
def phi1(y):
    return 0


def phi2(y):
    return 1 - y * y


def phi3(x):
    return 0


def phi4(x):
    return x * x - 1


# Аналитическое решение
def U(x, y):
    return x * x - y * y


# Норма
def norm(v1, v2):
    return np.amax(np.abs(v1 - v2))


# Функция для вычисления погрешностей
def error(lx, hy, Ny):
    Nx_array = [5, 10, 20, 40]
    size = np.size(Nx_array)
    hx_array = np.zeros(size)
    errors1 = np.zeros(size)
    errors2 = np.zeros(size)
    errors3 = np.zeros(size)
    for i in range(0, size):
        hx_array[i] = lx / Nx_array[i]
        x_array = np.arange(0, lx + hx_array[i], hx_array[i])
        u1, tmp = Liebman(hx_array[i], hy, Nx_array[i], Ny)
        u2, tmp = Seidel(hx_array[i], hy, Nx_array[i], Ny)
        u3, tmp = UpperRelaxation(hx_array[i], hy, Nx_array[i], Ny)
        if (np.size(x_array) != Nx_array[i] + 1):
            x_array = x_array[:Nx_array[i] + 1]
        y = hy * Ny / 5
        u_correct = np.zeros(np.size(x_array))
        for j in range(np.size(x_array)):
            u_correct[j] = U(x_array[j], y)
        u1_calculated = u1[:, int(Ny / 5)]
        u2_calculated = u2[:, int(Ny / 5)]
        u3_calculated = u3[:, int(Ny / 5)]
        errors1[i] = norm(u_correct, u1_calculated)
        errors2[i] = norm(u_correct, u2_calculated)
        errors3[i] = norm(u_correct, u3_calculated)
    return Nx_array, errors1, errors2, errors3


# Функция для построения графика ошибок
def show_errors(lx, hy, Ny):
    Nx_array, errors1, errors2, errors3 = error(lx, hy, Ny)
    colors = ['blue', 'green', 'red']
    
    fig, ax = plt.subplots()
    plt.plot(Nx_array, errors1, color=colors[0], label='Метод Либмана', marker='o')
    plt.plot(Nx_array, errors2, color=colors[1], label='Метод Зейделя', marker='s')
    plt.plot(Nx_array, errors3, color=colors[2], label='Метод простых итераций с верхней релаксацией', marker='^')
    
    ax.set_xlabel('Количество узлов сетки Nₓ')
    ax.set_ylabel('Погрешность')
    ax.set_title('Зависимость погрешности от числа узлов сетки')
    ax.set_yscale('log')  # Рекомендуется для лучшей визуализации сходимости
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    ax.legend()
    plt.show()


# Функция для отрисовки решения в сечениях по y
def show_solution_slices_y(Nx, Ny, hx, hy, U, ulieb, usei, usor, y_slices=[0.2, 0.5, 0.8]):
    # y_slices: [0.2, 0.5, 0.8] => y = 0.2*ly, 0.5*ly, 0.8*ly
    x_array = np.array([i * hx for i in range(Nx + 1)])
    colors = ['black', 'blue', 'green', 'red']

    for y_ratio in y_slices:
        y_fixed = y_ratio * ly
        y_index = int(y_fixed / hy)
        if y_index >= Ny + 1:
            y_index = Ny

        fig, ax = plt.subplots()
        u_correct = U(x_array, y_fixed)
        u_liebman = ulieb[:, y_index]  # Сечение по y (столбец)
        u_seidel = usei[:, y_index]
        u_sor = usor[:, y_index]

        plt.plot(x_array, u_correct, color=colors[0], label=f'Точное решение')
        plt.plot(x_array, u_liebman, color=colors[1], label='Метод Либмана')
        plt.plot(x_array, u_seidel, color=colors[2], label='Метод Зейделя')
        plt.plot(x_array, u_sor, color=colors[3], label='Метод простых итераций с верхней релаксацией')

        ax.set_xlabel('x')
        ax.set_ylabel('U(x, y)')
        ax.set_title(f'Решение при y = {y_fixed:.2f}')
        plt.grid()
        ax.legend()
        plt.show()


def show_solution_slices_x(Nx, Ny, hx, hy, U, ulieb, usei, usor, x_slices=[0.2, 0.5, 0.8]):
    # Вычисляем длину области
    lx = Nx * hx
    ly = Ny * hy
    
    # Массив координат y
    y_array = np.array([j * hy for j in range(Ny + 1)])
    
    colors = ['black', 'blue', 'green', 'red']
    
    for x_ratio in x_slices:
        x_fixed = x_ratio * lx
        x_index = int(x_fixed / hx)
        if x_index >= Nx + 1:
            x_index = Nx
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Точное решение
        u_correct = U(x_fixed, y_array)
        
        # Численные решения (сечение по строке)
        u_liebman = ulieb[x_index, :]
        u_seidel = usei[x_index, :]
        u_sor = usor[x_index, :]
        
        plt.plot(y_array, u_correct, color=colors[0], label=f'Точное решение')
        plt.plot(y_array, u_liebman, color=colors[1], label='Метод Либмана')
        plt.plot(y_array, u_seidel, color=colors[2], label='Метод Зейделя')
        plt.plot(y_array, u_sor, color=colors[3], label='Метод простых итераций с верхней релаксацией')

        ax.set_xlabel('x')
        ax.set_ylabel('U(x, y)')
        ax.set_title(f'Решение при x = {x_fixed:.2f}')
        plt.grid()
        ax.legend()
        plt.show()


# Функция для построения трёхмерного графика решения
def show_solution3d(Nx, Ny, hx, hy, u, method_name, elev=45, azim=45):
    x = np.array([i * hx for i in range(Nx + 1)])
    y = np.array([j * hy for j in range(Ny + 1)])
    xgrid, ygrid = np.meshgrid(x, y)
    z = u
    fig = plt.figure()
    axes = fig.add_subplot(projection='3d')
    cmap = LinearSegmentedColormap.from_list('red_blue', ['b', 'r'], 256)
    axes.view_init(elev=elev, azim=azim)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('U(x,y)')
    axes.set_title('3D solution: ' + method_name)
    if z.shape != (len(y), len(x)):
        z = z.T
    axes.plot_surface(xgrid, ygrid, z, cmap=cmap, rcount=Ny + 1, ccount=Nx + 1, edgecolor='none')
    plt.show()


# Метод простых итераций (Либмана)
def Liebman(hx, hy, Nx, Ny, eps = 1e-5):
    u = np.zeros((Nx + 1, Ny + 1))
    # Граничные условия 1-го рода
    for i in range(Nx + 1):
        u[i][Ny] = phi4(i * hx)
    for j in range(Ny + 1):
        u[Nx][j] = phi2(j * hy)

    iter_num = 0
    while True:
        u_next = deepcopy(u)
        for i in range(1, Nx):
            for j in range(1, Ny):
                u_next[i][j] = 1 / (2 / hx ** 2 + 2 / hy ** 2) * ((u[i - 1][j] + u[i + 1][j]) / hx ** 2 + (u[i][j - 1] + u[i][j + 1]) / hy ** 2)
        # Граничные условия 2-го рода
        for i in range(Nx):
            u_next[i][0] = u_next[i][1] - hy * phi3(i * hx)
        for j in range(Ny):
            u_next[0][j] = u_next[1][j] - hx * phi1(j * hy)

        if norm(u, u_next) < eps:
            break
        u = deepcopy(u_next)
        iter_num += 1
    return u, iter_num


# Метод Зейделя
def Seidel(hx, hy, Nx, Ny, eps = 1e-5):
    u = np.zeros((Nx + 1, Ny + 1))
    # Граничные условия 1-го рода
    for i in range(Nx + 1):
        u[i][Ny] = phi4(i * hx)
    for j in range(Ny + 1):
        u[Nx][j] = phi2(j * hy)

    iter_num = 0
    while True:
        u_prev = deepcopy(u)
        u_next = deepcopy(u)
        for i in range(1, Nx):
            for j in range(1, Ny):
                u_next[i][j] = 1 / (2 / hx ** 2 + 2 / hy ** 2) * ((u_next[i - 1][j] + u_prev[i + 1][j]) / hx ** 2 + (u_next[i][j - 1] + u_prev[i][j + 1]) / hy ** 2)
        # Граничные условия 2-го рода
        for i in range(Nx):
            u_next[i][0] = u_next[i][1] - hy * phi3(i * hx)
        for j in range(Ny):
            u_next[0][j] = u_next[1][j] - hx * phi1(j * hy)

        if norm(u_prev, u_next) < eps:
            break
        u = deepcopy(u_next)
        iter_num += 1
    return u, iter_num


# Метод простых итераций с верхней релаксации
def UpperRelaxation(hx, hy, Nx, Ny, eps = 1e-5, omega = 1.25):
    u = np.zeros((Nx + 1, Ny + 1))
    # Граничные условия 1-го рода
    for i in range(Nx + 1):
        u[i][Ny] = phi4(i * hx)
    for j in range(Ny + 1):
        u[Nx][j] = phi2(j * hy)

    iter_num = 0
    while True:
        u_prev = deepcopy(u)
        u_next = deepcopy(u)
        for i in range(1, Nx):
            for j in range(1, Ny):
                u_next[i][j] = 1 / (2 / hx ** 2 + 2 / hy ** 2) * ((u_next[i - 1][j] + u_prev[i + 1][j]) / hx ** 2 + (u_next[i][j - 1] + u_prev[i][j + 1]) / hy ** 2)
        # Граничные условия 2-го рода
        for i in range(Nx):
            u_next[i][0] = u_next[i][1] - hy * phi3(i * hx)
        for j in range(Ny):
            u_next[0][j] = u_next[1][j] - hx * phi1(j * hy)
        # Релаксация
        u_next = omega * u_next + (1 - omega) * u_prev

        if norm(u_prev, u_next) < eps:
            break
        u = deepcopy(u_next)
        iter_num += 1
    return u, iter_num


def main():
    global lx, ly
    lx = ly = 1
    Nx = Ny = 30
    hx = lx / Nx
    hy = ly / Ny

    u1, iter_num1 = Liebman(hx, hy, Nx, Ny)
    print("Кол-во итераций метода Либмана:", iter_num1)
    u2, iter_num2 = Seidel(hx, hy, Nx, Ny)
    print("Кол-во итераций метода Зейделя:", iter_num2)
    u3, iter_num3 = UpperRelaxation(hx, hy, Nx, Ny)
    print("Кол-во итераций метода простых итераций с верхней релаксацией:", iter_num3)

    # Построение 3D графика
    show_solution3d(Nx, Ny, hx, hy, u1, "Liebman_method", elev=20, azim=110)
    show_solution3d(Nx, Ny, hx, hy, u2, "Seidel_method", elev=20, azim=110)
    show_solution3d(Nx, Ny, hx, hy, u3, "UpperRelaxation_method", elev=20, azim=110)

    # Построение графиков решений в сечениях по y
    show_solution_slices_y(Nx, Ny, hx, hy, U, u1, u2, u3, y_slices=[0.2, 0.5, 0.8])

    # Построение графиков решений в сечениях по x
    show_solution_slices_x(Nx, Ny, hx, hy, U, u1, u2, u3, x_slices=[0.2, 0.5, 0.8])

    # График погрешностей
    show_errors(lx, hy, Ny)


# Функция для вычисления погрешностей в зависимости от hx (при фиксированном hy)
def error_hx_dependence(lx, hy, Ny):
    Nx_array = [5, 10, 20, 40]
    size = np.size(Nx_array)
    hx_array = np.zeros(size)
    errors1 = np.zeros(size)
    errors2 = np.zeros(size)
    errors3 = np.zeros(size)
    for i in range(0, size):
        hx_array[i] = lx / Nx_array[i]
        x_array = np.arange(0, lx + hx_array[i], hx_array[i])
        u1, tmp = Liebman(hx_array[i], hy, Nx_array[i], Ny)
        u2, tmp = Seidel(hx_array[i], hy, Nx_array[i], Ny)
        u3, tmp = UpperRelaxation(hx_array[i], hy, Nx_array[i], Ny)

        if len(x_array) > Nx_array[i] + 1:
            x_array = x_array[:Nx_array[i] + 1]
        elif len(x_array) < Nx_array[i] + 1:
            x_array = np.append(x_array, lx)

        # Выберем сечение по y = 0.2 * ly
        y_fixed = 0.2 * ly
        y_index = int(y_fixed / hy)  # Индекс строки в сетке, соответствующий y_fixed
        if y_index >= Ny + 1:
            y_index = Ny  # Защита от выхода за границы

        u_correct = np.zeros(len(x_array))
        for j in range(len(x_array)):
            u_correct[j] = U(x_array[j], y_fixed)

        u1_calculated = u1[:, y_index]
        u2_calculated = u2[:, y_index]
        u3_calculated = u3[:, y_index]

        errors1[i] = norm(u_correct, u1_calculated)
        errors2[i] = norm(u_correct, u2_calculated)
        errors3[i] = norm(u_correct, u3_calculated)

    return hx_array, errors1, errors2, errors3


# Функция для вычисления погрешностей в зависимости от hy (при фиксированном hx)
def error_hy_dependence(lx, hx, Nx):
    Ny_array = [5, 10, 20, 40]
    size = np.size(Ny_array)
    hy_array = np.zeros(size)
    errors1 = np.zeros(size)
    errors2 = np.zeros(size)
    errors3 = np.zeros(size)
    for i in range(0, size):
        hy_array[i] = ly / Ny_array[i]
        y_array = np.arange(0, ly + hy_array[i], hy_array[i])
        u1, tmp = Liebman(hx, hy_array[i], Nx, Ny_array[i])
        u2, tmp = Seidel(hx, hy_array[i], Nx, Ny_array[i])
        u3, tmp = UpperRelaxation(hx, hy_array[i], Nx, Ny_array[i])
        # Убедимся, что размеры совпадают
        if len(y_array) > Ny_array[i] + 1:
            y_array = y_array[:Ny_array[i] + 1]
        elif len(y_array) < Ny_array[i] + 1:
            y_array = np.append(y_array, ly)

        # Выберем сечение по x = 0.3 * lx
        x_fixed = 0.3 * lx  
        x_index = int(x_fixed / hx)  # Индекс столбца в сетке, соответствующий x_fixed
        if x_index >= Nx + 1:
            x_index = Nx

        u_correct = np.zeros(len(y_array))
        for j in range(len(y_array)):
            u_correct[j] = U(x_fixed, y_array[j])

        u1_calculated = u1[x_index, :]
        u2_calculated = u2[x_index, :]
        u3_calculated = u3[x_index, :]

        errors1[i] = norm(u_correct, u1_calculated)
        errors2[i] = norm(u_correct, u2_calculated)
        errors3[i] = norm(u_correct, u3_calculated)

    return hy_array, errors1, errors2, errors3


# Функция для построения графика ошибок в зависимости от hx
def show_errors_hx(lx, hy, Ny):
    hx_array, errors1, errors2, errors3 = error_hx_dependence(lx, hy, Ny)
    colors = ['blue', 'green', 'red']
    fig, ax = plt.subplots()
    plt.plot(hx_array, errors1, color=colors[0], label='Метод Либмана')
    plt.plot(hx_array, errors2, color=colors[1], label='Метод Зейделя')
    plt.plot(hx_array, errors3, color=colors[2], label='Метод простых итераций с верхней релаксацией')
    ax.set_xlabel('h_x')
    ax.set_ylabel('Погрешность')
    ax.set_title('Зависимость погрешности от шага h_x')
    plt.grid()
    ax.legend()
    plt.show()


# Функция для построения графика ошибок в зависимости от hy
def show_errors_hy(lx, hx, Nx):
    hy_array, errors1, errors2, errors3 = error_hy_dependence(lx, hx, Nx)
    colors = ['blue', 'green', 'red']
    fig, ax = plt.subplots()
    plt.plot(hy_array, errors1, color=colors[0], label='Метод Либмана')
    plt.plot(hy_array, errors2, color=colors[1], label='Метод Зейделя')
    plt.plot(hy_array, errors3, color=colors[2], label='Метод простых итераций с верхней релаксацией')
    ax.set_xlabel('h_y')
    ax.set_ylabel('Погрешность')
    ax.set_title('Зависимость погрешности от шага h_y')
    plt.grid()
    ax.legend()
    plt.show()


def main_convergence():
    Nx = Ny = 30
    hx = lx / Nx
    hy = ly / Ny

    # Исследование зависимости погрешности от hx
    show_errors_hx(lx, hy, Ny)

    # Исследование зависимости погрешности от hy
    show_errors_hy(lx, hx, Nx)


if __name__ == "__main__":
    main()
    # main_convergence()
