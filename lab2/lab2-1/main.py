import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**6 - 5 * x - 2

def df(x):
    return 6 * x**5 - 5

def phi(x):
        return (5 * x + 2)**(1/6)

def dphi(x):
        return 5 / 6 / (5 * x + 2)**(5/6)

def ddphi(x):
     return - 125 / 36 / (5 * x + 2)**(11/6)

def simple_iteration_method(phi, a, b, epsilon, max_iter=1000):
    q = max([abs(dphi(x)) for x in np.arange(a, b, epsilon)])
    if q >= 1:
        print("Не удовлетворяется условие q < 1")
        exit()

    errors = []
    x_s = []
    x_prev = (a + b) / 2

    iterations = 0
    while True:
        x_next = phi(x_prev)
        error = q / (1 - q) * abs(x_next - x_prev)
        errors.append(error)
        x_s.append(x_prev)
        iterations += 1
        if error < epsilon or iterations >= max_iter:
            break
        x_prev = x_next
    return x_next, iterations, errors, x_s

def newton_method(f, df, a, b, epsilon, max_iter=1000):
    M2 = max(([abs(ddphi(x)) for x in np.arange(a, b, epsilon)]))
    m1 = min([abs(dphi(x)) for x in np.arange(a, b, epsilon)])

    errors = []
    x_s = []
    x_prev = (a + b) / 2

    iterations = 0
    while True:
        x_next = x_prev - f(x_prev) / df(x_prev)
        error = M2 / (2 * m1) * (x_next - x_prev)**2
        errors.append(error)
        x_s.append(x_prev)
        iterations += 1
        if error < epsilon or iterations >= max_iter:
            break
        x_prev = x_next
    return x_next, iterations, errors, x_s

def draw_graph_errors(iter_si, errors_si, iter_n, errors_n):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, iter_si + 1), errors_si, label='Метод простых итераций', marker='o')
    plt.plot(range(1, iter_n + 1), errors_n, label='Метод Ньютона', marker='s')
    plt.xlabel('Количество итераций')
    plt.ylabel('Погрешность')
    plt.title('Зависимость погрешности от количества итераций')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.show()

def draw_graph_x(iter_si, x_vals_si, iter_n, x_vals_n):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, iter_si + 1), x_vals_si, label='Метод простых итераций', marker='o')
    plt.plot(range(1, iter_n + 1), x_vals_n, label='Метод Ньютона', marker='s')
    plt.xlabel('Количество итераций')
    plt.ylabel('Значения')
    plt.title('График изменения значения x')
    plt.ylim(1, 3.5)
    plt.yticks(np.arange(0, 3.1, 0.5))
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    with open('input.txt', 'r') as file:
        line = file.readline()
        a, b = map(float, line.split())
        epsilon = float(file.readline())

    if (f(a) * f(b) >= 0):
        print("Выбраны неподходящие границы a и b")
        exit()

    # Метод простых итераций
    x_si, iter_si, errors_si, x_vals_si = simple_iteration_method(phi, a, b, epsilon)
    
    # Метод Ньютона
    x_n, iter_n, errors_n, x_vals_n = newton_method(f, df, a, b, epsilon)

    with open('output.txt', 'w') as file:
        file.write(f"Nonlinear equation:\nf(x) = x^6 - 5x - 2\n")
        file.write(f"Interval: [{a}, {b}]\n")
        file.write(f"Epsilon: {epsilon}\n")
        file.write(f"\nSimple iterations method:\n")
        file.write(f"Root: {x_si}\nNumber of iterations: {iter_si}\nf(x) = {f(x_si)}\nError: {errors_si[-1]}\n")
        file.write(f"\nNewton method:\n")
        file.write(f"Root: {x_n}\nNumber of iterations: {iter_n}\nf(x) = {f(x_n)}\nError: {errors_n[-1]}")

    draw_graph_x(iter_si, x_vals_si, iter_n, x_vals_n)
    draw_graph_errors(iter_si, errors_si, iter_n, errors_n)
    
if __name__ == "__main__":
    main()