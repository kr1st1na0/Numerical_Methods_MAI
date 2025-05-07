import math
import numpy as np
import matplotlib.pyplot as plt

def f1(x1, x2):
    return 3 * x1**2 - x1 + x2**2 - 1

def f2(x1, x2):
    return x2 - math.tan(x1)

def df1_dx1(x1, x2): return 6 * x1 - 1
def df1_dx2(x1, x2): return 2 * x2
def df2_dx1(x1, x2): return -1 / (math.cos(x1))**2
def df2_dx2(x1, x2): return 1

def phi1(x1, x2): 
    return x1 - f1(x1, x2) / 10

def phi2(x1, x2): 
    return math.tan(x1)

def dphi1_dx1(x1, x2): return 1 - (6 * x1 - 1) / 10
def dphi1_dx2(x1, x2): return - (2 * x2) / 10
def dphi2_dx1(x1, x2): return 1 / (math.cos(x1))**2 
def dphi2_dx2(x1, x2): return 0

def simple_iteration_method(intervals, eps, max_iter=100):
    a1, b1 = intervals[0][0], intervals[0][1]
    a2, b2 = intervals[1][0], intervals[1][1]
    x1 = (a1 + b1) / 2
    x2 = (a2 + b2) / 2
    iters = 0
    errors = []
    history = []

    J = np.array([[dphi1_dx1(x1, x2), dphi1_dx2(x1, x2)],
                  [dphi2_dx1(x1, x2), dphi2_dx2(x1, x2)]])

    q = np.linalg.norm(J, ord=np.inf)
    if (q < 1):
        while True:
            x1_new, x2_new = phi1(x1, x2), phi2(x1, x2)
            error = q / (1 - q) * max(abs(x1_new - x1), abs(x2_new - x2))
            errors.append(error)
            history.append((x1, x2))
            iters += 1
            if error < eps or iters >= max_iter:
                break
            x1, x2 = x1_new, x2_new
    else:
        while True:
            x1_new, x2_new = phi1(x1, x2), phi2(x1, x2)
            error = max(abs(x1_new - x1), abs(x2_new - x2))
            errors.append(error)
            history.append((x1, x2))
            iters += 1
            if error < eps or iters >= max_iter:
                break
            x1, x2 = x1_new, x2_new
    return [x1, x2], iters, errors, history


def newton_method(intervals, eps, max_iter=100):
    a1, b1 = intervals[0][0], intervals[0][1]
    a2, b2 = intervals[1][0], intervals[1][1]
    x1 = (a1 + b1) / 2
    x2 = (a2 + b2) / 2
    iters = 0
    errors = []
    history = []
    
    while True:
        J = np.array([[df1_dx1(x1, x2), df1_dx2(x1, x2)],
                      [df2_dx1(x1, x2), df2_dx2(x1, x2)]])
        F = np.array([f1(x1, x2), f2(x1, x2)])
        
        delta = np.linalg.solve(J, -F)
        x1_new, x2_new = x1 + delta[0], x2 + delta[1]
        error = max(abs(x1_new - x1), abs(x2_new - x2))
        iters += 1
        errors.append(error)
        history.append((x1, x2))
        if error < eps or iters >= max_iter:
            break
        x1, x2 = x1_new, x2_new
    return [float(x1), float(x2)], iters, errors, history

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

def draw_graph_system(iter_si, history_si, iter_n, history_n):
    x1_si = [point[0] for point in history_si]
    x2_si = [point[1] for point in history_si]
    
    x1_n = [point[0] for point in history_n]
    x2_n = [point[1] for point in history_n]

    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, iter_si + 1), x1_si, label='Метод простых итераций', marker='o')
    plt.plot(range(1, iter_n + 1), x1_n, label='Метод Ньютона', marker='s')
    plt.xlabel('Количество итераций')
    plt.ylabel('Значения')
    plt.title('График изменения значения x1')
    plt.ylim(1, 2)
    plt.yticks(np.arange(0, 2.1, 0.25))
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, iter_si + 1), x2_si, label='Метод простых итераций', marker='o')
    plt.plot(range(1, iter_n + 1), x2_n, label='Метод Ньютона', marker='s')
    plt.xlabel('Количество итераций')
    plt.ylabel('Значения')
    plt.title('График изменения значения x2')
    plt.ylim(1, 2)
    plt.yticks(np.arange(0, 2.1, 0.25))
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    with open('input.txt', 'r') as file:
        line = file.readline()
        a1, b1 = map(float, line.split())
        line = file.readline()
        a2, b2 = map(float, line.split())
        epsilon = float(file.readline())

    intervals = [(a1, b1), (a2, b2)]

    # Метод простых итераций
    x_si, iter_si, errors_si, history_si = simple_iteration_method(intervals, epsilon)
    
    # Метод Ньютона
    x_n, iter_n, errors_n, history_n = newton_method(intervals, epsilon)

    with open('output.txt', 'w') as file:
        file.write(f"System of nonlinear equations:\n | 3*x1^2 - x1 + x2^2 - 1 = 0\n | x2 - tg(x1) = 0\n")
        file.write(f"Intervals: [{a1}, {b1}], [{a2}, {b2}]\n")
        file.write(f"Epsilon: {epsilon}\n")
        file.write(f"\nSimple iterations method:\n")
        file.write(f"Root: {x_si}\nNumber of iterations: {iter_si}\n")
        file.write(f"f1(x1, x2) = {f1(*x_si)}\nf2(x1, x2) = {f2(*x_si)}\nError: {errors_si[-1]}\n")
        file.write(f"\nNewton method:\n")
        file.write(f"Root: {x_n}\nNumber of iterations: {iter_n}\n")
        file.write(f"f1(x1, x2) = {f1(*x_n)}\nf2(x1, x2) = {f2(*x_n)}\nError: {errors_n[-1]}\n")

    draw_graph_errors(iter_si, errors_si, iter_n, errors_n)
    draw_graph_system(iter_si, history_si, iter_n, history_n)
    

if __name__ == "__main__":
    main()