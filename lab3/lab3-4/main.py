from scipy.interpolate import CubicSpline
import numpy as np

def df(x_test, x, y):
    assert len(x) == len(y)
    for interval in range(len(x)):
        if x[interval] <= x_test < x[interval + 1]:
            i = interval
            break

    a1 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    # наклон линейной интерполяции
    a2 = ((y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - a1) / (x[i + 2] - x[i]) * (2 * x_test - x[i] - x[i + 1])
    # учитываем кривизну сплайна
    return a1 + a2

def d2f(x_test, x, y):
    assert len(x) == len(y)
    for interval in range(len(x)):
        if x[interval] <= x_test < x[interval + 1]:
            i = interval
            break

    num = (y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    return 2 * num / (x[i + 2] - x[i])

def check_with_scipy(x_test, x, y):
    cs = CubicSpline(x, y)
    df_scipy = cs(x_test, 1)
    d2f_scipy = cs(x_test, 2)
    return df_scipy, d2f_scipy

def main():
    with open('input.txt', 'r') as file:
        data = [list(map(float, line.split())) for line in file.readlines()]
    points = data[:-1]
    x_test = data[-1][0]
    x = points[0]
    y = points[1]

    x_np, y_np = np.array(x), np.array(y)
    df_scipy, d2f_scipy = check_with_scipy(x_test, x_np, y_np)

    with open('output.txt', 'w') as file:
        file.write(f"First derivative:\ndf({x_test}) = {df(x_test, x, y)}\n")
        file.write(f"Check:\ndf_scipy({x_test}) = {df_scipy}\n\n")
        file.write(f"Second derivative:\nd2f({x_test}) = {d2f(x_test, x, y)}\n")
        file.write(f"Check:\nd2f_scipy({x_test}) = {d2f_scipy}")

if __name__ == '__main__':
    main()