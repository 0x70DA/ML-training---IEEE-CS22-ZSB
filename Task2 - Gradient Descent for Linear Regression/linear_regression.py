import csv
import matplotlib.pyplot as plt
import random


def load_data(file_name):
    x, y = [], []
    with open(file_name, "r") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    return x, y


def gradient_descent(x, y):
    alpha = 0.0001
    theta_0, theta_1 = random.random(), random.random()
    m = len(x)
    iterations = 100
    for _ in range(iterations):
        # For theta_0
        j_0 = 0
        for i in range(m):
            j_0 += (theta_0 + (theta_1 * x[i])) - y[i]
        temp_0 = theta_0 - alpha * (1/m) * j_0

        # For theta_1
        j_1 = 0
        for i in range(m):
            j_1 += ((theta_0 + (theta_1 * x[i])) - y[i]) * x[i]
        temp_1 = theta_1 - alpha * (1/m) * j_1

        theta_0, theta_1 = temp_0, temp_1

    return theta_0, theta_1


if __name__ == '__main__':
    x, y = load_data("dataset.csv")
    theta_0, theta_1 = gradient_descent(x, y)

    # Plot the result
    result = []
    for i in x:
        result.append(theta_0 + theta_1 * i)
    plt.plot(x, y, 'bo')
    plt.plot(x, result, color='green')
    plt.show()
