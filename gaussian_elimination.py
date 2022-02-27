import sys


def gaussian_elimination(a, n):
    x = [0] * n
    for i in range(n):
        if a[i][i] == 0:
            sys.exit("Error: Division by zero")

        for j in range(i+1, n):
            b = a[j][i] / a[i][i]
            for k in range(n+1):
                a[j][k] = a[j][k] - b * a[i][k]

    x[n-1] = a[n-1][n] / a[n-1][n-1]
    for i in range(n-2, -1, -1):
        x[i] = a[i][n]

        for j in range(i+1, n):
            x[i] = x[i] - a[i][j] * x[j]

        x[i] = x[i] / a[i][i]

    return x


if __name__ == '__main__':
    n = int(input('Number of variables: '))
    print("Enter augmented matrix coefficients row by row:")
    a = []
    for _ in range(n):
        a.append(list(map(int, input().split())))

    x = gaussian_elimination(a, n)
    print("The solution is:")
    for i in range(n):
        print(f"X{i} = {x[i]:.2f}")
