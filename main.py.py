# Source of data https://www.kaggle.com/datasets/4c8d766e253c62e5910952e619db9267f34c58497a74001571106b157080ee9b?resource=download
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/data.csv')

def analytical_solution(dataset):
    avg_x = dataset['Temperature'].mean()
    avg_y = dataset['Revenue'].mean()

    numerator = sum((dataset['Temperature'] - avg_x) * (dataset['Revenue'] - avg_y))
    denominator = sum((dataset['Temperature'] - avg_x) ** 2)

    A = numerator/denominator
    B = avg_y - A*avg_x
    return A, B

def gradient_descent(A, B, dataset, L):
    A_gradient = 0
    B_gradient = 0
    n = len(dataset)

    for i in range(n):
        x = dataset.iloc[i].Temperature
        y = dataset.iloc[i].Revenue

        A_gradient += -(2/n) * x * (y - (A*x + B))
        B_gradient += -(2/n) * (y - (A*x + B))

    A = A - L * A_gradient
    B = B - L * B_gradient
    return A, B

def gradient_descent_solution(A, B, dataset, L, epochs):
    for i in range(epochs):
        A, B = gradient_descent(A, B, dataset, L)
    return A, B

def find_approximation_error(A, B, dataset):
    approximation_error = 0
    for i in range(len(dataset)):
        x = dataset.iloc[i].Temperature
        y = dataset.iloc[i].Revenue
        approximation_error += (y - (A*x+B)) **2
    approximation_error = approximation_error / float(len(dataset))
    return approximation_error

A = 0
B = 0
#Finding A and B using analtyical solution
A, B = analytical_solution(data)
#Measure approximation error for analytical solution
approximation_error = find_approximation_error(A, B, data)
print(f'For analtyical solution approximation error is {approximation_error}')
#Represent a function with A and B from analtyical solution
plt.plot(list(range(0, 50)), [A*x+B for x in range(0, 50)], color="red")

A = 0
B = 0
# L is learning rate 
L = 0.001
epochs = 10000

#Finding A and B using gradient descent
A, B = gradient_descent_solution(A, B, data, L, epochs)
#Measure approximation error for gradient descent method
approximation_error = find_approximation_error(A, B, data)
print(f'For gradient descent method approximation error is {approximation_error}')
#Represent a function with A and B using gradient descent
plt.plot(list(range(0, 50)), [A*x+B for x in range(0, 50)], color="green")

#Represent dataset on XY chart
plt.scatter(data.Temperature, data.Revenue)

plt.show()