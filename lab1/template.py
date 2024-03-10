import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def function(x, y):
    return 2*np.sin(x)+3*np.cos(y)

def gradient_descent(initial_guess, learning_rate, tol=1e-6, max_iter=1000):
    """
    Gradient descent algorithm
    
    Parameters:
    - initial_guess: initial 2D coordinate vector
    - learning_rate: learning rate
    - tol: tolerance, convergence criteria
    - max_iter: maximum number of iterations

    """

def visualize():
    """
    Visualization function: creates 3D plot of the function. Use colors to show the Z-coordinate
    """

visualize()

#Example usage:
initial_guess_1 = [2.0, 2.0]
learning_rate_1 = 0.1
minimum_1, iterations_1 = gradient_descent(initial_guess_1, learning_rate_1)


print(f"Minimum approximation with initial guess {initial_guess_1}: {minimum_1}, Iterations: {iterations_1}")
