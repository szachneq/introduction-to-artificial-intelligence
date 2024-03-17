import argparse
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt


def function(x: float, y: float) -> float:
    return 2 * np.sin(x) + 3 * np.cos(y)

def gradient_descent(
    initial_guess: Tuple[float, float],
    learning_rate: float,
    epsilon=1e-6,
    max_iter=1000,
) -> Tuple[Tuple[float, float], float]:
    """Find local minimum of function

    Args:
        initial_guess (Tuple[float, float]): point at which we start searching for the local minimum
        learning_rate (float): coefficient dictating size of step made during the search
        epsilon (_type_, optional): accuracy at which we consider calculations "good enough". Defaults to 1e-6.
        max_iter (int, optional): maximum allowed amount of iterations. Defaults to 1000.

    Returns:
        Tuple[Tuple[float, float], float]: tuple containing coordinates of the local minima, amount of iterations needed to reach the point
    """
    x, y = initial_guess

    for i in range(1, max_iter + 1):
        # calculate partial derivatives
        df_dx = 2 * np.cos(x)
        df_dy = - 3 * np.sin(y)

        # calculate gradient
        grad = (df_dx, df_dy)

        # find next point to move to
        new_x, new_y = np.array([x, y]) - learning_rate * np.array(grad)

        # calculate displacement vector between current and next position
        displacement_vector = np.array([new_x, new_y]) - np.array([x, y])
        # find length between current and next position
        displacement = np.linalg.norm(displacement_vector)
        # if the length is sufficiently small, return the calculated result
        if displacement < epsilon:
            return (new_x, new_y), i

        # move to the next position
        x, y = new_x, new_y

    # return current position after reaching the iteration limit
    return (x, y), max_iter


LEARNING_RATE = 0.1

def num_iterations(x: float, y: float):
    """Find the number of iterations needed to reach the local minima from given starting point

    Args:
        x (float): x position of the starting point
        y (float): y position of the starting point

    Returns:
        int: number of iterations needed to reach the local minima
    """
    _, n = gradient_descent((x, y), LEARNING_RATE)
    return n

def main() -> None:
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--num", type=int, default=100, help="number of experiments")
    parser.add_argument("--rate", type=float, default=0.85, help="learning rate")

    # Parse the command line arguments
    args = parser.parse_args()
    num_tests = args.num
    global LEARNING_RATE
    LEARNING_RATE = args.rate
    
    # Generate the x and y coordinate arrays
    x = np.linspace(-5, 5, int(np.sqrt(num_tests)))
    y = np.linspace(-5, 5, int(np.sqrt(num_tests)))
    x, y = np.meshgrid(x, y)

    # Define the function to be plotted
    f = np.vectorize(num_iterations) # Vectorize the function
    z = f(x, y) # Apply the function directly to numpy arrays

    # Create the figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface with a colormap
    surf = ax.plot_surface(x, y, z, cmap="viridis")

    # Add a colorbar to show the Z-coordinate color mapping
    fig.colorbar(surf)
    
    plt.xlabel("X")  # Label for the x-axis
    plt.ylabel("Y")  # Label for the y-axis
    plt.title(f"Effect of changing the starting point on the number of iterations for learning rate={LEARNING_RATE}")  # Title of the plot

    # Show the plot
    plt.show()



if __name__ == "__main__":
    main()

