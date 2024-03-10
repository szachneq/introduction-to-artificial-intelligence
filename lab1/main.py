import argparse
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt


def function(x: float, y: float) -> float:
    return 2 * np.sin(x) + 3 * np.cos(y)


def gradient(
    f: Callable[[float, float], float], x: float, y: float, h=1e-5
) -> Tuple[float, float]:
    """Compute gradient of function f at point (x, y) using the central difference method.

    Args:
        f (Callable[[float, float], float]): the function for which we want to compute the gradient
            to ensure reasonable results the function should be continuous function of 2 parameters
        x (float): x position of the point
        y (float): y position of the point
        h (float, optional): step size used during computation. Defaults to 1e-5.

    Returns:
        Tuple[float, float]: computed gradient value
    """
    df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
    df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
    return (df_dx, df_dy)


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
        # find the value of gradient at current position
        grad = gradient(function, x, y)
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


def visualize() -> None:
    """Create a 3d plot of the function"""
    # Generate the x and y coordinate arrays
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    x, y = np.meshgrid(x, y)

    # Define the function to be plotted
    f = np.vectorize(function)
    z = f(x, y)

    # Create the figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the surface with a colormap
    surf = ax.plot_surface(x, y, z, cmap="viridis")

    # Add a colorbar to show the Z-coordinate color mapping
    fig.colorbar(surf)

    plt.xlabel("X")  # Label for the x-axis
    plt.ylabel("Y")  # Label for the y-axis
    plt.title("Visualization of the function in 3D space")  # Title of the plot

    # Show the plot
    plt.show()


def main() -> None:
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--x", type=float, default=0.0, help="starting x position")
    parser.add_argument("--y", type=float, default=0.0, help="starting y position")
    parser.add_argument("--rate", type=float, default=0.1, help="learning rate")

    # Parse the command line arguments
    args = parser.parse_args()
    initial_guess = (args.x, args.y)
    learning_rate = args.rate

    # Compute the result
    minimum, num_iterations = gradient_descent(initial_guess, learning_rate)

    # Print the result
    print(
        f"Minimum approximation with initial guess {initial_guess}: {minimum}, Iterations: {num_iterations}"
    )

    # Visualize the function
    visualize()


if __name__ == "__main__":
    main()
