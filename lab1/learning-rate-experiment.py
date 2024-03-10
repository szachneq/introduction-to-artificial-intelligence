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


def main():
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--x", type=float, default=0.0, help="starting x position")
    parser.add_argument("--y", type=float, default=0.0, help="starting y position")
    parser.add_argument("--num", type=int, default=100, help="number of experiments")

    # Parse the command line arguments
    args = parser.parse_args()
    initial_guess = (args.x, args.y)
    num_tests = args.num

    learning_rates = np.linspace(1 / num_tests, 1, num_tests)

    # compute how much iterations were needed for each learning rate
    iteration_amounts = []
    for rate in learning_rates:
        _, num_iterations = gradient_descent(initial_guess, rate)
        iteration_amounts.append(num_iterations)

    # Creating the plot
    plt.plot(learning_rates, iteration_amounts, marker="o", linestyle="-", color="b")

    # Drawing a horizontal line at the minimal point
    min_iterations = min(iteration_amounts)
    plt.axhline(y=min_iterations, color="r", linestyle="--", label="Minimum Iterations")
    # Annotating the minimal value
    plt.text(
        0.9,
        min_iterations,
        f" Min: {min_iterations}",
        verticalalignment="bottom",
        color="red",
    )

    # Filtering values where num iterations is minimal to plot it
    min_x = [
        xi for xi, yi in zip(learning_rates, iteration_amounts) if yi <= min_iterations
    ]
    min_y = [yi for yi in iteration_amounts if yi <= min_iterations]
    plt.plot(min_x, min_y, marker="x", linestyle="-", color="g", label="minimum")

    plt.xlabel("Learning rate")  # Label for the x-axis
    plt.ylabel("Number of iterations")  # Label for the y-axis
    plt.title(
        f"Effect of changing the learning rate on the number of iterations for initial guess ({'{0:.3g}'.format(args.x)},{'{0:.3g}'.format(args.y)})"
    )  # Title of the plot

    # Finding the range where the smallest amount of iterations occurs
    min_start = min(min_x)
    min_end = max(min_x)
    print(
        f"best convergence accomplished for learning rates from {'{0:.3g}'.format(min_start)} to {'{0:.3g}'.format(min_end)}"
    )

    plt.show()  # Displays the plot


if __name__ == "__main__":
    main()
