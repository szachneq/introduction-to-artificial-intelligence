import numpy as np


def rosenbrock_2d(x, y):
    """
    Rosenbrock function in 2D.

    Parameters:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        float: The value of the Rosenbrock function at the given point.
    """
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def rastrigin_2d(x, y):
    """
    Rastrigin function in 2D.

    Parameters:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        float: The value of the Rastrigin function at the given point.
    """
    return (
        20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))
    )


def bukin_2d(x, y):
    """
    Bukin function in 2D.

    Parameters:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        float: The value of the Bukin function at the given point.
    """
    return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)


def booth_2d(x, y):
    """
    Booth function in 2D.

    Parameters:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        float: The value of the Booth function at the given point.
    """
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def styblinski_tang_2d(x, y):
    """
    Styblinski-Tang function in 2D.

    Parameters:
        x (float): The x-coordinate.
        y (float): The y-coordinate.

    Returns:
        float: The value of the Styblinski-Tang function at the given point.
    """
    return 0.5 * (x**4 - 16 * x**2 + 5 * x + y**4 - 16 * y**2 + 5 * y)


init_ranges = {
    booth_2d: ((-5, 5), (-5, 5)),
    styblinski_tang_2d: ((-5, 5), (-5, 5)),
    rastrigin_2d: ((-5, 5), (-5, 5)),
    rosenbrock_2d: ((-5, 5), (-5, 5)),
    bukin_2d: ((-15, -5), (-3, 3)),
}
