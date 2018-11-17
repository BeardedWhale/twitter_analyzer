from typing import Tuple, Callable

import numpy as np

VARIANCE = 0.5
LAMBDA_W = 0.5
LAMBDA_THETA = 0.5
NUMBER_OF_SIMILARITIES = 4
NUMBER_OF_PAIRS = 100
SIMILARITY_MATRIX_SIZE = (NUMBER_OF_PAIRS, NUMBER_OF_SIMILARITIES)
Z_MATRIX_SIZE = (NUMBER_OF_PAIRS, 1)
W_MATRIX_SIZE = (NUMBER_OF_SIMILARITIES, 1)
# TODO add initialization, parameters sizes and methods for updating weights

def perform_newton_rapson_step(theta: np.array, z: np.array, w: np.array) -> Tuple[np.array, np.array, np.array]:

    new_theta = 0
    new_z = 0
    new_w = 0


def update_theta(theta: np.array, z: np.array, w: np.array) -> np.array:
    k = 9


def update_z(theta: np.array, z: np.array, w: np.array) -> np.array:
    k = 4

def update_w(S: np.array, z: np.array, w: np.array) -> np.array:
    """

    :param S:
    :param z:
    :param w:
    :return: W' = (LAMBDA_W*I + S(T)S)^-1 * S(T)z
    """
    s_transposed = S.transpose()
    s_squared = s_transposed.dot(S)
    rows, columns = s_squared.shape
    lambda_matrix = LAMBDA_W * np.identity(rows)
    matrix_sum = lambda_matrix + s_squared
    inverted = np.linalg.inv(matrix_sum)
    w = inverted.dot(s_transposed).dot(z)
    return w


def initialize_normal_distribution(mean: int, variance: int) -> np.array:
    j= 0


def newton_raphson_step(p0: np.array, funct: Callable, fder: Callable, fder2: Callable, *args) -> np.array:
    """
    Above commented code spizhen from scipy.optimize newton method
    :param p0:
    :param funct:
    :param fder:
    :param fder2:
    :param args:
    :return:
    """
    # Multiply by 1.0 to convert to floating point.  We don't use float(x0)
    # so it still works if x0 is complex.
    # p0 = 1.0 * x0
    # if fprime is not None:
    #     # Newton-Rapheson method
    #     for iter in range(maxiter):
    #         fder = fprime(p0, *args)
    #         if fder == 0:
    #             msg = "derivative was zero."
    #             warnings.warn(msg, RuntimeWarning)
    #             return p0
    #         fval = func(p0, *args)
    #         newton_step = fval / fder
    #         if fprime2 is None:
    #             # Newton step
    #             p = p0 - newton_step
    #         else:
    #             fder2 = fprime2(p0, *args)
    #             # Halley's method
    #             p = p0 - newton_step / (1.0 - 0.5 * newton_step * fder2 / fder)
    #         if abs(p - p0) < tol:
    #             return p
    #         p0 = p


def initialize_parameter(size: Tuple, mu: float, sigma: float) -> np.array:
    """
    Initialize with normal distribution
    :param size:
    :return:
    """


    return np.random.normal(mu, sigma, size= size)


n = 4 # number of similarity measures
number_of_pairs = 100
S = initialize_parameter(SIMILARITY_MATRIX_SIZE, 0.5, 0.5)

w = initialize_parameter(W_MATRIX_SIZE, 0.5, 0.5)

z = initialize_parameter(Z_MATRIX_SIZE, 0.5, 0.5)
w_2 = update_w(S, z, w)