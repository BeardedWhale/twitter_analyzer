"""
This module implements training algorithm for bayesian network modelling socialmedia relationships
For more details check this paper: https://dl.acm.org/citation.cfm?id=1772790

MODEL PARAMETERS:
- z: matrix which maps relationship strength: [0, 1] with each pair of users      [LATENT]
- s: similarity matrix associates each pair of users with similarity vector       [OBSERVED]
- w: weight matrix for similarity parameters                                      [LATENT]
- y: interactions matrix associates each pair of users with interaction vector    [OBSERVED]
- a: auxiliary parameters for interactions                                        [OBSERVED]
- u: parent vector for y(t)<-u = [a(t), z]
- theta: weights for u                                                            [LATENT]

To obtain latent variable z and infer latent parameters theta and w we will use iterative updates
for each of them.

ALGORITHM:
While not converged:
1. For each Newton-Raphson step:
    For t = 1,...,m: update θt based on first and second gradient (Eq. (7) and Eq. (10)) m - number of interactions
2. For each Newton-Raphson step:
    For (i, j) ∈ D: update z(ij) based on first and second gradient (Eq. (6) and Eq. (9))
3. Update w based on Eq. (11)

Note: relationships between users are not bidirectional, i.e. z(i,j)!=z(j,i)
"""

from typing import Tuple, Callable
from scipy.optimize import newton
import numpy as np

from math import exp

VARIANCE = 0.5
LAMBDA_W = 0.5
LAMBDA_THETA = 0.5
NUMBER_OF_SIMILARITIES = 4
NUMBER_OF_INTERACTIONS = 5
NUMBER_OF_PAIRS = 100
NUMBER_OF_AUXILIARY_VARIABLES = 1
SIMILARITY_MATRIX_SIZE = (NUMBER_OF_PAIRS, NUMBER_OF_SIMILARITIES)
Z_MATRIX_SIZE = (NUMBER_OF_PAIRS, 1)
W_MATRIX_SIZE = (NUMBER_OF_SIMILARITIES, 1)
THETA_MATRIX_SIZE = (NUMBER_OF_INTERACTIONS, NUMBER_OF_AUXILIARY_VARIABLES + 1)  # (n, 2)
Y_MATRIX_SIZE = (NUMBER_OF_PAIRS, NUMBER_OF_INTERACTIONS)
A_MATRIX_SIZE = (NUMBER_OF_PAIRS, NUMBER_OF_INTERACTIONS)
b = 0.1


# TODO add initialization, parameters sizes and methods for updating weights

def perform_newton_rapson_step(theta: np.array, z: np.array, w: np.array) -> Tuple[np.array, np.array, np.array]:
    new_theta = 0
    new_z = 0
    new_w = 0


def update_theta(theta: np.array, z: np.array, w: np.array) -> np.array:
    k = 0


def update_z(theta: np.array, z: np.array, w: np.array) -> np.array:
    k = 4


def update_w(S: np.array, z: np.array, w: np.array) -> np.array:
    """

    :param S: similarity matrix, size: (NUMBER_OF_PAIRS, NUMBER_OF_SIMILARITIES)
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
    j = 0


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

    return np.random.normal(mu, sigma, size=size)


VARIANCE = 0.5
LAMBDA_W = 0.5
LAMBDA_THETA = 0.5
NUMBER_OF_SIMILARITIES = 4
NUMBER_OF_INTERACTIONS = 5
NUMBER_OF_PAIRS = 100
NUMBER_OF_AUXILIARY_VARIABLES = 1
SIMILARITY_MATRIX_SIZE = (NUMBER_OF_PAIRS, NUMBER_OF_SIMILARITIES)
Z_MATRIX_SIZE = (NUMBER_OF_PAIRS, 1)
W_MATRIX_SIZE = (NUMBER_OF_SIMILARITIES, 1)
THETA_MATRIX_SIZE = (NUMBER_OF_INTERACTIONS, NUMBER_OF_AUXILIARY_VARIABLES + 1)  # (n, 2)
Y_MATRIX_SIZE = (NUMBER_OF_PAIRS, NUMBER_OF_INTERACTIONS)
A_MATRIX_SIZE = (NUMBER_OF_PAIRS, NUMBER_OF_INTERACTIONS)
b = 0.1


def theta_first_gradient(z: np.array, y: np.array, theta: np.array,
                         a: np.array) -> np.array:
    """
    Equation (7)
    :param z: relationship matrix  shape: (NUMBER_OF_PAIRS, 1)
    :param y: matrix of interactions, (NUMBER_OF_PAIRS, NUMBER_OF_INTERACTIONS)
    :param theta: matrix of interaction weights (NUMBER_OF_INTERACTIONS, NUMBER_OF_AUXILIARY_VERIABLES + 1)
    :param a: matrix of auxiliary values for interactions
    :return: 2d array, where t-th element gradient for theta_t
    """
    gradients = np.zeros(shape=THETA_MATRIX_SIZE)
    for t in range(NUMBER_OF_INTERACTIONS):
        theta_t = theta[t]  # shape = (NUMBER_OF_AUXILIARY_VARIABLES+1,)
        sum = np.zeros(shape=(NUMBER_OF_AUXILIARY_VARIABLES + 1,))
        for pair in range(NUMBER_OF_PAIRS):
            z_pair = z[pair]
            a_t = a[pair, t]
            u_t = np.concatenate(([a_t], z_pair))
            term = y[pair, t] - 1.0 / (1.0 - _get_exponent(pair, t, theta_t, z, a))
            sum = np.add(sum, u_t.dot(term))
        term2 = theta_t.dot(LAMBDA_THETA)
        gradients[t] = np.add(sum, term2)
    return gradients


def theta_second_gradient(z: np.array, y: np.array, theta: np.array,
                          a: np.array) -> np.array:
    """
    Equation (10)
    :param z: relationship matrix  shape: (NUMBER_OF_PAIRS, 1)
    :param y: matrix of interactions, (NUMBER_OF_PAIRS, NUMBER_OF_INTERACTIONS)
    :param theta: matrix of interaction weights (NUMBER_OF_INTERACTIONS, NUMBER_OF_AUXILIARY_VERIABLES + 1)
    :param a: matrix of auxiliary values for interactions
    :return: 3d array, where t-th element second gradient(square matrix) for thetha_t
    """
    gradients = np.zeros(
        shape=(NUMBER_OF_INTERACTIONS, NUMBER_OF_AUXILIARY_VARIABLES + 1, NUMBER_OF_AUXILIARY_VARIABLES + 1))
    identity = np.identity(NUMBER_OF_AUXILIARY_VARIABLES + 1)
    for t in range(NUMBER_OF_INTERACTIONS):
        theta_t = theta[t]  # shape = (NUMBER_OF_AUXILIARY_VARIABLES+1,)
        sum = np.zeros(shape=(NUMBER_OF_AUXILIARY_VARIABLES + 1, NUMBER_OF_AUXILIARY_VARIABLES + 1))
        for pair in range(NUMBER_OF_PAIRS):
            z_pair = z[pair]
            a_t = a[pair, t]
            u_t = np.concatenate(([a_t], z_pair))
            exp = _get_exponent(pair, t, theta_t, z, a)
            term1 = y[pair, t] - exp / (1.0 - exp) ** 2
            term2 = u_t.reshape(NUMBER_OF_AUXILIARY_VARIABLES + 1, 1).dot(
                u_t.reshape(NUMBER_OF_AUXILIARY_VARIABLES + 1, 1).transpose())
            sum = np.add(sum, term2.dot(term1))
        term = identity.dot(LAMBDA_THETA)
        gradients[t] = np.add(-sum, term)
    return gradients


def z_first_gradient(variance: int, w: np.array, s: np.array, z: np.array, y: np.array, theta: np.array,
                     a: np.array) -> np.array:
    """
    Computes first graient of relationships parameter
    :param variance: number, constant
    :param w: vector of size (NUMBER_OF_SIMILARITIES, 1)
    :param s: matrix of similarities for each pair (NUMBER_OF_PAIRS, NUMBER_OF_SIMILARITIES)
    :param a: matrix of auxiliary values for interactions
    :param z: relationship matrix  shape: (NUMBER_OF_PAIRS, 1)
    :param y: matrix of interactions, (NUMBER_OF_PAIRS, NUMBER_OF_INTERACTIONS)
    :param theta: matrix of interaction weights (NUMBER_OF_INTERACTIONS, NUMBER_OF_AUXILIARY_VERIABLES + 1)
    :return: gradient values
    """

    gradient1_values = []
    for i, z_ith in enumerate(z[:, 0]):  # z_ith shape: (1, 1)
        z_gradient1 = (1.0 / variance) * (w.transpose().dot(s[i])) - z_ith  # current shape: (1, 1)

        sum_of_interactions = 0
        for t in range(NUMBER_OF_INTERACTIONS):
            y_t = y[i, t]  # number
            theta_t = theta[t]  # shape: (1, 2)
            theta_parameter_for_z = theta_t[0, NUMBER_OF_AUXILIARY_VARIABLES]  # number
            exponent = _get_exponent(i, t, theta, z, a)
            intermediate_result = (y_t - 1.0 / (1.0 + exponent)) * theta_parameter_for_z
            sum_of_interactions += intermediate_result

        z_gradient1 = z_gradient1[0, 0] + sum_of_interactions
        gradient1_values.append(z_gradient1)
    gradient1 = np.array(gradient1_values)
    gradient1 = gradient1.reshape(Z_MATRIX_SIZE)
    return gradient1


def z_second_gradient(variance: int, theta: np.array, a: np.array, z: np.array) -> np.array:
    """
    Computes second graient of relationships parameter
    :param variance:  number, constant
    :param theta: matrix of interaction weights (NUMBER_OF_INTERACTIONS, NUMBER_OF_AUXILIARY_VERIABLES + 1)
    :param a: matrix of auxiliary values for interactions
    :param z: relationship matrix  shape: (NUMBER_OF_PAIRS, 1)
    :return: second gradient values
    """
    gradient2_values = []
    for i, z_ith in enumerate(z[:, 0]):
        z_gradient2 = - 1.0 / variance

        sum_of_interactions = 0
        for t in range(NUMBER_OF_INTERACTIONS):
            theta_t = theta[t]  # shape: (1, 2)
            theta_parameter_for_z = theta_t[0, NUMBER_OF_AUXILIARY_VARIABLES]  # number
            exponent = _get_exponent(i, t, theta, z, a)
            intermediate_result = (theta_parameter_for_z ** 2) * exponent / ((1 + exponent) ** 2)
            sum_of_interactions += intermediate_result
        z_gradient2 -= sum_of_interactions
        gradient2_values.append(z_gradient2)
    gradient2 = np.array(gradient2_values)
    gradient2 = gradient2.reshape(Z_MATRIX_SIZE)

    return gradient2


def _get_exponent(pair_index: int, interaction_index: int, theta: np.array, z: np.array, a: np.array) -> float:
    """
    computes special exponent value that was defined in Eq. (3) and used further in all gradients
    :param pair_index: index of user pair
    :param interaction_index: index of certain interaction
    :param theta: matrix of interaction weights (NUMBER_OF_INTERACTIONS, NUMBER_OF_AUXILIARY_VERIABLES + 1)
    :param z: relationship matrix  shape: (NUMBER_OF_PAIRS, 1)
    :param a: matrix of auxiliary values for interactions
    :return: value of exponent
    """
    z_i = z[pair_index]
    a_t = a[pair_index, interaction_index]
    u_t = np.concatenate((a_t, z_i))  # TODO: a_t это же число?
    # TODO: если да, надо поменять на u_t = np.concatenate(([a_t], z_i)) а то ругается
    exponent_power = - 1.0 * (theta.dot(u_t) + b)[0, 0]  # number; without [0,0] shape: (1,1)
    return exp(exponent_power)


# TODO Remove
n = 4  # number of similarity measures
number_of_pairs = 100
S = initialize_parameter(SIMILARITY_MATRIX_SIZE, 0.5, 0.5)

w = initialize_parameter(W_MATRIX_SIZE, 0.5, 0.5)

z = initialize_parameter(Z_MATRIX_SIZE, 0.5, 0.5)
