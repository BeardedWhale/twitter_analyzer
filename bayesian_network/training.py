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
import math

import scipy
import sys
from typing import Tuple
import numpy as np

import json

from scipy.special._ufuncs import expit
from scipy.stats import logistic

VARIANCE = 0.5
LAMBDA_W = 0.5
LAMBDA_THETA = 0.5
NUMBER_OF_SIMILARITIES = 4
NUMBER_OF_INTERACTIONS = 5
NUMBER_OF_PAIRS = 1563
NUMBER_OF_AUXILIARY_VARIABLES = 1
SIMILARITY_MATRIX_SIZE = (NUMBER_OF_PAIRS, NUMBER_OF_SIMILARITIES)
Z_MATRIX_SIZE = (NUMBER_OF_PAIRS, 1)
W_MATRIX_SIZE = (NUMBER_OF_SIMILARITIES, 1)
THETA_MATRIX_SIZE = (NUMBER_OF_INTERACTIONS, NUMBER_OF_AUXILIARY_VARIABLES + 1)  # (n, 2)
Y_MATRIX_SIZE = (NUMBER_OF_PAIRS, NUMBER_OF_INTERACTIONS)
A_MATRIX_SIZE = (NUMBER_OF_PAIRS, NUMBER_OF_INTERACTIONS)
b = 0.1
ALPHA = 0.1


# TODO add initialization, parameters sizes and methods for updating weights

def learning_algorithm(variance: int, w: np.array, s: np.array, z: np.array, y: np.array, theta: np.array,
                       a: np.array) -> Tuple[np.array, np.array, np.array]:
    global ALPHA
    new_theta = theta
    new_z = z
    i = 0
    new_w = w
    error = 3000  # TODO decide error
    dw = sys.maxsize
    while dw > error or i < 20:
        print(f'iteration {i}')
        print("------------")
        dtheta, new_theta = update_theta(new_z, y, new_theta, a)
        print("--Update theta--")
        print(f'dtheta ={dtheta} error ={error}')
        while dtheta > error:
            dtheta, new_theta = update_theta(new_z, y, new_theta, a)
            print("--Update theta in while loop--")
            print(f'dtheta ={dtheta} error ={error}')
            print(f'theta shape={new_theta.shape}')

        print("------------")
        dz, new_z = update_z(variance, new_w, s, new_z, y, new_theta, a)
        print("--Update z--")
        print(f'dz ={dz} error ={error}')
        while dz > error:
            dz, new_z = update_z(variance, new_w, s, new_z, y, new_theta, a)
            print("--Update z in while loop--")
            print(f'dz ={dz} error ={error}')
            print(f'z shape={new_z.shape}')

        print("------------")
        print("--Update w--")
        print(f'dw ={dw} error ={error}')
        new_w = update_w(s, new_z)
        dw = np.abs(np.sum(np.add(w ** 2, -new_w ** 2)))
        w = new_w
        print("--Update w--")
        print(f'dw ={dw} error ={error}')
        i += 1
        ALPHA /=10
        error /= i
    return new_z, new_w, new_theta


def update_theta(z: np.array, y: np.array, theta: np.array,
                 a: np.array) -> np.array:
    dtheta = np.zeros(shape=THETA_MATRIX_SIZE)
    for t in range(NUMBER_OF_INTERACTIONS):
        first_gradient = theta_first_gradient(z, y, theta, a)
        second_gradient = theta_second_gradient(z, y, theta, a)
        denom_t = np.linalg.inv(second_gradient[t])
        dtheta_t = - ALPHA * first_gradient[t].dot(denom_t)  # TODO
        dtheta[t] = dtheta_t
    new_theta = np.add(theta, dtheta)
    dtheta = dtheta ** 2
    dtheta = dtheta.sum()
    return dtheta, new_theta


def update_z(variance: int, w: np.array, s: np.array, z: np.array, y: np.array, theta: np.array,
             a: np.array) -> np.array:
    if theta.shape != (5, 2):
        kek = 30
    denom = z_second_gradient(variance, theta, a, z) ** -1
    first_gradient = z_first_gradient(variance, w, s, z, y, theta, a).transpose()
    dz = - ALPHA * first_gradient.dot(denom)  # TODO

    new_z = np.add(z, dz)
    dz = dz ** 2
    dz = dz.sum()

    return dz, new_z


def update_w(S: np.array, z: np.array) -> np.array:
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


def initialize_parameter(size: Tuple, mu: float, sigma: float) -> np.array:
    """
    Initialize with normal distribution
    :param size:
    :return:
    """

    return np.random.normal(mu, sigma, size=size)


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
            term = y[pair, t] - _get_sigmoid(pair, t, theta, z, a)
            sum = np.add(sum, u_t * term[0])
        term2 = theta_t.dot(LAMBDA_THETA)
        gradients[t] = np.add(sum, term2)
        # print(f'{t/NUMBER_OF_INTERACTIONS*100}%')
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
            try:
                logistic = _get_logistic(pair_index=pair, interaction_index=t, theta=theta, z=z, a=a)
                term1 = y[pair, t] - logistic
            except Exception:
                term1 = sys.maxsize
            term2 = u_t.reshape(NUMBER_OF_AUXILIARY_VARIABLES + 1, 1).dot(
                u_t.reshape(NUMBER_OF_AUXILIARY_VARIABLES + 1, 1).transpose())
            sum = np.add(sum, term2 * term1[0])
        term = identity.dot(LAMBDA_THETA)
        gradients[t] = np.add(-sum, term)
        # print(f'{t/NUMBER_OF_INTERACTIONS*100}%')
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
            theta_parameter_for_z = theta_t[NUMBER_OF_AUXILIARY_VARIABLES]  # number
            intermediate_result = (y_t - _get_sigmoid(i, t, theta, z, a)) * theta_parameter_for_z
            sum_of_interactions += intermediate_result

        z_gradient1 = z_gradient1[0] + sum_of_interactions
        gradient1_values.append(z_gradient1)
        # print(f'{i/NUMBER_OF_INTERACTIONS*100}%')
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
            theta_parameter_for_z = theta_t[NUMBER_OF_AUXILIARY_VARIABLES]  # number
            logistic = _get_logistic(pair_index=i, interaction_index=t, theta=theta, z=z, a=a)
            intermediate_result = (theta_parameter_for_z ** 2) * logistic
            sum_of_interactions += intermediate_result
        z_gradient2 -= sum_of_interactions
        gradient2_values.append(z_gradient2)
        # print(f'{i/NUMBER_OF_INTERACTIONS*100}%')
    gradient2 = np.array(gradient2_values)
    gradient2 = gradient2.reshape(Z_MATRIX_SIZE)

    return gradient2


def _get_sigmoid(pair_index: int, interaction_index: int, theta: np.array, z: np.array, a: np.array) -> float:
    z_i = z[pair_index]
    a_t = a[pair_index, interaction_index]
    u_t = np.concatenate(([a_t], z_i))
    th = theta[interaction_index]
    th = th.reshape((2, 1))
    # th = th.transpose()
    exponent_power = (u_t.dot(th) + b)
    return expit(exponent_power)


def _get_logistic(pair_index: int, interaction_index: int, theta: np.array, z: np.array, a: np.array) -> float:
    z_i = z[pair_index]
    a_t = a[pair_index, interaction_index]
    u_t = np.concatenate(([a_t], z_i))
    th = theta[interaction_index]
    th = th.reshape((2, 1))
    exponent_power = - 1.0 * (u_t.dot(th) + b)  # number; without [0,0] shape: (1,1)
    try:
        log = logistic().pdf(exponent_power)
        return log
    except Exception:
        return -1


def load_data(file):
    """
    Load data from file
    :param file: file to read from
    :return: A_MATRIX shape = (NUMBER_OF_PAIRS, NUMBER_OF_INTERACTIONS),
             SIMILARITY_MATRIX shape = (NUMBER_OF_PAIRS, NUMBER_OF_SIMILARITIES),
             Y_MATRIX shape = (NUMBER_OF_PAIRS, NUMBER_OF_INTERACTIONS),
             list of pairs: (screen_name1, screen_name2)
    """
    similarity_matrix = []
    a_matrix = []
    y_matrix = []
    pairs = []
    with open(file) as f:
        data = json.load(f)
        users = data.keys()
        for user in users:
            auxilirary_values = _order_dictionary(data[user]['auxiliary_vector'])
            subusers = data[user]['users']
            for subuser in subusers:
                pairs.append((user, subuser))
                a_matrix.append(auxilirary_values)
                similarity_matrix.append(list(subusers[subuser]['similarity_vector'].values()))
                y_matrix.append(_order_dictionary(subusers[subuser]['interaction_vector']))
    return a_matrix, y_matrix, similarity_matrix, pairs


def _order_dictionary(dict):
    """
    Orders given dictionary by keys
    Puts result into a new list
    :param dict:
    :return: new list
    """
    ordering = ['comment', 'retweet', 'mention', 'like', 'follow']
    ordered = []
    while len(ordered) != len(ordering):
        for key in dict:
            if len(ordered) == len(ordering):
                break
            if ordering[len(ordered)] in key.lower():
                ordered.append(dict[key])
    return ordered


def calculate_log_likelihood(s, w, theta, z, y, a)->int:
    log_likelihood = 0
    pair_sum = 0
    for pair in range(NUMBER_OF_PAIRS):
        w_transposed = w.transpose()
        pair_sim = s[pair]
        ws_prod = w_transposed.dot(pair_sim)
        z_pair = z[pair]
        z_likelihood = ((ws_prod[0]*z_pair)**2)/2*VARIANCE
        pair_sum += z_likelihood
        interaction_sum = 0
        for t in range(NUMBER_OF_INTERACTIONS):
            pair_interaction = y[pair][t]
            sub_diff = 1 - pair_interaction
            theta_t_transpose = theta[t].transpose()
            a_t = a[pair][t]
            u_t = np.concatenate(([a_t], z_pair))
            theta_u_prod = theta_t_transpose.dot(u_t) + b
            log_sigmoid = _get_sigmoid(pair, a=a, interaction_index=t, theta=theta, z=z)
            interaction_val = -1 * sub_diff*theta_u_prod + math.log(log_sigmoid)
            interaction_sum += interaction_val
        pair_sum+=interaction_sum
    weight_val = -0.5 * LAMBDA_W * (w.transpose().dot(w)[0])
    theta_val = 0
    for t in range(NUMBER_OF_INTERACTIONS):
        theta_val+= LAMBDA_THETA* 0.5* (theta[t].transpose().dot(theta[t]))

    log_likelihood = pair_sum + weight_val - theta_val
    return log_likelihood


def _normalize__by_columns(matrix: np.array):
    result = []
    for i in matrix.transpose():
        if np.sum(i) != 0:
            result.append(i.dot(1 / np.sum(i)))
        else:
            result.append(i)

    return np.array(result).transpose()


a_matrix, y_matrix, similarity_matrix, pairs = load_data(file='pairs_data.txt')
mu, sigma = 0.5, 0.5
a_matrix = np.array(a_matrix)
a_norm = _normalize__by_columns(a_matrix)
y_matrix = np.array(y_matrix)
similarity_matrix = np.array(similarity_matrix)
weight = np.ones(shape=W_MATRIX_SIZE)
z = np.random.normal(mu, sigma, Z_MATRIX_SIZE)
theta = np.random.normal(mu, sigma, THETA_MATRIX_SIZE)
new_z = np.load('z.npy')
new_w = np.load('w.npy')
new_theta = np.load('theta.npy')
# new_z, new_w, new_theta = learning_algorithm(sigma, weight, similarity_matrix, z, y_matrix, theta, a_norm)
log = calculate_log_likelihood(s=similarity_matrix, w=new_w, theta=new_theta, z=new_z, y=y_matrix, a=a_matrix)
print(log)
# print("DONE")
# print("---------")
# print(new_z)
# print("---------")
# print(new_w)
# print("---------")
# print(new_theta)
