import math

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

from training import _normalize__by_columns, W_MATRIX_SIZE, Z_MATRIX_SIZE, THETA_MATRIX_SIZE, \
    NUMBER_OF_PAIRS, VARIANCE, NUMBER_OF_INTERACTIONS, LAMBDA_W, LAMBDA_THETA, b, load_data, \
    NUMBER_OF_SIMILARITIES, NUMBER_OF_AUXILIARY_VARIABLES

a_matrix, y_matrix, similarity_matrix, pairs = load_data(file='pairs_data.txt')
mu, sigma = 0.5, 0.5
a_matrix = np.array(a_matrix)
a_norm = _normalize__by_columns(a_matrix)
y_matrix = np.array(y_matrix)
similarity_matrix = np.array(similarity_matrix)
weight = np.ones(shape=W_MATRIX_SIZE)
z = np.random.normal(mu, sigma, Z_MATRIX_SIZE)
theta = np.random.normal(mu, sigma, THETA_MATRIX_SIZE)


cost_funct = 0

def _get_sigmoid(pair_index: int, interaction_index: int, theta: np.array, z: np.array, a: np.array) -> float:
    z_i = z[pair_index]
    a_t = a[pair_index, interaction_index].reshape(1)
    u_t = torch.cat((a_t, z_i), 0).reshape(2, 1)
    th = theta[interaction_index]
    th = th.reshape((1, 2))
    exponent_power = (torch.mm(th, u_t)) + b
    return torch.sigmoid(exponent_power)


def calculate_log_likelihood(s, w, theta, z, y, a)->int:
    w_prod = torch.mm(w.transpose(0,1), w)
    log_likelihood =  -0.5 * LAMBDA_W * (w_prod[0][0])
    log_likelihood = log_likelihood.reshape(1)
    for pair in range(NUMBER_OF_PAIRS):
        w_transposed = w.transpose(0,1)
        pair_sim = s[pair].reshape(NUMBER_OF_SIMILARITIES, 1)
        ws_prod = torch.mm(w_transposed, pair_sim)
        # ws_prod = w_transposed.dot(pair_sim)
        z_pair = z[pair]
        z_likelihood = ((ws_prod[0]*z_pair)**2)/2*VARIANCE
        log_likelihood += z_likelihood
        interaction_sum = 0
        for t in range(NUMBER_OF_INTERACTIONS):
            pair_interaction = y[pair][t]
            sub_diff = 1 - pair_interaction
            theta_t = theta[t].reshape(NUMBER_OF_AUXILIARY_VARIABLES+1, 1)
            theta_t_transpose = theta_t.transpose(0, 1)
            a_t = a[pair][t].reshape(1)
            # u_t = np.concatenate(([a_t], z_pair))
            u_t = torch.cat((a_t, z_pair), 0).reshape(2, 1)
            # theta_u_prod = theta_t_transpose.dot(u_t) + b
            theta_u_prod = torch.mm(theta_t_transpose, u_t)
            theta_u_prod += b
            log_sigmoid = _get_sigmoid(pair, a=a, interaction_index=t, theta=theta, z=z)
            log_sigmoid = torch.log(log_sigmoid)
            interaction_val = -1 * sub_diff*theta_u_prod[0] + log_sigmoid
            interaction_sum += interaction_val
        log_likelihood = log_likelihood.reshape(1)
        log_likelihood+=interaction_sum.reshape(1)

        log_likelihood+=0.00001


    for t in range(NUMBER_OF_INTERACTIONS):
        theta_t = theta[t].reshape(NUMBER_OF_AUXILIARY_VARIABLES + 1, 1)
        theta_t_transpose = theta_t.transpose(0, 1)
        theta_t_square = torch.mm(theta_t_transpose, theta_t)
        log_likelihood-= LAMBDA_THETA* 0.5* theta_t_square.reshape(1)

    # log_likelihood = pair_sum + weight_val - theta_val
    return log_likelihood




w = Variable(torch.DoubleTensor(weight), requires_grad=True)
z = Variable(torch.DoubleTensor(z), requires_grad=True)
theta = Variable(torch.DoubleTensor(theta), requires_grad=True)
optimizer = torch.optim.Adam([w, z ,theta], lr = 0.0001)
w_optimizer = torch.optim.Adam([w], lr=0.001)
z_optimizer = torch.optim.Adam([z], lr=0.01)
theta_optimizer = torch.optim.Adam([theta], lr=0.00001)
optimizer.zero_grad()
a = torch.tensor(a_matrix, dtype=torch.double)
y = torch.tensor(y_matrix, dtype=torch.double)
s = torch.tensor(similarity_matrix, dtype=torch.double)
w_prod = torch.mm(w.transpose(0, 1), w)
weight_val = -0.5 * LAMBDA_W * (w_prod[0])
LOG_LIKELIHOOD = torch.Tensor(1,1).normal_()
LOG_LIKELIHOOD.requires_grad_(True)
PREV_LOGLIKELIHOOD = None
theta_diff = torch.Tensor([[1000000]])
z_diff = torch.Tensor([[10000]])
w_diff = torch.Tensor([[100000]])

theta_diffs = []
z_diffs = []
w_diffs = []
theta_converge = 10000
z_converge = 5000
w_converge = 5000


def draw_plot(theta_diffs, z_diffs, w_diffs, log):
    theta_x = np.arange(len(theta_diffs))
    z_x = np.arange(len(z_diffs))
    w_x = np.arange(len(w_diffs))
    print(theta_diffs)
    plt.plot(theta_x, theta_diffs, color = 'red', linestyle='solid')
    plt.plot(z_x, z_diffs, color = 'blue', linestyle='solid')
    plt.plot(w_x, w_diffs, color = 'green', linestyle='solid')
    plt.show()
    plt.plot(np.arange(len(log)), log, color = 'black', linestyle='solid')
    plt.show()



iterations = 10
log_likelihoods = []

for i in range(100):
    k = 0
    print('THETA UPDATE:')
    while theta_diff > theta_converge or k < iterations+1:
        LOG_LIKELIHOOD = calculate_log_likelihood(s, w, theta, z, y, a)
        log_likelihoods.append(LOG_LIKELIHOOD[0].item())
        LOG_LIKELIHOOD.backward()
        theta_optimizer.step()
        if not PREV_LOGLIKELIHOOD:
            PREV_LOGLIKELIHOOD = LOG_LIKELIHOOD
        else:
            theta_diff = torch.abs(PREV_LOGLIKELIHOOD - LOG_LIKELIHOOD)
            print(f'    diff_{k}: {theta_diff.item()}')
            theta_diffs.append(theta_diff.item())

        print(f'    LG {LOG_LIKELIHOOD}')
        k+=1
    k = 0
    print('Z UPDATE:')
    while z_diff > z_converge or k < iterations:
        LOG_LIKELIHOOD = calculate_log_likelihood(s, w, theta, z, y, a)
        log_likelihoods.append(LOG_LIKELIHOOD[0].item())
        LOG_LIKELIHOOD.backward()
        z_optimizer.step()
        z_diff = torch.abs(PREV_LOGLIKELIHOOD - LOG_LIKELIHOOD)
        z_diffs.append(z_diff.item())
        print(f'    diff_{k}: {z_diff.item()}')
        print(f'    LG: {LOG_LIKELIHOOD}')
        k += 1

    k=0
    print('W UPDATE:')
    while w_diff > w_converge or k<iterations:
        LOG_LIKELIHOOD = calculate_log_likelihood(s, w, theta, z, y, a)
        log_likelihoods.append(LOG_LIKELIHOOD[0].item())
        LOG_LIKELIHOOD.backward()
        w_optimizer.step()
        w_diff = torch.abs(PREV_LOGLIKELIHOOD - LOG_LIKELIHOOD)
        w_diffs.append(w_diff.item())
        print(f'    diff_{k}: {w_diff.item()}')
        print(f'    LG: {LOG_LIKELIHOOD}')
        k += 1

    draw_plot(theta_diffs, z_diffs, w_diffs, log_likelihoods)

