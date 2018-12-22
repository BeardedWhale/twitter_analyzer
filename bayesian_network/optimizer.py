import math
import random

import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch import functional as F

from training import _normalize__by_columns, W_MATRIX_SIZE, Z_MATRIX_SIZE, THETA_MATRIX_SIZE, \
    NUMBER_OF_PAIRS, VARIANCE, NUMBER_OF_INTERACTIONS, LAMBDA_W, LAMBDA_THETA, b, load_data, \
    NUMBER_OF_SIMILARITIES, NUMBER_OF_AUXILIARY_VARIABLES

a_matrix, y_matrix, similarity_matrix, pairs = load_data(file='pairs_data.txt')
mu, sigma = 0.5, 0.5
a_matrix = np.array(a_matrix)
a_norm = _normalize__by_columns(a_matrix)
y_matrix = np.array(y_matrix)
pair_matrix_2 = np.array(pairs)
similarity_matrix = np.array(similarity_matrix)


cost_funct = 0

def dropout(item, p):
    if random.random() < p:
        return 0
    else:
        return item

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
    return -1.0 * log_likelihood



w_loaded = Variable(torch.load('w_updated_pairs.pt'))
z_loaded = Variable(torch.load('z_updated_pairs.pt'))
theta_loaded = Variable(torch.load('theta_updated_pairs.pt'))

p = 0.0001
w_dropout = torch.tensor([dropout(w_loaded[i].item(), p) for i in range(w_loaded.shape[0])], dtype=torch.float64).reshape(w_loaded.shape)
theta_dropout = torch.tensor(
    [[dropout(theta_loaded[i][j].item(), p) for j in range(theta_loaded.shape[1])]for i in range(theta_loaded.shape[0])], dtype=torch.float64).reshape(theta_loaded.shape)
z_dropout = torch.tensor([dropout(z_loaded[i].item(), p) for i in range(z_loaded.shape[0])], dtype=torch.float64).reshape(z_loaded.shape)

THETA_ITER = 30
W_ITER = 10
Z_ITER = 10
w = Variable(w_dropout, requires_grad=True)
z = Variable(z_dropout, requires_grad=True)
theta = Variable(theta_dropout, requires_grad=True)

w_optimizer = torch.optim.Adam([w], lr=0.05)
w_optimizer.zero_grad()
z_optimizer = torch.optim.Adam([z], lr=0.001)
z_optimizer.zero_grad()
theta_optimizer = torch.optim.Adam([theta], lr=0.015)
theta_optimizer.zero_grad()

a = torch.tensor(a_matrix, dtype=torch.double)
y = torch.tensor(y_matrix, dtype=torch.double)
s = torch.tensor(similarity_matrix, dtype=torch.double)
w_prod = torch.mm(w.transpose(0, 1), w)
weight_val = -0.5 * LAMBDA_W * (w_prod[0])
LOG_LIKELIHOOD = torch.Tensor(1,1).normal_()
LOG_LIKELIHOOD.requires_grad_(True)
PREV_LOGLIKELIHOOD = None
theta_diff = torch.tensor([[1000000]], dtype=torch.double)
z_diff = torch.tensor([[10000]], dtype=torch.double)
w_diff = torch.tensor([[100000]], dtype=torch.double)

theta_diffs = []
z_diffs = []
w_diffs = []
theta_converge = 1000
z_converge = 10
w_converge = 10


def draw_plot(theta_diffs, z_diffs, w_diffs, log, z):
    theta_x = np.arange(len(theta_diffs))
    z_x = np.arange(len(z_diffs))
    w_x = np.arange(len(w_diffs))
    z_vals_first = [z[i].item() for i in range(z.shape[0])][:50]
    z_vals_sec = [z[i].item() for i in range(z.shape[0])][50:100]
    print(theta_diffs)
    plt.plot(theta_x, theta_diffs, color = 'red', linestyle='solid')
    plt.plot(z_x, z_diffs, color = 'blue', linestyle='solid')
    plt.plot(w_x, w_diffs, color = 'green', linestyle='solid')
    plt.show()
    plt.plot(np.arange(len(log)), log, color = 'black', linestyle='solid')
    plt.show()
    similarity = [sum( similarity_matrix[i]) for i in range(similarity_matrix.shape[0])][:50]
    similarity_2 = [sum( similarity_matrix[i]) for i in range(similarity_matrix.shape[0])][50:100]
    interaction = [sum(y_matrix[i]) for i in range(y_matrix.shape[0])][:50]
    interaction_2 = [sum(y_matrix[i]) for i in range(y_matrix.shape[0])][50:100]
    auxiliary = [sum(a_matrix[i]) for i in range(a_matrix.shape[0])]
    auxiliary = [auxiliary[i]/sum(auxiliary) for i in range(len(auxiliary))]
    plt.plot(np.arange(len(similarity)), similarity, color='black', linestyle='solid')
    plt.plot(np.arange(len(z_vals_first)), z_vals_first,  color='red', linestyle='solid')
    plt.plot(np.arange(len(interaction)), interaction, color='green', linestyle='solid')
    plt.plot(np.arange(len(auxiliary[:50])), auxiliary[:50], color='blue', linestyle='solid')
    plt.show()
    plt.plot(np.arange(len(similarity_2)), similarity_2, color='black', linestyle='solid')
    plt.plot(np.arange(len(z_vals_sec)), z_vals_sec, color='red', linestyle='solid')
    plt.plot(np.arange(len(interaction_2)), interaction_2, color='green', linestyle='solid')
    plt.plot(np.arange(len(auxiliary[50:100])), auxiliary[50:100], color='blue', linestyle='solid')
    plt.show()




iterations = 10
log_likelihoods = []
prev_log_diff = torch.tensor([[0]], dtype=torch.double)

draw_plot(theta_diffs, z_diffs, w_diffs, log_likelihoods, z)

for i in range(5):
    k = 0
    print('THETA UPDATE:')
    while torch.abs(theta_diff) > theta_converge or k < iterations+1:
        LOG_LIKELIHOOD = calculate_log_likelihood(s, w, theta, z, y, a)
        log_likelihoods.append(LOG_LIKELIHOOD[0].item())
        LOG_LIKELIHOOD.backward()
        theta_optimizer.step()
        if not PREV_LOGLIKELIHOOD:
            PREV_LOGLIKELIHOOD = LOG_LIKELIHOOD
            theta_diff = torch.Tensor([[0]])
        else:
            log_diff = PREV_LOGLIKELIHOOD - LOG_LIKELIHOOD
            theta_diff = prev_log_diff - log_diff
            prev_log_diff = log_diff

            print(f'    diff_{k}: {theta_diff.item()}')
            theta_diffs.append(theta_diff.item())

        print(f'    LG {LOG_LIKELIHOOD}')
        theta.grad.data.zero_()
        k+=1
        if k>THETA_ITER: break
    k = 0
    print('Z UPDATE:')
    while torch.abs(z_diff) > z_converge or k < iterations:
        LOG_LIKELIHOOD = calculate_log_likelihood(s, w, theta, z, y, a)
        log_likelihoods.append(LOG_LIKELIHOOD[0].item())
        LOG_LIKELIHOOD.backward()
        z_optimizer.step()
        log_diff = PREV_LOGLIKELIHOOD - LOG_LIKELIHOOD
        z_diff = prev_log_diff - log_diff
        prev_log_diff = log_diff
        PREV_LOGLIKELIHOOD = LOG_LIKELIHOOD
        if k !=0:
            z_diffs.append(z_diff.item())
        z.grad.data.zero_()
        print(f'    diff_{k}: {z_diff.item()}')
        print(f'    LG: {LOG_LIKELIHOOD}')
        k += 1
        if k>Z_ITER: break

    k=0
    print('W UPDATE:')
    while torch.abs(w_diff) > w_converge or k<iterations:
        w_optimizer.zero_grad()
        LOG_LIKELIHOOD = calculate_log_likelihood(s, w, theta, z, y, a)
        log_likelihoods.append(LOG_LIKELIHOOD[0].item())
        LOG_LIKELIHOOD.backward()
        w_optimizer.step()
        log_diff = PREV_LOGLIKELIHOOD - LOG_LIKELIHOOD
        w_diff = prev_log_diff - log_diff
        prev_log_diff = log_diff
        PREV_LOGLIKELIHOOD = LOG_LIKELIHOOD
        if k!=0:
            w_diffs.append(w_diff.item())
        print(f'    diff_{k}: {w_diff.item()}')
        print(f'    LG: {LOG_LIKELIHOOD}')
        k += 1
        w.grad.data.zero_()
        if k > W_ITER:
            break




    draw_plot(theta_diffs, z_diffs, w_diffs, log_likelihoods, z)

    torch.save(w, 'w_updated_pairs.pt')
    torch.save(theta, 'theta_updated_pairs.pt')
    torch.save(z, 'z_updated_pairs.pt')
