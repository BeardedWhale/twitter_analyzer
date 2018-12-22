import json

import numpy as np
import torch
import matplotlib.pyplot as plt

from training import _normalize__by_columns, load_data
#
# z = torch.load('z_updated_pairs.pt')
# z_old = torch.load('updated pairs weights/z_updated_pairs.pt')
# num = np.ndarray
# num = z.data.numpy()
# num_old = z_old.data.numpy()
# m = num.min()
# num = num - m
# distance = 1/num
# np.save('pair_distance', num)

num = np.load('pair_distance.npy')

a_matrix, y_matrix, similarity_matrix, pairs = load_data(file='pairs_data.txt')
mu, sigma = 0.5, 0.5

# pairs_d = json.dumps(pairs)

a_matrix = np.array(a_matrix)
a_norm = _normalize__by_columns(a_matrix)
y_matrix = np.array(y_matrix)
interaction = [sum(y_matrix[i]) for i in range(y_matrix.shape[0])][50:100]
# plt.plot(np.arange(len(interaction)), interaction, color='green', linestyle='solid')
# plt.show()
auxiliary = [sum(a_matrix[i]) for i in range(a_matrix.shape[0])]
plt.plot(np.arange(len(auxiliary)), auxiliary, color='blue', linestyle='solid')
plt.show()

pair_matrix_2 = np.array(pairs)
np.save('pairs', pair_matrix_2)
similarity_matrix = np.array(similarity_matrix)
# np.save(num, 'z_updated_pairs')
kek = 5