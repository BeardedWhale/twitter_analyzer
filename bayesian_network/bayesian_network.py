"""
This class implements bayesian network for social media
It involves bayesian network training using Newton-Raphson step
"""
from typing import Dict, Tuple
from scipy.optimize import newton

import numpy as np


class BayesianNetwork():
    __data = {}
    __amount_of_similarity_params = 0
    __theta_size = 0
    __z_size = 0  # equal to number of user pairs

    def __init__(self, data: Dict):
        self.__data = data

    def _perform_newton_rapson_step(self, theta: np.array, z: np.array, w: np.array) -> \
            Tuple[np.array, np.array, np.array]:
        new_theta = 0
        new_z = 0
        new_w = 0

    def _update_theta(self, theta: np.array, z: np.array, w: np.array):
        k = 0


    def _update_z(self, ):
        j = 0
