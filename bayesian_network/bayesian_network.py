"""
This class implements bayesian network for social media
It involves bayesian network training using Newton-Raphson step
"""
from typing import Dict, Tuple
import numpy as np
import json
import math


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

    def load_data(self, file):
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
                auxilirary_values = self._order_dictionary(data[user]['auxiliary_vector'])
                subusers = data[user]['users']
                for subuser in subusers:
                    pairs.append((user, subuser))
                    a_matrix.append(auxilirary_values)
                    print(list(subusers[subuser]['similarity_vector'].values()))
                    similarity_matrix.append(list(subusers[subuser]['similarity_vector'].values()))
                    y_matrix.append(self._order_dictionary(subusers[subuser]['interaction_vector']))
        return a_matrix, y_matrix, similarity_matrix, pairs

    def _order_dictionary(self, dict):
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
