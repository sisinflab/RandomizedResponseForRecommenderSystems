import numpy as np
import math
import pandas as pd
from scipy.sparse import csr_matrix
from collections import Counter


class RandomizedResponseCoin:

    def __init__(self, first_coin_prb=0.5, second_coin_prb=0.5, first_class=0, second_class=1,
                 first_class_name=None, second_class_name=None, random_seed=42):

        self._array = None
        self._array_len = None

        self._probabilities = None
        if first_class_name is None:
            self.first_class_name = str(first_class_name)
        if second_class_name is None:
            self.second_class_name = str(second_class_name)

        np.random.seed(random_seed)

        self._fcp = first_coin_prb
        self._scp = second_coin_prb

        self._first_class = first_class
        self._second_class = second_class
        self._classes = [first_class, second_class]

        self.set_probabilities()

    def set_probabilities(self):

        probabilities = np.zeros((2, 2))

        for i, data in enumerate(self._classes):
            for j, private in enumerate(self._classes):
                p = 0
                # probability of the chance of making just one flip
                if private == data:
                    p += 1 - self._fcp

                # second flip probabilities
                if private == self._first_class:
                    p += self._fcp * self._scp
                else:
                    p += self._fcp * (1 - self._scp)
                probabilities[i][j] = p

        self._probabilities = probabilities

    def set_array(self, array: np.ndarray):
        self._array = array
        self._array_len = len(self._array)

    def print_probabilities(self):
        for i, data in enumerate(self._classes):
            for j, private in enumerate(self._classes):
                print(f'The probability of responding {private} given {data} is {self._probabilities[i][j]}')

    def randomize(self):

        if self._array is None:
            print('Before randomizing an array MUST be loaded. Please call \'RandomizedResponse.set_array\'.')
            return None
        # toss a coin
        fp = np.random.random(self._array_len)

        # probabilità fp
        randomized_array = np.ma.masked_where(fp < self._fcp, self._array).filled(self._second_class)
        randomized_array = np.ma.masked_where(fp < self._fcp * self._scp, randomized_array).filled(self._first_class)

        self._randomized_array = randomized_array

    def analysis(self):
        not_changed = sum(self._array == self._randomized_array)
        print(f'{not_changed} values changed over {self._array_len} ({not_changed / self._array_len * 100}%)')


class RandomizedResponse:

    def __init__(self, epsilon=0, first_class=0, second_class=1,
                 first_class_name=None, second_class_name=None, random_seed=42):

        self._users_idx = None
        self._items_idx = None
        self._users_names = None
        self._items_names = None

        self._matrix = None
        self._shape = None

        self._eps = epsilon

        self._P = None
        self._pi = None
        self.build_p()

        if first_class_name is None:
            self.first_class_name = str(first_class_name)
        if second_class_name is None:
            self.second_class_name = str(second_class_name)

        np.random.seed(random_seed)

        self._first_class = first_class
        self._second_class = second_class
        self._classes = [first_class, second_class]

        self._users = dict()
        self._items = dict()

        self._randomized_dataset = None

    def set_matrix(self, matrix: np.ndarray):
        self._matrix = matrix

        users = np.unique(matrix[:, 0])
        items = np.unique(matrix[:, 1])

        self._users_idx = dict(zip(users, range(len(users))))
        self._items_idx = dict(zip(items, range(len(items))))

        self._users_names = {v: k for k, v in self._users_idx.items()}
        self._items_names = {v: k for k, v in self._items_idx.items()}

        rows = [self._users_idx[u] for u in matrix[:, 0]]
        cols = [self._items_idx[i] for i in matrix[:, 1]]

        items_counter = Counter(cols)
        pi_1 = np.array([p[1] for p in sorted({i: c/len(users) for i, c in items_counter.items()}.items())])
        self._pi = np.c_[1-pi_1, pi_1]

        n_r = len(users)
        n_c = len(items)

        self._matrix = csr_matrix(([1] * len(rows), (rows, cols)), shape=(n_r, n_c))
        self._shape = self._matrix.shape

    def randomize(self):

        if self._matrix is None:
            print('Before randomizing a matrix MUST be loaded. Please call \'RandomizedResponse.set_matrix\'.')
            return None

        # toss a coin
        fp = np.random.random(self._shape)

        ones_mask = (self._matrix == 1).A

        fp_0 = np.ma.masked_where(ones_mask, fp).filled(1)
        fp_1 = np.ma.masked_where(~ones_mask, fp).filled(1)

        randomized_matrix = np.ma.masked_where(fp_0 < self._P[0, 1], self._matrix.A).filled(1)
        randomized_matrix = np.ma.masked_where(fp_1 < self._P[1, 0], randomized_matrix).filled(0)

        self._randomized_dataset = pd.DataFrame(zip(*csr_matrix(randomized_matrix).nonzero()), columns=['u', 'i'])
        self._randomized_dataset['r'] = 1
        self._randomized_dataset.u = self._randomized_dataset.u.map(lambda x: self._users_names[x])
        self._randomized_dataset.i = self._randomized_dataset.i.map(lambda x: self._items_names[x])

    def lambdas(self):
         self._pi.dot(self._P)

    # def p_from_reported(P, pi):
    #     lambdas = reported_probability(P, pi)
    #     return P * pi.reshape(-1, 1) / lambdas
    #
    # def shannon_entropy(p_from_reported):
    #     entropies = - p_from_reported * np.log2(p_from_reported)
    #     return entropies.sum(axis=0)

    def plausible_deniability(self):
        lambdas = self._pi.dot(self._P)
        p_from_reported = ((self._pi / lambdas)[:, np.newaxis] * self._P)
        entropies = - p_from_reported * np.log2(p_from_reported)
        return entropies.sum(axis=1)

    def epsilon(self):
        return np.max(self._P.max(axis=0) / self._P.min(axis=0))

    def build_p(self, epsilon=None):
        if epsilon is None:
            epsilon = self._eps
        e_eps = math.exp(epsilon)
        den = 1 + e_eps
        self._P = np.array([[e_eps / den, 1 / den], [1 / den, e_eps / den]])

    @property
    def randomized_dataset(self):
        return self._randomized_dataset
