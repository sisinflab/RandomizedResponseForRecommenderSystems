import numpy as np
import pandas as pd
import os
import math
from .paths import DATA_DIR, RESULT_DIR
from scipy.sparse import csr_matrix


class Dataset:

    def __init__(self, data_name, data_dir=None, result_dir=None, header=None, names=None, threshold=3.0):

        self._users = None
        self._n_users = None
        self._items = None
        self._n_items = None
        self._ratings = None
        self._v_ratings = None
        self._n_ratings = None
        self._transactions = None

        self._threshold = threshold

        if data_dir is None:
            data_dir = DATA_DIR

        self._data_name = data_name.split('/')[-1].split('.')[0]
        self._data_path = os.path.join(data_dir, data_name)

        if result_dir is None:
            result_dir = RESULT_DIR
        self._result_dir = result_dir

        self.dataset = None
        self._sp_ratings = None

        self.header = header
        self.names = names

        self._user_col = None
        self._item_col = None
        self._ratings_col = None
        self._timestamp_col = None

        self.set_columns(names)

        self._user_idx = None
        self._item_idx = None

        self._binary = None

        self.load_dataset()
        self.extract_dataset_info()

        # metrics
        self._space_size_log = None
        self._shape_log = None
        self._density = None
        self._density_log = None
        self._gini_item = None
        self._gini_user = None

        self._sorted_items = None
        self._sorted_users = None

    def load_dataset(self):
        self.dataset = pd.read_csv(self._data_path, sep='\t', header=self.header, names=self.names)

    def extract_dataset_info(self):

        assert self.dataset is not None

        self.set_users()
        self.set_items()
        self.set_ratings()

    def set_columns(self, names):
        if len(names) < 2:
            raise KeyError('dataset must have at least two columns: user and item')
        self._user_col = names[0]
        self._item_col = names[1]
        if len(names) > 2:
            self._ratings_col = names[2]
        if len(names) > 3:
            self._timestamp_col = names[3]

    def set_ratings(self, ratings=None):

        if self._ratings_col in self.dataset.columns:

            if ratings is not None:
                self.dataset[self._ratings_col] = ratings

            self._ratings = self.dataset[self._ratings_col]
            self._v_ratings = self.dataset[self._ratings_col].unique()
            self._n_ratings = self.dataset[self._ratings_col].nunique()
            self._binary = True if self._n_ratings <= 2 else False

        # in case ratings column has been dropped
        else:
            self._ratings = pd.Series([], dtype=object)
            self._v_ratings = pd.Series([1])
            self._n_ratings = pd.Series([1])
            self._binary = True

    def set_users(self):
        self._users = self.dataset[self._user_col].unique()
        self._n_users = self.dataset[self._user_col].nunique()

    def set_items(self):
        self._items = self.dataset[self._item_col].unique()
        self._n_items = self.dataset[self._item_col].nunique()

    def binarize(self, drop_zeros=False, drop_ratings=False):
        if not self._binary:
            ratings = self.dataset[self._ratings_col].tolist()
            ratings = np.array(ratings)

            postive_ratings = ratings >= self._threshold
            # reset_ratings
            ratings[:] = 0
            # set positive ratings
            ratings[postive_ratings] = 1
            # copy modified values into the dataframe
            self.dataset[self._ratings_col] = ratings
            # drop rows with negative implicit feedback
            if drop_zeros:
                negative_ratings = self.dataset[self.dataset[self._ratings_col] == 0]
                self.dataset.drop(negative_ratings.index, inplace=True)
            self.set_ratings()
        else:
            print(f'The dataset is already binarized. The classes are: {self._v_ratings}')
        if drop_ratings:
            self.dataset.drop(self._ratings_col, axis=1, inplace=True)

    def export_dataset(self, parameters: dict = None, directory=None, zero_indexed=False):
        if parameters is None:
            parameters = {}
        if directory is None:
            directory = self._result_dir
        else:
            directory = os.path.join(self._result_dir, directory)

        result_name = self._data_name

        if parameters:
            result_name += '_' + '_'.join([f'{k}{v}' for k, v in parameters.items()])
        result_name += '.tsv'

        if os.path.isdir(self._result_dir) is False:
            os.makedirs(self._result_dir)
        result_path = os.path.join(directory, result_name)

        result_path_dir = os.path.dirname(result_path)
        if not os.path.exists(result_path_dir):
            os.makedirs(result_path_dir)

        if zero_indexed:
            dataset = pd.DataFrame()
            dataset[self._user_col] = self.dataset[self._user_col].map(lambda x: self.user_idx.get(x))
            dataset[self._item_col] = self.dataset[self._item_col].map(lambda x: self.item_idx.get(x))
            for c in self.dataset.columns:
                if c not in {self._user_col, self._item_col}:
                    dataset[c] = self.dataset[c]
        else:
            dataset = self.dataset
        dataset.to_csv(result_path, sep='\t', header=False, index=False)

    def drop_column(self, column):
        if column in self.dataset.columns:
            self.dataset.drop(column, axis=1, inplace=True)
        else:
            raise KeyError('column not present in the dataset')

    def info(self):
        self.extract_dataset_info()
        print('-'*40)
        print('Dataset')
        print(f'rows: {len(self.dataset)}')
        for c in self.dataset.columns:
            print(f'{c}: {len(self.dataset[c].unique())} unique values')
        print('-'*40)

    @property
    def values(self):
        return self.dataset.values

    @property
    def users(self):
        return self._users

    @property
    def items(self):
        return self._items

    @property
    def n_users(self):
        return self._n_users

    @property
    def n_items(self):
        return self._n_items

    @property
    def space_size_log(self):
        if self._space_size_log is None:
            scale_factor = 1000
            self._space_size_log = math.log10(self._n_users * self._n_items / scale_factor)
        return self._space_size_log

    @property
    def shape_log(self):
        if self._shape_log is None:
            self._shape_log = math.log10(self._n_users / self._n_items)
        return self._shape_log

    @property
    def density(self):
        if self._density is None:
            self._density = self.transactions / (self._n_users * self._n_items)
        return self._density

    @property
    def density_log(self):
        if self._density_log is None:
            self._density_log = math.log10(self.density)
        return self._density_log

    @property
    def gini_item(self):

        def gini_item_term():
            return (self._n_items + 1 - idx) / (self._n_items + 1) * self.sorted_items[item] / self._transactions

        gini_terms = 0
        for idx, (item, ratings) in enumerate(self.sorted_items.items()):
            gini_terms += gini_item_term()

        self._gini_item = 1 - 2*gini_terms
        return self._gini_item

    @property
    def gini_user(self):

        def gini_user_term():
            return (self._n_users + 1 - idx) / (self._n_users + 1) * self.sorted_users[user] / self._transactions

        gini_terms = 0
        for idx, (user, ratings) in enumerate(self.sorted_users.items()):
            gini_terms += gini_user_term()

        self._gini_user = 1 - 2*gini_terms
        return self._gini_user

    @property
    def sorted_items(self):
        if self._sorted_items is None:
            self._sorted_items = self.dataset.groupby(self._item_col).count()\
                .sort_values(by=[self._user_col]).to_dict()[self._user_col]
        return self._sorted_items

    @property
    def sorted_users(self):
        if self._sorted_users is None:
            self._sorted_users = self.dataset.groupby(self._user_col).count()\
                .sort_values(by=[self._item_col]).to_dict()[self._item_col]
        return self._sorted_users

    @property
    def transactions(self):
        if self._transactions is None:
            self._transactions = len(self.dataset)
        return self._transactions

    @property
    def sp_ratings(self):
        if self._sp_ratings is None:
            row = self.dataset[self._user_col].map(lambda x: self.user_idx.get(x)).to_list()
            col = self.dataset[self._item_col].map(lambda x: self.item_idx.get(x)).to_list()
            self._sp_ratings = csr_matrix(([1]*len(row), (row, col)), shape=(self._n_users, self._n_items), dtype=bool)
        return self._sp_ratings

    @property
    def user_idx(self):
        if self._user_idx is None:
            if self._users is None:
                self.set_users()
            self._user_idx = dict(zip(self._users, range(self._n_users)))
        return self._user_idx

    @property
    def item_idx(self):
        if self._item_idx is None:
            if self._items is None:
                self.set_items()
            self._item_idx = dict(zip(self._items, range(self._n_items)))
        return self._item_idx
