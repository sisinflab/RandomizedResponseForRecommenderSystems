import random
from scipy.sparse import csr_matrix

class DatasetGenerator:
    def __init__(self, dataset: csr_matrix, random_seed=42,
                 min_users=100, max_users=2000, min_items=100, max_items=2000,
                 min_density=0.0007, max_density=0.04):

        random.seed(random_seed)
        self.dataset = dataset
        self._n_users, self._n_items = dataset.shape

        self._min_users = min_users
        self._max_users = max_users
        self._min_items = min_items
        self._max_items = max_items
        self._min_density = min_density
        self._max_density = max_density

    def generate_dataset(self):
        n_users = random.randint(self._min_users, self._max_users)
        users = random.sample(range(self._n_users), k=n_users)
        n_items = random.randint(self._min_items, self._max_items)
        items = random.sample(range(self._n_items), k=n_items)
        generated = self.dataset[users, :][:, items]
        density = len(generated.indices) / (n_items * n_users)
        if density < self._min_density or density > self._max_density:
            generated = self.generate_dataset()
        return generated
