from randomized_response import Dataset, RandomizedResponse

# relative path respect to 'data' folder
dataset_path = 'Movielens1M/Movielens1M_train.tsv'

data = Dataset(dataset_path, names=['u', 'i', 'r', 't'], header=0)
data.binarize(drop_zeros=True, drop_ratings=True)
data.drop_timestamp()
data.export_dataset()
data.info()

for eps in [0.5, 1, 2, 3]:
    print(f'eps: {eps}')
    randomizer = RandomizedResponse(epsilon=eps, first_class=1, second_class=0)
    randomizer.set_matrix(data.values())
    randomizer.randomize()

    data.dataset = randomizer.randomized_dataset
    data.info()
    data.export_dataset(parameters={'eps': eps})
