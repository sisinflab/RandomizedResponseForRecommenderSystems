from randomized_response import Dataset, RandomizedResponse

# relative path respect to 'data' folder
dataset_path = ['Movielens1M/train.tsv',
                'LibraryThing/train.tsv',
                'AmazonDigitalMusic/train.tsv']

for d, directory in zip(dataset_path, ['Movielens1M', 'LibraryThing', 'AmazonDigitalMusic']):
    data = Dataset(d, names=['u', 'i', 'r'], header=None, result_dir='data')

    for eps in [0.5, 1, 2, 3]:
        print(f'eps: {eps}')
        randomizer = RandomizedResponse(epsilon=eps, first_class=1, second_class=0, random_seed=42)
        randomizer.set_matrix(data.values)
        randomizer.randomize()

        data.dataset = randomizer.randomized_dataset
        data.info()
        data.export_dataset(parameters={'eps': eps}, directory=directory)
