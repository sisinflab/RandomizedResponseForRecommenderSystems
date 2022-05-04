from randomized_response import Dataset, RandomizedResponse

# relative path respect to 'data' folder
#dataset_path = 'Movielens1M/Movielens1M.tsv'
dataset_path = ['binarized/Movielens1M.tsv',
                'binarized/LibraryThing.tsv',
                'binarized/AmazonDigitalMusic.tsv']

for d in dataset_path:
    data = Dataset(d, names=['u', 'i'], header=None)

print()
print()

for eps in [0.5, 1, 2, 3]:
    print(f'eps: {eps}')
    randomizer = RandomizedResponse(epsilon=eps, first_class=1, second_class=0)
    randomizer.set_matrix(data.values())
    randomizer.randomize()

    data.dataset = randomizer.randomized_dataset
    data.info()
    data.export_dataset(parameters={'eps': eps})
