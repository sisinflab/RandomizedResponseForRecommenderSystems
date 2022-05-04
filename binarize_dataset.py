from randomized_response import Dataset

# relative path respect to 'data' folder
dataset_path = ['Movielens1M/Movielens1M.tsv',
                'LibraryThing/LibraryThing.tsv',
                'AmazonDigitalMusic/AmazonDigitalMusic.tsv']

for path, threshold in zip(dataset_path, [3.0, 6.0, 3.0]):
    data = Dataset(path, names=['u', 'i', 'r', 't'], header=None, threshold=threshold, result_dir='data')
    data.binarize(drop_zeros=True, drop_ratings=True)
    data.drop_column('t')
    data.export_dataset(zero_indexed=True, directory='binarized')
    data.info()
