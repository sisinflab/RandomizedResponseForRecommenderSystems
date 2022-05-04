from randomized_response import Dataset


dataset_path = ['Movielens1M/Movielens1M_binarized.tsv',
                'LibraryThing/LibraryThing_binarized.tsv',
                'AmazonDigitalMusic/AmazonDigitalMusic_binarized.tsv']

for path, folder in zip(dataset_path, ['Movielens1M', 'LibraryThing', 'AmazonDigitalMusic']):
    data = Dataset(path, names=['u', 'i', 'r'], header=None, result_dir='data')
    data.train_test_splitting(ratio=0.2, result_folder=folder, random_seed=42)
