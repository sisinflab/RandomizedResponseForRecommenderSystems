from randomized_response import Dataset, DatasetGenerator

dataset_path = 'Movielens1M/Movielens1M_dataset.tsv'
dataset_path1 = 'LibraryThing/LibraryThing_dataset.tsv'
dataset_path2 = 'AmazonDigitalMusic/AmazonDigitalMusic_dataset.tsv'

for p in [dataset_path, dataset_path1, dataset_path2]:
    data = Dataset(dataset_path, names=['u', 'i', 'r', 't'], header=0)

    print()

    generator = DatasetGenerator(data.sp_ratings)
    generator.generate_dataset()
