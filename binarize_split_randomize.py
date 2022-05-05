import os.path
from dataset_manipulation import binarize_dataset, apply_randomized_response

RANDOM_SEED = 42

result_main_dir = 'data'

datasets = [{
    'path': 'Movielens1M/MovieLens1M.tsv',
    'dir': 'Movielens1M/',
    'threshold': 3.0,
    'names': ['u', 'i', 'r', 't'],
    'header': None}]

for dataset in datasets:

    dataset_path = dataset['path']
    result_sub_dir = dataset['dir']
    threshold = dataset['threshold']
    names = dataset['names']
    header = dataset['header']
    generated_path = os.path.join(result_main_dir, result_sub_dir, 'generated')

    # binarize dataset: from explicit to implicit feedbacks
    binarized_data, binarized_path = binarize_dataset(
        dataset_path, threshold=threshold, result_main_dir=result_main_dir, result_sub_dir=result_sub_dir,
        names=names, header=header, drop_zeros=True, drop_ratings=False, columns_to_drop=['t'])

    # split the dataset in train and test with random subsampling
    train_path, test_path = binarized_data.train_test_splitting(
        ratio=0.2, result_folder=result_sub_dir, random_seed=RANDOM_SEED)

    # compute differential privacy on training set
    apply_randomized_response(data_path=train_path)

