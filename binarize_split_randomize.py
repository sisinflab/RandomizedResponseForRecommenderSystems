import os.path
import argparse
from randomized_response import binarize_dataset, apply_randomized_response

RANDOM_SEED = 42

result_main_dir = 'data'

datasets = [{
    'path': 'Movielens1M/MovieLens1M.tsv',
    'dir': 'Movielens1M/',
    'threshold': 3.0,
    'names': ['u', 'i', 'r', 't'],
    'header': None}]

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--folder', required=True, type=str)
parser.add_argument('--threshold', required=False, type=float, default=3.0)
parser.add_argument('--names', required=False, type=str, nargs='+', default=['u', 'i', 'r', 't'])
parser.add_argument('--header', required=False, type=int, default=None)
parser.add_argument('--eps', required=False, type=float, nargs='+', default=None)


args = parser.parse_args()

dataset_path = args.dataset
result_sub_dir = args.folder
threshold = args.threshold
names = args.names
header = args.header
epsilon = args.eps

generated_path = os.path.join(result_main_dir, result_sub_dir, 'generated')

# binarize dataset: from explicit to implicit feedbacks
binarized_data, binarized_path = binarize_dataset(
    dataset_path, threshold=threshold, result_main_dir=result_main_dir, result_sub_dir=result_sub_dir,
    names=names, header=header, drop_zeros=True, drop_ratings=False, columns_to_drop=['t'])

# split the dataset in train and test with random subsampling
train_path, test_path = binarized_data.train_test_splitting(
    ratio=0.2, result_folder=result_sub_dir, random_seed=RANDOM_SEED)

# compute differential privacy on training set
if epsilon:
    print('randomized response...')
    apply_randomized_response(data_path=train_path, epsilon=epsilon)

