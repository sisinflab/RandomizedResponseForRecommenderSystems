import argparse
from randomized_response import binarize_dataset

# --dataset data/LibraryThing/LibraryThing.tsv --result data/LibraryThing --threshold 6

datasets = [{
    'path': 'Movielens1M/MovieLens1M.tsv',
    'dir': 'Movielens1M/',
    'threshold': 3.0,
    'names': ['u', 'i', 'r', 't'],
    'header': None}]

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--result', required=True, type=str, default='result')
parser.add_argument('--threshold', required=False, type=float, default=3.0)
parser.add_argument('--names', required=False, type=str, nargs='+', default=['u', 'i', 'r', 't'])
parser.add_argument('--header', required=False, type=int, default=None)
parser.add_argument('--seed', required=False, type=int, default=42)

args = parser.parse_args()

dataset_path = args.dataset
result_dir = args.result
threshold = args.threshold
names = args.names
header = args.header
RANDOM_SEED = args.seed

result_sub_dir = 'binarized'

# binarize dataset: from explicit to implicit feedbacks
binarized_data, binarized_path = binarize_dataset(
    data_path=dataset_path, threshold=threshold, result_main_dir=result_dir, result_sub_dir=result_sub_dir,
    names=names, header=header, drop_zeros=True, drop_ratings=False, columns_to_drop=['t'], random_seed=RANDOM_SEED)
