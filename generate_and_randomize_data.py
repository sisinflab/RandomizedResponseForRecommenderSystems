import argparse
from randomized_response import generate_and_randomize_datasets

# --dataset data/LibraryThing/binarized/LibraryThing.tsv --result data/LibraryThing --n 600

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--result', required=True, type=str)
parser.add_argument('--n', required=True, type=int)
parser.add_argument('--start', required=False, type=int)
parser.add_argument('--end', required=False, type=int)
parser.add_argument('--seed', required=False, type=int)

args = parser.parse_args()

dataset = args.dataset
result_dir = args.result
n = args.n
start = args.start
end = args.end
random_seed = args.seed

generate_and_randomize_datasets(dataset, result_dir, n, start=start, end=end, random_seed=random_seed)
