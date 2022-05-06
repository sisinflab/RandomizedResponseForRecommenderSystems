from elliot.run import run_experiment
from config_template import TEMPLATE
import os
import argparse

config_dir = 'config_files/'
RANDOM_SEED = 42

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, type=str, nargs='+')
parser.add_argument('--start', required=False, type=int)
parser.add_argument('--end', required=False, type=int)
parser.add_argument('--eps', required=True, type=str, nargs='+')

args = parser.parse_args()

datasets = args.dataset
start = args.start
end = args.end
epsilons = args.eps

# check if datasets exist
for dataset in datasets:
    for gen in range(start, end):
        for eps in epsilons:
            train = os.path.exists(f'./data/{dataset}/generated/{gen}/train_{gen}_eps{eps}.tsv')
            test = os.path.exists(f'./data/{dataset}/generated/{gen}/test_{gen}.tsv')
            if not train:
                raise FileNotFoundError(f'Missing data set for {dataset} at gen {gen} with eps {eps}')
            if not test:
                raise FileNotFoundError(f'Missing test set for {dataset} at gen {gen} with eps {eps}')

# run experiments on each generated dataset
for dataset in datasets:
    for gen in range(start, end):
        for eps in epsilons:
            config = TEMPLATE.format(dataset=dataset, generated=gen, epsilon=eps)
            config_path = os.path.join(config_dir, 'runtime_conf.yml')
            with open(config_path, 'w') as file:
                file.write(config)
            run_experiment(config_path)
