import os.path
from dataset_manipulation import generate_datasets, binarize_dataset, apply_randomized_response

RANDOM_SEED = 42

result_main_dir = 'data'
dataset_to_generate = 100
epsilon = [3, 2, 1, 0.5]

datasets = [{
    'train': 'data/Movielens1M/train.tsv',
    'dir': 'Movielens1M/'}]

for dataset in datasets:

    result_sub_dir = dataset['dir']
    train_path = dataset['train']
    generated_path = os.path.join(result_main_dir, result_sub_dir, 'generated')

    # generate new datasets
    generated_paths = generate_datasets(train_path, n=dataset_to_generate, data_dir='', result_path=generated_path,
                                        split=True, random_seed=RANDOM_SEED, start=15, end=30)

    # compute differential privacy on generated training sets
    for d_path in generated_paths:
        apply_randomized_response(data_path=d_path[1])
