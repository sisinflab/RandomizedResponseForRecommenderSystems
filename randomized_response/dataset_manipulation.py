import random

from randomized_response import Dataset, DatasetGenerator, RandomizedResponse
from randomized_response.paths import RESULT_DIR, DATA_DIR
import pandas as pd
import os

METRICS = ['n_users', 'n_items', 'density', 'density_log',
           'transactions', 'space_size_log', 'shape_log',
           'gini_item', 'gini_user']


def binarize_dataset(data_path, threshold=3, result_main_dir=None, result_sub_dir=None, names=None, header=0,
                     drop_zeros=True, drop_ratings=False, columns_to_drop=None, random_seed=42):

    if result_main_dir is None:
        result_main_dir = os.path.dirname(data_path)
    if result_sub_dir is None:
        result_sub_dir = ''
    if names is None:
        names = ['u', 'i', 'r']
    if columns_to_drop is None:
        columns_to_drop = []

    result_path = os.path.join(result_main_dir, result_sub_dir)

    print('-'*50 + '\n'
          'Binarizing dataset\n'
          f'Dataset path: {data_path}\n'
          f'Rating threshold: {threshold}\n'
          f'Results could be found in the following directory: \'{result_path}\'\n')

    data = Dataset(data_path, data_dir='',  names=names, header=header, threshold=threshold, result_dir=result_main_dir)
    data.binarize(drop_zeros=drop_zeros, drop_ratings=drop_ratings)
    for col_name in columns_to_drop:
        data.drop_column(col_name)
    data.info()
    result_path = data.export_dataset(zero_indexed=False, result_folder=result_sub_dir, parameters={})
    return data, result_path


def generate_datasets(dataset_name, n, data_dir=None, result_path=None, names=None, header=None,
                      split=False, train_test_ratio=0.2, random_seed=42, start=0, end=None):

    if start is None:
        start = 0
    if end is None:
        end = n
    if random_seed is None:
        random_seed = 42

    assert start <= end
    random_seeds = [random_seed + idx for idx in range(n)]

    if result_path is None:
        result_path = RESULT_DIR
    if names is None:
        names = ['u', 'i', 'r']
    if data_dir is None:
        data_dir = DATA_DIR

    stats = []
    results_path = []
    data = Dataset(dataset_name, data_dir=data_dir, names=names, header=header)
    stats.extend(data.get_metrics(METRICS))
    for idx in range(start, end):
        g_stats, g_paths = generate_and_split_sub_dataset(
            data, random_seeds[idx], idx, result_path, train_test_ratio, split)
        stats.append(g_stats)
        results_path.append(g_paths)

    stats_path = os.path.join(result_path, f'stats_{start}_{end}.tsv')
    pd.DataFrame(stats[1:], columns=stats[0]).to_csv(stats_path, sep='\t', index=False)
    print(f'stats stored at: \'{stats_path}\'')
    return results_path


def generate_and_split_sub_dataset(data, seed, idx, result_path, ratio, split):
    generated_random_seed = seed
    train_test_ratio = ratio
    generator = DatasetGenerator(data.sp_ratings, random_seed=generated_random_seed)
    generated_matrix = generator.generate_dataset()
    generated_dataset = Dataset(generated_matrix, result_dir=result_path)
    generated_dataset.name = data.name + f'_g{idx}'
    result_folder = f'{idx}'
    generated_path = generated_dataset.export_dataset(result_folder=result_folder)
    train_name = f'train_{idx}.tsv'
    test_name = f'test_{idx}.tsv'
    if split:
        train_path, test_path = generated_dataset.train_test_splitting(
            ratio=train_test_ratio, train_name=train_name, test_name=test_name, random_seed=generated_random_seed,
            result_folder=result_folder)
        returned_path = (generated_path, train_path, test_path)

        if train_path is None or test_path is None:
            random.seed(seed)
            new_seed = random.randint(0, 100000)
            random.seed(new_seed)
            return generate_and_split_sub_dataset(data, new_seed, idx, result_path, ratio, split)
    else:
        returned_path = [generated_path]

    return generated_dataset.get_metrics(METRICS)[1], returned_path


def apply_randomized_response(data_path, epsilon=None):

    if epsilon is None:
        epsilon = [0.5, 1, 2, 3]

    data_folder = os.path.dirname(data_path)
    data_name = os.path.basename(data_path)

    print(f'RANDOMIZED RESPONSE: applying randomized response on \'{data_name}\'\n'
          f'with epsilon values: {epsilon}')

    data = Dataset(data_name, data_dir=data_folder, names=['u', 'i', 'r'], header=None, result_dir=data_folder)

    for eps in epsilon:
        randomizer = RandomizedResponse(epsilon=eps, random_seed=42)
        randomizer.set_matrix(data.values)
        randomizer.randomize()

        data.dataset = randomizer.randomized_dataset
        data.export_dataset(parameters={'eps': eps})


def generate_and_randomize_datasets(dataset, folder, n=10, random_seed=42, start=0, end=None):

    result_main_dir = ''
    result_sub_dir = folder
    train_path = dataset
    generated_path = os.path.join(result_main_dir, result_sub_dir, 'generated')

    # generate new datasets
    generated_paths = generate_datasets(train_path, n=n, data_dir='', result_path=generated_path,
                                        split=True, random_seed=random_seed, start=start, end=end)
    # randomize the new datasets
    for d_path in generated_paths:
        apply_randomized_response(data_path=d_path[1])
