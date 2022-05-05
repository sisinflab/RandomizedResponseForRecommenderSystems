import os.path
from dataset_manipulation import generate_datasets, apply_randomized_response


def generate_and_train(dataset, folder, n=10, random_seed=42, start=0, end=None):

    result_main_dir = 'data'
    result_sub_dir = folder
    train_path = dataset
    generated_path = os.path.join(result_main_dir, result_sub_dir, 'generated')

    # generate new datasets
    generated_paths = generate_datasets(train_path, n=n, data_dir='', result_path=generated_path,
                                        split=True, random_seed=random_seed, start=start, end=end)

    for d_path in generated_paths:
        apply_randomized_response(data_path=d_path[1])
        # start experiment for the generated randomized datasets
