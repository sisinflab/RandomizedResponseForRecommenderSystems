import os
import pandas as pd
import tqdm
import argparse

"""
This script
"""


DEF_MODEL_NAMES = ['Random', 'MostPop', 'ItemKNN', 'EASER', 'RP3beta']
OUTPUT_TEMPLATE = 'performance_{dataset}_{model}.tsv'

parser = argparse.ArgumentParser()

parser.add_argument('--names', required=False, type=str, nargs='+', default=DEF_MODEL_NAMES)
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--subfolder', required=False, type=str, default='results')
parser.add_argument('--output', required=False, type=str, default='stats')
parser.add_argument('--end', required=False, type=int)
parser.add_argument('--seed', required=False, type=int)

args = parser.parse_args()
names = args.names
dataset = args.dataset
sub_folder = args.subfolder
output_folder = args.output


def get_result_parameters(folder_name):
    params = folder_name.split('_')
    name = params[0]
    gen = params[1].replace('gen', '')
    eps = params[2].replace('eps', '')
    return name, gen, eps


def map_models_names(models):
    new_names = []
    for c in models:
        for n in names:
            if n in c:
                new_names.append(n)
    return new_names


def write_on_files(path_and_line):
    for p, l in path_and_line:
        with open(p, 'a') as file:
            file.writelines(l)


output_files_path = {n: os.path.join(output_folder, OUTPUT_TEMPLATE.format(model=n, dataset=dataset)) for n in names}

directory_path = os.path.join('data', dataset, sub_folder)
files_path = os.listdir(directory_path)

if os.path.isdir(output_folder) is False:
    os.makedirs(output_folder)

# write header
fp = files_path[0]
performance_folder = os.path.join(directory_path, fp, 'performance')
intra_results_folders = os.listdir(performance_folder)
result_file = None
for files in intra_results_folders:
    if 'rec_cutoff' in files:
        result_file = os.path.join(performance_folder, files)
assert result_file is not None
data = pd.read_csv(result_file, sep='\t', header=0)
cols = ['dataset', 'model', 'generation', 'epsilon'] + list(data.columns)[1:]
cols_to_string = '\t'.join(cols) + '\n'
cols_to_file = [(p, cols_to_string) for p in output_files_path.values()]
write_on_files(cols_to_file)

for file_path in tqdm.tqdm(files_path):
    result_per_model = {m: '' for m in names}
    result_parameters = get_result_parameters(file_path)
    performance_folder = os.path.join(directory_path, file_path, 'performance')
    intra_results_folders = os.listdir(performance_folder)
    result_file = None
    for files in intra_results_folders:
        if 'rec_cutoff' in files:
            result_file = os.path.join(performance_folder, files)
    assert result_file is not None

    data = pd.read_csv(result_file, sep='\t', header=0)
    data.model = map_models_names(data.model)
    data['dataset'] = result_parameters[0]
    data['generation'] = result_parameters[1]
    data['epsilon'] = result_parameters[2]

    to_write = []
    for row in data[cols].values:
        model = row[1]
        string_row = [str(el) for el in row]
        row_to_write = '\t'.join(string_row) + '\n'
        to_write.append((output_files_path[model], row_to_write))
    write_on_files(to_write)

for path in output_files_path.values():
    print(f'file stored at {path}')
