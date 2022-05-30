import pandas as pd
import os
import argparse

METRICS = ['nDCGRendle2020', 'Recall',
           'HR', 'nDCG', 'Precision', 'F1', 'MAP', 'MAR', 'ItemCoverage', 'Gini',
           'SEntropy', 'EFD', 'EPC', 'PopREO', 'PopRSP', 'ACLT', 'APLT', 'ARP']

DATASET_METRICS = ['n_users', 'n_items', 'density', 'density_log', 'transactions', 'space_size_log', 'shape_log',
                   'gini_item', 'gini_user']

# TODO: this could be more than one
CHOSEN_METRICS = METRICS

PERFORMANCE_PATTERN = 'performance_{dataset}_{model}.tsv'
RESULT_PATTERN = 'delta_{dataset}_{metric}_{model}_eps_{eps}.tsv'

DEFAULT_MODELS = ['EASER', 'ItemKNN', 'MostPop', 'Random', 'RP3beta']

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, type=str, nargs='+')
parser.add_argument('--subfolder', required=False, type=str, default='stats')
parser.add_argument('--model', required=False, type=str, nargs='+', default=DEFAULT_MODELS)
parser.add_argument('--output', required=False, type=str, default='stats/delta/')
parser.add_argument('--eps', required=False, type=float, nargs='+', default=[3, 2, 1, 0.5])

args = parser.parse_args()
datasets = args.dataset
source_dir = args.subfolder
result_dir = args.output
epsilon = [int(eps) if eps%1 == 0 else eps for eps in args.eps]
models = args.model


def generated_dataset_stats(path):
    def gen_from_name(name):
        chuncks = name.split('_')
        if len(chuncks) == 2:
            return -1
        elif len(chuncks) == 3:
            gen = chuncks[-1]
            return int(gen.replace('g', ''))
        else:
            raise ValueError

    data = pd.read_csv(path, sep='\t', header=0)
    data['gen'] = data.name.map(gen_from_name)
    data = data.set_index('gen')
    return data.to_dict(orient='index')

for dataset in datasets:
    for metric in CHOSEN_METRICS:
        stats_path = os.path.join('data', dataset, f'{dataset}_stats.tsv')

        stats = generated_dataset_stats(stats_path)

        if os.path.isdir(result_dir) is False:
            os.makedirs(result_dir)

        performance_files = {m: os.path.join(source_dir,
                                             PERFORMANCE_PATTERN.format(dataset=dataset, model=m)) for m in models}

        for model in models:
            performance = pd.read_csv(performance_files[model], sep='\t', header=0)

            assert all(performance.model == model)
            assert all(performance.dataset == dataset)

            generations = performance.generation.unique()

            header = ['generation', 'epsilon', f'delta_{metric}']
            header += DATASET_METRICS
            eps_results = {eps: [] for eps in epsilon}

            # eps_results['baseline'] = []

            for gen in generations:
                gen_per = performance[performance.generation == gen]
                assert 'baseline' in gen_per.epsilon.values
                baseline_value = gen_per[gen_per.epsilon == 'baseline'][metric].values
                row = [gen, 'baseline', baseline_value[0]]
                row += [stats[gen][m] for m in DATASET_METRICS]

                for eps in epsilon:
                    value = gen_per[gen_per.epsilon == str(eps)][metric].values
                    delta = (baseline_value - value)[0]
                    row = [gen, eps, delta]
                    row += [stats[gen][m] for m in DATASET_METRICS]
                    eps_results[eps].append(row)

            # result = eps_results['baseline']
            # data_result = pd.DataFrame(result, columns=header)
            # result_path = os.path.join(result_dir, RESULT_PATTERN.format(dataset=dataset,
            #                                                              metric=CHOSEN_METRIC,
            #                                                              model=model,
            #                                                              eps='baseline'))
            # data_result.to_csv(result_path, sep='\t', index=False)
            # print(f'result stored at {result_path}')

            for eps, result in eps_results.items():
                data_result = pd.DataFrame(result, columns=header)
                result_path = os.path.join(result_dir, RESULT_PATTERN.format(dataset=dataset,
                                                                                  metric=metric,
                                                                                  model=model,
                                                                                  eps=eps))
                data_result.to_csv(result_path, sep='\t', index=False)
                print(f'result stored at {result_path}')
