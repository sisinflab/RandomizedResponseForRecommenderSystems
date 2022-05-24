import pandas as pd
import os

METRICS = ['nDCGRendle2020', 'Recall',
           'HR', 'nDCG', 'Precision', 'F1', 'MAP', 'MAR', 'ItemCoverage', 'Gini',
           'SEntropy', 'EFD', 'EPC', 'PopREO', 'PopRSP', 'ACLT', 'APLT', 'ARP']

DATASET_METRICS = ['n_users', 'n_items', 'density', 'density_log', 'transactions', 'space_size_log', 'shape_log',
                   'gini_item', 'gini_user']

CHOSEN_METRIC = 'HR'

models = ['EASER', 'ItemKNN', 'MostPop', 'Random', 'RP3beta']

dataset = 'LibraryThing'
source_dir = 'stats'
performance_pattern = 'performance_{dataset}_{model}.tsv'
result_dir = 'stats/delta/'
result_name_pattern = 'delta_{dataset}_{metric}_{model}_eps_{eps}.tsv'
stats_path = os.path.join('data', dataset, f'{dataset}_stats.tsv')
epsilon = [3, 2, 1, 0.5]


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
    data.set_index('gen')
    return data.to_dict(orient='index')


stats = generated_dataset_stats(stats_path)

if os.path.isdir(result_dir) is False:
    os.makedirs(result_dir)

performance_files = {m: os.path.join(source_dir,
                                     performance_pattern.format(dataset=dataset, model=m)) for m in models}

for model in models:
    performance = pd.read_csv(performance_files[model], sep='\t', header=0)

    assert all(performance.model == model)
    assert all(performance.dataset == dataset)

    generations = performance.generation.unique()

    header = ['generation', 'epsilon', f'delta_{CHOSEN_METRIC}']
    header += DATASET_METRICS
    eps_results = {eps: [] for eps in epsilon}

    for gen in generations:
        gen_per = performance[performance.generation == gen]
        assert 'baseline' in gen_per.epsilon.values
        baseline_value = gen_per[gen_per.epsilon == 'baseline'][CHOSEN_METRIC].values
        for eps in epsilon:
            value = gen_per[gen_per.epsilon == str(eps)][CHOSEN_METRIC].values
            delta = (baseline_value - value)[0]
            row = [gen, eps, delta]
            row += [stats[gen][m] for m in DATASET_METRICS]
            eps_results[eps].append(row)

    for eps, result in eps_results.items():
        data_result = pd.DataFrame(result, columns=header)
        result_path = os.path.join(result_dir, result_name_pattern.format(dataset=dataset,
                                                                          metric=CHOSEN_METRIC,
                                                                          model=model,
                                                                          eps=eps))
        data_result.to_csv(result_path, sep='\t', index=False)
        print(f'result stored at {result_path}')
