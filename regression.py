import os
import pandas as pd
from scipy.stats import zscore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import argparse


FILE_PATTERN = 'delta_{dataset}_{metric}_{model}_eps_{eps}.tsv'
GLOBAL_RESULT_PATTERN = '{result_folder}/regression/reg_glob_{metric}.tsv'
DEF_INDEPENDENT_VARS = ['space_size', 'shape',  'gini_item', 'ratings_per_user', 'ratings_per_item', 'epsilon']
ZSCORED = DEF_INDEPENDENT_VARS
NOT_ZSCORED = []
DEFAULT_MODELS = ['ItemKNN', 'EASER', 'MostPop', 'RP3beta']
VAR_PARAMS = ['pvalues']
MOD_PARAMS = ['rsquared', 'rsquared_adj']
DEFAULT_METRIC = ['Recall']
FEATURE_SELECTION = False


def selected_single_files(directory):
    selected_files = []
    for model in models:
        for dataset in datasets:
            for eps in epsilon:
                file_name = FILE_PATTERN.format(dataset=dataset, metric=metric, model=model, eps=eps)
                path = os.path.join(directory, file_name)
                selected_files.append(path)
    return selected_files


def selected_files(directory, models, datasets, epsilon):
    selected_files = []
    for model in models:
        for dataset in datasets:
            group = []
            for eps in epsilon:
                file_name = FILE_PATTERN.format(dataset=dataset, metric=metric, model=model, eps=eps)
                group.append(os.path.join(directory, file_name))
            selected_files.append((group, {'dataset': dataset, 'model': model}))
    return selected_files


def p_value(model, X, Y):
    from scipy import stats
    lm = model
    params = np.append(lm.intercept_, lm.coef_)
    predictions = lm.predict(X)
    new_X = np.append(np.ones((len(X), 1)), X, axis=1)
    M_S_E = (np.sum((Y.to_numpy() - predictions) ** 2)) / (len(new_X) - len(new_X[0]))
    v_b = M_S_E * (np.linalg.inv(np.dot(new_X.T, new_X)).diagonal())
    s_b = np.sqrt(v_b)
    t_b = params / s_b
    p_val = [2 * (1 - stats.t.cdf(np.abs(i), (len(new_X) - len(new_X[0])))) for i in t_b]
    p_val = np.round(p_val, 3)
    return p_val


def regression(X, Y):

    Xc = zscore(X)
    Xc = sm.add_constant(Xc)
    INDEPENDENT_VARS.insert(0, 'constant')
    Xd = pd.DataFrame(Xc, columns=INDEPENDENT_VARS)
    Y = Y

    est = sm.OLS(Y, Xd)
    result = est.fit()
    print(result.summary())
    return result


def plotting(X, Y, x_var, result):
    X = X[x_var]
    Y = Y.to_numpy().reshape(-1)
    slope = result[x_var]
    intercept = result['constant']
    x = np.linspace(X.min(), X.max(), 100)
    plt.plot(x, slope * x + intercept)
    plt.plot(X, Y, 'bo')
    plt.show()


def feature_selection(ivar, X, Y, topk=5):

    M = np.c_[zscore(X), zscore(Y)]
    corr_values = np.cov(M.T)[-1]
    corr_values = np.absolute(np.delete(corr_values, -1))
    corr_names = dict(zip(corr_values, ivar))
    corr_values[::-1].sort()
    print(corr_values)
    vars = [corr_names[x] for x in corr_values[:topk]]
    return vars


def result_tab(data_result, value='pvalues', folder='regression'):
    result_tab_pattern = '{folder}/regression/tab_{value}_{dataset}_{metric}.tsv'
    DECIMAL_NUMBERS = 5
    #VALUE = 'pvalues'
    #VALUE = 'params'
    VALUE = value
    x = None
    for dataset, result in data_result.items():
        cols = None
        result_data = []
        models = []
        for model_res, model in result:
            models.append(model)
            if x is not None:
                assert model_res.model.exog_names == x
            x = model_res.model.exog_names
            y = model_res.model.endog_names
            reg_metrics = ['rsquared']
            values = model_res.__getattribute__(VALUE)
            row = [values[f] for f in x]
            other_val = [model_res.__getattribute__(metric) for metric in reg_metrics]
            row = other_val + row
            result_data.append(row)
            cols = reg_metrics + x
        result_df = pd.DataFrame(result_data, columns=cols)
        result_df['model'] = models
        result_df.set_index('model', inplace=True)
        result_path = result_tab_pattern.format(value=VALUE, dataset=dataset, metric=y, folder=folder)
        result_df.round(DECIMAL_NUMBERS).to_csv(result_path, sep='\t')
        print(f'regression tab stored at: {result_path}')

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', required=True, type=str, nargs='+')
parser.add_argument('--dir', required=False, type=str, default='stats/delta')
parser.add_argument('--model', required=False, type=str, nargs='+', default=DEFAULT_MODELS)
parser.add_argument('--metric', required=False, type=str, nargs='+', default=DEFAULT_METRIC)
parser.add_argument('--eps', required=False, type=float, nargs='+', default=[3, 2, 0.5])
parser.add_argument('--target', required=False, type=str, default='pvalues') # possible values: pvalues or params
parser.add_argument('--result', required=False, type=str, default='regression_results')

args = parser.parse_args()

datasets = args.dataset
source_directory = args.dir
models = args.model
epsilon = args.eps
target_value = args.target
result_folder = args.result


for metric in args.metric:
    dataset_result = {d: [] for d in datasets}

    global_header = ['dataset', 'model', 'metric', 'R2', 'n_vars', 'p_value_over05']
    global_results = []

    for files, params in selected_files(source_directory, models, datasets, epsilon):

        TARGET_VAR = [f'delta_{metric}']
        dataset = params['dataset']
        model = params['model']

        print('**'*20)
        print(f'Dataset: {dataset}\nModel: {model}\nMetric: {metric}')
        print('**'*20)

        INDEPENDENT_VARS = DEF_INDEPENDENT_VARS
        print(f'running' + ' ' + ' + '.join(files))
        data = pd.DataFrame()
        for f in files:
            new_data = pd.read_csv(f, sep='\t', header=0)
            data = pd.concat([data, new_data])
        data.dropna(inplace=True)
        data.reset_index(inplace=True, drop=True)

        X = data[INDEPENDENT_VARS]
        Y = data[TARGET_VAR]

        Xv = X.values
        Yv = Y.values

        poly = PolynomialFeatures(degree=1, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(Xv)
        poly_names = poly.get_feature_names_out()
        x_names = {v: 'x' + str(idx) for idx, v in enumerate(INDEPENDENT_VARS)}
        names = []
        for p in poly_names:
            value = p
            for name, x_name in x_names.items():
                if value == x_name:
                    value = value.replace(x_name, name)
            names.append(value)

        INDEPENDENT_VARS = names
        Xv = poly_features

        if FEATURE_SELECTION:
            INDEPENDENT_VARS = feature_selection(INDEPENDENT_VARS, Xv, Yv, 7)

        print(f'selected vars: {INDEPENDENT_VARS}')
        var_idx = [idx for idx, name in enumerate(names) if name in set(INDEPENDENT_VARS)]

        ZSCORED = INDEPENDENT_VARS
        result = regression(Xv[:, var_idx], Y)
        global_row = [dataset, model, metric, result.rsquared, len(INDEPENDENT_VARS), sum(result.pvalues > .05)]
        global_results.append(global_row)
        dataset_result[dataset].append((result, model))

    global_results = pd.DataFrame(global_results, columns=global_header)
    glob_result_path = GLOBAL_RESULT_PATTERN.format(result_folder=result_folder, metric=metric)
    if not os.path.exists(os.path.dirname(glob_result_path)):
        os.makedirs(os.path.dirname(glob_result_path))
    global_results.to_csv(glob_result_path, sep='\t', index=False)
    print(f'File stored at: {glob_result_path}')
    result_tab(dataset_result, target_value, folder=result_folder)
