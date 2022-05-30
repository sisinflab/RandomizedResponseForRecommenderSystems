from elliot.run import run_experiment
import os

confs = ['AmazonDigitalMusic/adm_baselines2eps.yml']
VARIABLES = ['epsilon',	'delta_HR',	'n_users', 'n_items', 'density', 'density_log', 'transactions',
             'space_size_log', 'shape_log', 'gini_item', 'gini_user']

confs_paths = [os.path.join('config_files', c) for c in confs]

for c in confs_paths:
    assert os.path.isfile(c)

for c in confs_paths:
    run_experiment(c)
