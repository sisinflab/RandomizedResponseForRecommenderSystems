from elliot.run import run_experiment
import os

confs = ['config_files/AmazonDigitalMusic/adm_baselines2.yml']

confs_paths = [os.path.join('config_files', c) for c in confs]

for c in confs_paths:
    assert os.path.isfile(c)

for c in confs_paths:
    run_experiment(c)
