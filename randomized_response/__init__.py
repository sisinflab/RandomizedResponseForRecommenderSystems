import os

PROJECT_PATH = os.path.abspath('.')
DATA_DIR = os.path.join(PROJECT_PATH, 'data')
RESULT_DIR = os.path.join(PROJECT_PATH, 'results/randomized_response')


from .dataset import Dataset
from .randomized_response import RandomizedResponse
