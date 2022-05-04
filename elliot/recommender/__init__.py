"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from .base_recommender_model import BaseRecommenderModel


from .unpersonalized import Random, MostPop
from .autoencoders import MultiDAE, MultiVAE, EASER
from .graph_based import NGCF, LightGCN, RP3beta
from .knn import ItemKNN, UserKNN, AttributeItemKNN, AttributeUserKNN
from .generic import ProxyRecommender
from .Proxy import ProxyRecommender

