import warnings
import importlib

from .blog_rec import BLOG_REC
from .movie import MOVIE
DATASETS = {
    "BLOG_REC": BLOG_REC,
    "MOVIE": MOVIE 
}
URLS = {}
NUM_LINES = {}
MD5 = {}
for dataset in DATASETS:
    dataset_module_path = "reclab.datasets." + dataset.lower()
    dataset_module = importlib.import_module(dataset_module_path)
    URLS[dataset] = dataset_module.URL
    # NUM_LINES[dataset] = dataset_module.NUM_LINES
    # MD5[dataset] = dataset_module.MD5

__all__ = sorted(list(map(str, DATASETS.keys())))

