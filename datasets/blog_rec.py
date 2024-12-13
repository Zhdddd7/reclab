import os
from typing import Union, Tuple, List
from reclab.datasets.utils import _create_dataset_directory
from reclab.datasets.multiTableDataset import MultiTableDataset

DATASET_NAME = "BLOG_REC"

# 假设有不同split对应不同URL
URL = "https://www.kaggle.com/api/v1/datasets/download/yakshshah/blog-recommendation-data"

EXPECTED_TABLES = ["Author Data.csv", "Blog Ratings.csv", "Medium Blog Data.csv"]


@_create_dataset_directory(dataset_name=DATASET_NAME)
def BLOG_REC(
    root: str, 
    url =  URL,
    delimiter=','
):
    
    zip_path = os.path.join(root, DATASET_NAME)
    extract_folder = os.path.join(root,  "extracted")
    url = URL
    blog_rec_dataset = MultiTableDataset(
        url,
        zip_path,
        extract_folder,
        EXPECTED_TABLES,
        delimiter=delimiter
    )
    return blog_rec_dataset
