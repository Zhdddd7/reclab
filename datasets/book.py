import os
from typing import Union, Tuple, List
from reclab.datasets.utils import _create_dataset_directory
from reclab.datasets.multiTableDataset import MultiTableDataset

DATASET_NAME = "BOOK"

# 假设有不同split对应不同URL
URL = "https://www.kaggle.com/api/v1/datasets/download/arashnic/book-recommendation-dataset"

EXPECTED_TABLES = ["Books.csv", "Ratings.csv", "Users.csv"]


@_create_dataset_directory(dataset_name=DATASET_NAME)
def BOOK(
    root: str, 
    url =  URL,
    delimiter=','
):
    
    zip_path = os.path.join(root, DATASET_NAME)
    extract_folder = os.path.join(root,  "extracted")
    url = URL
    book_dataset = MultiTableDataset(
        url,
        zip_path,
        extract_folder,
        EXPECTED_TABLES,
        delimiter=delimiter
    )
    return book_dataset