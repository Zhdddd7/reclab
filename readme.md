# RecLab: A place for recommendation algorithms test
RecLab is a toolkit designed to streamline the development of recommendation system demos and experiments. It aims to provide a user-friendly environment for quickly building, evaluating, and experimenting with various recommendation algorithms, datasets, and feature engineering workflows.
## Key Features
* Easy Recommendation Demo Setup
* Built-in Datasets with Lazy and Chunked Loading
* Feature Engineering Tools
* Model Building and Classic Algorithms
## Getting Started
### Installation
```
pip install reclab
```
### Loading a Dataset
```
from reclab.datasets import BLOG_REC  # Example dataset loader

# Initialize the dataset; no data is loaded yet
test_ds = BLOG_REC()
# Show the table list in this dataset
test_ds.list_tables()
```
### Data Full Read
Data full read will load all the table to storage. This works good for small tables.
```
full_data_author = test_ds.get_table_data('Author Data.csv')
```
### Data Streaming Read
Data streaming will read data from a iterable file stream object, which only load the data when the data is used. This load strategy works good for big tables.
Besides, the dataset iter object is adapt to DataLoader in torch.utils.data, so you can definitely use this module just as a dataset reader and convert it to DataLoader without changing your original code.
```
from torch.utils.data import DataLoader
author_loader = test_ds.iter_loader('Author Data.csv')
author_loader_slice = test_ds.iter_loader('Author Data.csv', start = 1, end = 3)
author_loader_chunk = test_ds.iter_loader('Author Data.csv', chunk_size = 3)
loader = DataLoader(author_loader_slice, batch_size=None)
```
## Train and Evaluatin
## Version

