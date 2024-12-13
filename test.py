from reclab.datasets import BLOG_REC, MOVIE
from torch.utils.data import DataLoader

# dataset unit test
def blog_rec_test():
    test_ds = BLOG_REC()
    test_ds.list_tables()
    author_header = test_ds.get_table_header('Author Data.csv')
    author_loader = test_ds.iter_loader('Author Data.csv')
    author_loader_slice = test_ds.iter_loader('Author Data.csv', start = 1, end = 3)
    author_loader_chunk = test_ds.iter_loader('Author Data.csv', chunk_size = 3)
    loader = DataLoader(author_loader_slice, batch_size=None)
    print(f'the tables are {test_ds.list_tables()}')
    print(list(loader))
    print(f"the tables info are {test_ds.get_table_info('Author Data.csv')}")

def movie_test():
    test_ds = MOVIE()
    test_ds.list_tables()
    movie_loader = test_ds.iter_loader('movies.csv', start = 1, end = 5)
    loader = DataLoader(movie_loader, batch_size=None)
    print(list(movie_loader))
    print(f"the tables info are {test_ds.get_table_info('movies.csv')}")

movie_test()