from reclab.datasets import BLOG_REC, MOVIE, BOOK
from torch.utils.data import DataLoader
from reclab.data.test_autoFE import FeatureStreaming

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
    print(list(loader))
    print(f"the tables info are {test_ds.get_table_info('movies.csv')}")

def book_test():
    test_ds = BOOK()
    test_ds.list_tables()
    movie_loader = test_ds.iter_loader('Books.csv', start = 1, end = 20)
    print(test_ds.get_table_header('Books.csv'))
    loader = DataLoader(movie_loader, batch_size=None)
    for data in movie_loader:
        print(data)
    print(f"the tables info are {test_ds.get_table_info('Books.csv')}")

def test_featureStreaming():
    bk = BOOK()
    loader = DataLoader(bk.iter_loader('Books.csv', start = 1, end = 201))
    header = bk.get_table_header('Books.csv')
    fts = FeatureStreaming(loader, 100, header)
    while True:
        batch_df = fts.process()
        if batch_df.empty:
            break
        print(batch_df)

test_featureStreaming()