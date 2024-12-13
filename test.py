from reclab.datasets.blog_rec import BLOG_REC

# test_ds = BLOG_REC()

# for i, ele in enumerate(test_ds):
#     if i < 5:
#         print(ele)
#     else:
#         break

from torch.utils.data import DataLoader
test_ds = BLOG_REC()
test_ds.list_tables()

author_header = test_ds.get_table_header('Author Data.csv')
# author_info = test_ds.get_table_info('Author Data.csv')
author_loader = test_ds.iter_loader('Author Data.csv')
author_loader_slice = test_ds.iter_loader('Author Data.csv', start = 1, end = 3)
author_loader_chunk = test_ds.iter_loader('Author Data.csv', chunk_size = 3)
loader = DataLoader(author_loader_slice, batch_size=None)
print(f'the tables are {test_ds.list_tables()}')
print(list(loader))
print(f"the tables info are {test_ds.get_table_info('Author Data.csv')}")