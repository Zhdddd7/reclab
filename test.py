from reclab.datasets.blog_rec import BLOG_REC

test_ds = BLOG_REC()

for i, ele in enumerate(test_ds):
    if i < 5:
        print(ele)
    else:
        break
