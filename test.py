from reclab.datasets import BLOG_REC, MOVIE, BOOK
from torch.utils.data import DataLoader
from reclab.data.test_autoFE import FeatureStreaming
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from reclab.data.gradientSelector import FeatureGradientSelector


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

def test_featureSelector():
    # 加载癌证数据集
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # 打印原始特征列名
    print("The Original feature names are:")
    print(list(X.columns))

    # 初始化特征选择器并选择 Top 10 特征
    selector = FeatureGradientSelector(n_features=10)
    selector.fit(X, y)
    selected_features = selector.get_features(indices=True)

    # 打印选择的 Top 10 特征列名
    print("\nThe Selected feature names are:")
    print(list(X.columns[selected_features]))

    # 数据集分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # 使用原始特征训练 SVM
    svm_all = SVC(random_state=42)
    svm_all.fit(X_train, y_train)
    y_pred_all = svm_all.predict(X_test)
    accuracy_all = accuracy_score(y_test, y_pred_all)
    print("\nThe acc trained by original features on SVM:", accuracy_all)

    # 使用选择的特征训练 SVM
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]

    svm_selected = SVC(random_state=42)
    svm_selected.fit(X_train_selected, y_train)
    y_pred_selected = svm_selected.predict(X_test_selected)
    accuracy_selected = accuracy_score(y_test, y_pred_selected)
    print("The acc trained by selected features on SVM:", accuracy_selected)

test_featureSelector()