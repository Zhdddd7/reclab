import pandas as pd
from torch.utils.data import DataLoader

class FeatureStreaming:
    def __init__(self, dataloader: DataLoader, batch_size: int, columns: list):
        """
        :param dataloader: DataLoader，负责数据加载
        :param batch_size: 每批次加载的数据条目数
        :param columns: 表的列名，用于返回的 DataFrame
        """
        self.dataloader = iter(dataloader)  # 转为迭代器
        self.batch_size = batch_size
        self.columns = columns

    def process(self):
        """
        从 DataLoader 中加载一个批次的数据，进行特征工程，返回 DataFrame。
        """
        try:
            # 从 DataLoader 加载下一个批次的数据
            batch_data = [next(self.dataloader) for _ in range(self.batch_size)]
        except StopIteration:
            # 如果到达 DataLoader 尾部，返回空
            return pd.DataFrame(columns=self.columns)

        # 假设 batch_data 是一个张量或嵌套结构，需要转为 DataFrame
        # 如果 batch_data 是张量，示例如下：
        batch_df = pd.DataFrame(batch_data, columns=self.columns)

        # 执行特征工程（示例：添加新列）
        # batch_df["new_feature"] = batch_df["feature1"] * 2  # 示例特征工程

        return batch_df

    def process_all(self):
        """
        持续处理所有批次，每次返回一个 DataFrame。
        """
        while True:
            batch_df = self.process_next_batch()
            if batch_df.empty:
                break
            yield batch_df
