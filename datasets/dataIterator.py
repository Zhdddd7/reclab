import os
import zipfile
import csv
from typing import Iterator, List, Optional
from torch.utils.data import IterableDataset
from reclab._download_hooks import DownloadManager

class LazyCSVIterableDataset(IterableDataset):
    def __init__(
        self, 
        url: str, 
        zip_path: str, 
        extract_folder: str, 
        expected_csv_files: List[str], 
        chunk_size: int = 10000, 
        delimiter: str = ","
    ):
        """
        Args:
            url (str): the url for the dataset
            zip_path (str): zip folder
            extract_folder (str): data folder for extract files
            expected_csv_files (List[str]): table name list
            chunk_size (int): chunk read size
            delimiter (str): csv delimiter, often refers to ','
        """
        super().__init__()
        self.url = url
        self.zip_path = zip_path
        self.extract_folder = extract_folder
        self.expected_csv_files = expected_csv_files
        self.chunk_size = chunk_size
        self.delimiter = delimiter

        self._data_cache: Optional[List[List[str]]] = None
        self._extracted = False
        self._loaded_to_memory = False

    def _download_if_needed(self):
        """如有必要下载ZIP文件。"""
        if not os.path.exists(self.zip_path):
            os.makedirs(os.path.dirname(self.zip_path), exist_ok=True)
            dm = DownloadManager()
            dm.get_local_path(self.url, self.zip_path)

    def _extract_if_needed(self):
        """如果ZIP还未解压则进行解压，并校验所需的CSV文件是否存在。"""
        if not os.path.exists(self.extract_folder):
            os.makedirs(self.extract_folder, exist_ok=True)
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                zf.extractall(self.extract_folder)

        # 检查期望的文件是否存在
        actual_files = os.listdir(self.extract_folder)
        missing = [f for f in self.expected_csv_files if f not in actual_files]
        if missing:
            raise ValueError(f"Some expected CSV files are missing: {missing}")
        
        self._extracted = True

    def _load_data_into_memory(self):
        """把所有CSV文件内容一次性读入内存，以方便后续迭代快速访问。"""
        # 首次需要下载与解压
        self._download_if_needed()
        self._extract_if_needed()

        all_data = []
        for csv_file in self.expected_csv_files:
            csv_path = os.path.join(self.extract_folder, csv_file)
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=self.delimiter)
                for row in reader:
                    all_data.append(row)

        self._data_cache = all_data
        self._loaded_to_memory = True

    def __iter__(self) -> Iterator[List[str]]:
        # 如果已经缓存了全部数据，直接从内存中迭代
        if self._loaded_to_memory and self._data_cache is not None:
            for i in range(0, len(self._data_cache), self.chunk_size):
                chunk = self._data_cache[i : i + self.chunk_size]
                for row in chunk:
                    yield row
            return

        # 如果未缓存，则执行懒加载逻辑
        self._download_if_needed()
        self._extract_if_needed()

        # 分块读取文件并yield
        data = []
        for csv_file in self.expected_csv_files:
            csv_path = os.path.join(self.extract_folder, csv_file)
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=self.delimiter)
                current_chunk = []
                count = 0
                for row in reader:
                    current_chunk.append(row)
                    count += 1
                    if count % self.chunk_size == 0:
                        # 输出当前块
                        for r in current_chunk:
                            yield r
                        data.extend(current_chunk)
                        current_chunk = []
                # 处理最后不满chunk的部分
                if current_chunk:
                    for r in current_chunk:
                        yield r
                    data.extend(current_chunk)

        # 完整迭代后，把数据缓存进内存中，便于下次快速访问
        self._data_cache = data
        self._loaded_to_memory = True