import os
import zipfile
import csv
from typing import List, Optional, Dict, Any, Iterator
from torch.utils.data import IterableDataset
from reclab._download_hooks import DownloadManager

class TableIterableDataset(IterableDataset):
            def __init__(self, data, chunk_size):
                self.data = data
                self.chunk_size = chunk_size
            def __iter__(self):
                if self.chunk_size is None:
                    for row in self.data:
                        yield row
                else:
                    for i in range(0, len(self.data), self.chunk_size):
                        chunk = self.data[i:i+self.chunk_size]
                        for row in chunk:
                            yield row

class MultiTableDataset:
    def __init__(
        self, 
        url: str, 
        zip_path: str, 
        extract_folder: str, 
        expected_csv_files: List[str], 
        delimiter: str = ","
    ):
        """
        A multi-table dataset class that lazily loads multiple CSV files from a ZIP archive.
        
        Args:
            url (str): URL of the dataset ZIP file
            zip_path (str): Local path to save the downloaded ZIP file
            extract_folder (str): Directory where the ZIP will be extracted
            expected_csv_files (List[str]): List of CSV files expected to be found in the ZIP
            delimiter (str): CSV delimiter, often ','
        """
        self.url = url
        self.zip_path = zip_path
        self.extract_folder = extract_folder
        self.expected_csv_files = expected_csv_files
        self.delimiter = delimiter

        self._data_loaded = False
        self._tables_data: Dict[str, List[List[str]]] = {}   # {table_name: [ [row], [row], ... ]}
        self._tables_header: Dict[str, Optional[List[str]]] = {}  # {table_name: [col_names] or None}

    def _download_if_needed(self):
        # Download the ZIP file if it does not exist locally
        if not os.path.exists(self.zip_path):
            os.makedirs(os.path.dirname(self.zip_path), exist_ok=True)
            dm = DownloadManager()
            dm.get_local_path(self.url, self.zip_path)

    def _extract_if_needed(self):
        # Extract the ZIP file if not already extracted
        if not os.path.exists(self.extract_folder):
            os.makedirs(self.extract_folder, exist_ok=True)
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                zf.extractall(self.extract_folder)

        # Check if all expected CSV files are present
        actual_files = os.listdir(self.extract_folder)
        missing = [f for f in self.expected_csv_files if f not in actual_files]
        if missing:
            raise ValueError(f"Some expected CSV files are missing: {missing}")

    def _load_data_into_memory(self):
        # Load all CSV files into memory if not done yet
        if self._data_loaded:
            return
        self._download_if_needed()
        self._extract_if_needed()

        for csv_file in self.expected_csv_files:
            csv_path = os.path.join(self.extract_folder, csv_file)
            table_data = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter=self.delimiter)
                header = next(reader, None)  # Assuming the first row is the header
                self._tables_header[csv_file] = header
                for row in reader:
                    table_data.append(row)
            self._tables_data[csv_file] = table_data

        self._data_loaded = True

    def list_tables(self) -> List[str]:
        """List all table names in this dataset."""
        return self.expected_csv_files

    def get_table_data(self, table_name: str) -> List[List[str]]:
        """Return all rows of a specific table as a list of lists."""
        if table_name not in self.expected_csv_files:
            raise ValueError(f"Table {table_name} not in dataset.")
        self._load_data_into_memory()
        return self._tables_data[table_name]

    def get_table_header(self, table_name: str) -> Optional[List[str]]:
        """Return the header (column names) of a table if available."""
        if table_name not in self.expected_csv_files:
            raise ValueError(f"Table {table_name} not in dataset.")
        self._load_data_into_memory()
        return self._tables_header[table_name]

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Return information about a table, such as number of rows, number of columns, and header."""
        data = self.get_table_data(table_name)
        header = self.get_table_header(table_name)
        num_rows = len(data)
        num_cols = len(data[0]) if num_rows > 0 else 0
        return {
            "num_rows": num_rows,
            "num_cols": num_cols,
            "header": header
        }

    def get_table_iter(self, table_name: str, chunk_size: Optional[int] = None,
                       start: Optional[int] = None, end: Optional[int] = None) -> Iterator[List[str]]:
        """
        Return an iterator over the specified table's rows, supporting slicing and optional chunking.
        
        Args:
            table_name (str): The table to iterate over
            chunk_size (int, optional): If provided, rows are yielded in chunks of this size
            start (int, optional): Start index (inclusive)
            end (int, optional): End index (exclusive)
        """
        data = self.get_table_data(table_name)
        sliced_data = data[start:end]

        if chunk_size is None:
            for row in sliced_data:
                yield row
        else:
            for i in range(0, len(sliced_data), chunk_size):
                chunk = sliced_data[i:i+chunk_size]
                for row in chunk:
                    yield row

    def get_table_dataset(self, table_name: str, 
                          chunk_size: Optional[int] = None,
                          start: Optional[int] = None, 
                          end: Optional[int] = None) -> IterableDataset:
        """
        Return a PyTorch IterableDataset object for the specified table, supporting slicing and chunking.

        Args:
            table_name (str): The table name
            chunk_size (int, optional): If provided, rows are yielded in chunks
            start (int, optional): Start index (inclusive)
            end (int, optional): End index (exclusive)
        """
        data = self.get_table_data(table_name)
        sliced_data = data[start:end]
        return TableIterableDataset(sliced_data, chunk_size)