import os
import zipfile
import csv
from typing import List, Optional, Dict, Any, Iterator
from torch.utils.data import IterableDataset
from reclab._download_hooks import DownloadManager

class FileIterableDataset(IterableDataset):
            def __init__(self, parent, table_name, delimiter, chunk_size, start, end):
                self.parent = parent
                self.table_name = table_name
                self.delimiter = delimiter
                self.chunk_size = chunk_size
                self.start = start
                self.end = end

            def __iter__(self) -> Iterator[List[str]]:
                # Ensure data is available on disk
                self.parent._download_if_needed()
                self.parent._extract_if_needed()

                csv_path = os.path.join(self.parent.extract_folder, self.table_name)
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter=self.delimiter)
                    header = next(reader, None)  # skip the header line

                    # Skip 'start' lines if specified
                    if self.start is not None:
                        for _ in range(self.start):
                            next(reader, None)

                    current_index = 0
                    buffer = []
                    rows_to_read = None
                    if self.end is not None:
                        offset = self.start if self.start is not None else 0
                        rows_to_read = self.end - offset

                    for row in reader:
                        if rows_to_read is not None and current_index >= rows_to_read:
                            break

                        if self.chunk_size is None:
                            yield row
                        else:
                            buffer.append(row)
                            if len(buffer) == self.chunk_size:
                                for r in buffer:
                                    yield r
                                buffer.clear()

                        current_index += 1

                    # If chunking and leftover in buffer
                    if self.chunk_size is not None and buffer:
                        for r in buffer:
                            yield r

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
            expected_csv_files (List[str]): List of CSV files expected in the ZIP
            delimiter (str): CSV delimiter, often ','
        """
        self.url = url
        self.zip_path = zip_path
        self.extract_folder = extract_folder
        self.expected_csv_files = expected_csv_files
        self.delimiter = delimiter

        self._data_loaded = False
        self._tables_data: Dict[str, List[List[str]]] = {}
        self._tables_header: Dict[str, Optional[List[str]]] = {}

    def _download_if_needed(self):
        if not os.path.exists(self.zip_path):
            os.makedirs(os.path.dirname(self.zip_path), exist_ok=True)
            dm = DownloadManager()
            dm.get_local_path(self.url, self.zip_path)

    def _extract_if_needed(self):
        if not os.path.exists(self.extract_folder):
            os.makedirs(self.extract_folder, exist_ok=True)
            with zipfile.ZipFile(self.zip_path, 'r') as zf:
                zf.extractall(self.extract_folder)

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
        """Return all rows of a specific table as a list of lists (fully loaded in memory)."""
        if table_name not in self.expected_csv_files:
            raise ValueError(f"Table {table_name} not in dataset.")
        self._load_data_into_memory()
        return self._tables_data[table_name]

    def get_table_header(self, table_name: str) -> Optional[List[str]]:
        """Return the header (column names) of a table without fully loading data."""
        if table_name not in self.expected_csv_files:
            raise ValueError(f"Table {table_name} not in dataset.")

        # Ensure the file is available
        self._download_if_needed()
        self._extract_if_needed()

        csv_path = os.path.join(self.extract_folder, table_name)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            header = next(reader, None)  # The first line is the header
            return header

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

    def iter_loader(self, table_name: str, 
                          chunk_size: Optional[int] = None,
                          start: Optional[int] = None, 
                          end: Optional[int] = None) -> IterableDataset:
        """
        Return a PyTorch IterableDataset object for the specified table, streaming from the file.

        Args:
            table_name (str): The table name
            chunk_size (int, optional): If provided, rows are yielded in chunks
            start (int, optional): Start index (inclusive)
            end (int, optional): End index (exclusive)
        """

        if table_name not in self.expected_csv_files:
            raise ValueError(f"Table {table_name} not in dataset.")


        return FileIterableDataset(self, table_name, self.delimiter, chunk_size, start, end)
