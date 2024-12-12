import requests
from tqdm import tqdm

def _stream_response(r, chunk_size=16 * 1024):
    total_size = int(r.headers.get("Content-length", 0))
    
    if total_size is None:
        total_size = 0
    else:
        total_size = int(total_size)

    ## here is an update: unit_scale=True
    with tqdm(total=total_size, unit="B", unit_scale=True) as t:
        for chunk in r.iter_content(chunk_size):
            if chunk:
                t.update(len(chunk))
                yield chunk


class DownloadManager:
    def get_local_path(self, url, destination):
        if not url: 
            raise ValueError("URL must be provided.")
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        # More data source can be added here
        with open(destination, "wb") as f:
            for chunk in _stream_response(response):
                f.write(chunk)
