"""
Utility functions for the ID Photo Validator, including the model downloader.
"""

import os
import requests
import time
from typing import Optional

from .config import MODEL_DIR

def download_file(url: str, dest_path: str, desc: Optional[str] = None) -> None:
    """
    Downloads a file from a URL to a destination path with progress indication.

    Args:
        url (str): The URL of the file to download.
        dest_path (str): The local path to save the file.
        desc (str, optional): A description of the file being downloaded. Defaults to None.
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if desc:
        print(f"Downloading {desc} from {url}...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes

        total_size = int(response.headers.get('content-length', 0))
        start_time = time.time()

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if total_size > 0:
            speed = (total_size / 1024) / duration
            print(f"Successfully downloaded {os.path.basename(dest_path)} ({total_size / 1024:.2f} KB) in {duration:.2f}s ({speed:.2f} KB/s).")
        else:
            print(f"Successfully downloaded {os.path.basename(dest_path)} in {duration:.2f}s.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        # Clean up partially downloaded file
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise
