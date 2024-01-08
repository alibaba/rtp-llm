import os
import traceback
import logging

__all__ = ['fetch_remote_file_to_local']

try:
    from py_inference.sdk.files.remote_file_tools import fetch_remote_file_to_local
except Exception as e:
    logging.info(f"failed to import py_inference, error: [{str(e)}], use fake instead")

    def fetch_remote_file_to_local(path: str):
        if not os.path.exists(path):
            raise Exception("failed to read path from local")
        return path