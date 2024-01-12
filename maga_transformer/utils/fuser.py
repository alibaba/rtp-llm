import os
import time
import json
import logging
import requests
import hashlib
import atexit
import functools

from urllib.parse import urlparse
from typing import Optional

class RetryableError(Exception):
    pass

def retry_with_timeout(timeout_seconds: int = 300, retry_interval: float = 1.0, exceptions: tuple = (requests.exceptions.RequestException, RetryableError)):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= timeout_seconds:
                        raise TimeoutError(f"Function {func.__name__} timed out after {timeout_seconds} seconds") from e
                    logging.info(f"Retrying {func.__name__} after catching exception: {e}")
                    time.sleep(retry_interval)
        return wrapper
    return decorator

# Fuser is a wrapper class for c2 sidecar fuse.
# see documents at https://aliyuque.antfin.com/owt27z/ohohhg/xyardt2bwbyfmhn5
class Fuser:
    def __init__(self) -> None:
        self._fuse_uri = "http://0:28006"
        self._fuse_path_prefix = "/mnt/fuse"
        self._mount_src_map = {}
        atexit.register(self.umount_all)

    @retry_with_timeout()
    def mount_dir(self, path: str) -> Optional[str]:
        mnt_path = os.path.join(self._fuse_path_prefix, hashlib.md5(path.encode("utf-8")).hexdigest())
        req_json = {
            "uri": path,
            "mountDir":mnt_path
        }
        logging.info(f"mount request to {self._fuse_uri}/FuseService/mount: {req_json}")
        mount_result = requests.post(f"{self._fuse_uri}/FuseService/mount", json=req_json, timeout=600).json()
        error_code = mount_result['errorCode']
        if error_code != 0:
            raise RetryableError(f"mount {path} -> {mnt_path} failed: {mount_result}")
        logging.info(f"mount dir success: {path} -> {mnt_path}")
        self._mount_src_map[mnt_path] = path
        return mnt_path

    def umount_fuse_dir(self, mnt_path: str) -> bool:
        if mnt_path not in self._mount_src_map:
            raise Exception(f'{mnt_path} not mounted')
        req_json = {
            "mountDir": mnt_path
        }
        umount_result = requests.post(f"{self._fuse_uri}/FuseService/umount", json=req_json, timeout=600).json()
        error_code = umount_result['errorCode']
        if error_code != 0:
            raise Exception(f"umount {mnt_path} failed: {umount_result}")
        logging.info(f"umount dir success: {mnt_path}")
        mount_src = self._mount_src_map[mnt_path]
        del self._mount_src_map[mnt_path]

    def umount_all(self) -> None:
        for mnt_path in self._mount_src_map.copy().keys():
            self.umount_fuse_dir(mnt_path)

_fuser = Fuser()

def fetch_remote_file_to_local(path: str):
    parse_result = urlparse(path)
    if parse_result.scheme == '':
        return path
    else:
        return _fuser.mount_dir(path)
