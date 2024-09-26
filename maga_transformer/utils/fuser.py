import os
import hashlib
import logging
import requests
import atexit
import time
import functools
from typing import Optional
from urllib.parse import urlparse
import threading

class RetryableError(Exception):
    pass

def retry_with_timeout(timeout_seconds: int = 300, retry_interval: float = 1.0, 
                       exceptions: tuple = (requests.exceptions.RequestException, RetryableError)):
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

class Fuser:
    def __init__(self) -> None:
        self._fuse_uri = "http://0:28006"
        self._fuse_path_prefix = "/mnt/fuse"
        self._mount_src_map = {}  # Maps mount path to (original path, ref count)
        self.lock = threading.RLock()  # 使用重入锁
        atexit.register(self.umount_all)

    @retry_with_timeout()
    def mount_dir(self, path: str) -> Optional[str]:
        mnt_path = os.path.join(self._fuse_path_prefix, hashlib.md5(path.encode("utf-8")).hexdigest())
        req_json = {
            "uri": path,
            "mountDir": mnt_path
        }
        logging.info(f"mount request to {self._fuse_uri}/FuseService/mount: {req_json}")
        mount_result = requests.post(f"{self._fuse_uri}/FuseService/mount", json=req_json, timeout=600).json()
        error_code = mount_result['errorCode']
        if error_code != 0:
            raise RetryableError(f"mount {path} -> {mnt_path} failed: {mount_result}")
        logging.info(f"mount dir success: {path} -> {mnt_path}")
        
        with self.lock:
            if mnt_path in self._mount_src_map:
                # Increment reference count if already mounted
                original_path, count = self._mount_src_map[mnt_path]
                self._mount_src_map[mnt_path] = (original_path, count + 1)
            else:
                # Initialize reference count if first mount
                self._mount_src_map[mnt_path] = (path, 1)

        return mnt_path

    def _perform_umount(self, mnt_path: str) -> None:
        req_json = {
            "mountDir": mnt_path
        }
        umount_result = requests.post(f"{self._fuse_uri}/FuseService/umount", json=req_json, timeout=600).json()
        error_code = umount_result['errorCode']
        if error_code != 0:
            raise Exception(f"umount {mnt_path} failed: {umount_result}")
        logging.info(f"umount dir success: {mnt_path}")

    def umount_fuse_dir(self, mnt_path: str, force: bool = False) -> bool:
        with self.lock:  # Ensure exclusive access to the mount source map
            if mnt_path not in self._mount_src_map:
                logging.info(f"{mnt_path} is not mounted.")
                return

            # If force is True, remove the entry regardless of the reference count
            if force:
                logging.info(f"Force unmounting {mnt_path}.")
                self._perform_umount(mnt_path)
                del self._mount_src_map[mnt_path]  # Remove the entry
                return True
            
            # Decrease the reference count if not forcing
            original_path, count = self._mount_src_map[mnt_path]
            count -= 1
            if count > 0:
                # Still references left, do not umount
                self._mount_src_map[mnt_path] = (original_path, count)
                logging.info(f"Reference count for {mnt_path} is still {count}, skipping umount.")
                return True

            # Perform umount if reference count is zero
            self._perform_umount(mnt_path)
            del self._mount_src_map[mnt_path]  # Remove the entry once unmounted

    def umount_all(self, force: bool = True) -> None:
        # Only allow unmounting when there's no other operation ongoing
        with self.lock:
            for mnt_path in list(self._mount_src_map.keys()):
                self.umount_fuse_dir(mnt_path, force=force)

_fuser = Fuser()

def fetch_remote_file_to_local(path: str):
    parse_result = urlparse(path)
    if parse_result.scheme == '':
        return path
    else:
        return _fuser.mount_dir(path)

def umount_file(path: str, force: bool = False):
    _fuser.umount_fuse_dir(path, force=force)