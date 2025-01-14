import os
import hashlib
import logging
import requests
import atexit
import time
import functools
from typing import Optional, Dict, List
from urllib.parse import urlparse
import threading
from enum import Enum
from maga_transformer.aios.kmonitor.python_client.kmonitor.utils.hippo_helper import HippoHelper
from subprocess import check_call


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

class MountRwMode(Enum):
    RWMODE_RO = 0  # 只读模式, 默认只读
    RWMODE_WO = 1  # 只写模式
    RWMODE_RW = 2  # 读写模式

# Fuser is a wrapper class for c2 sidecar fuse.
# see documents at https://aliyuque.antfin.com/owt27z/ohohhg/xyardt2bwbyfmhn5
class Fuser:
    def __init__(self) -> None:
        if HippoHelper.host_fuse_port():
            self._fuse_uri = f"http://{HippoHelper.host_ip}:{HippoHelper.host_fuse_port()}"
            self._fuse_path_prefix = f"{HippoHelper.app_workdir}/fuse"
        else:
            self._fuse_uri = "http://0:28006"
            self._fuse_path_prefix = "/mnt/fuse"
        self._mount_src_map = {}  # Maps mount path to (original path, ref count)
        self.lock = threading.RLock()  # 使用重入锁
        atexit.register(self.umount_all)

    @retry_with_timeout()
    def mount_dir(self, path: str, mount_mode:MountRwMode = MountRwMode.RWMODE_RO) -> Optional[str]:
        mnt_path = os.path.join(self._fuse_path_prefix, hashlib.md5(path.encode("utf-8")).hexdigest())
        req_json = {
            "uri": path,
            "mountDir":mnt_path,
            "rwMode":mount_mode.name
        }
        if mount_mode in [MountRwMode.RWMODE_WO, MountRwMode.RWMODE_RW]:
            req_json.update({"cacheOptions":{"writeMode":"WRITE_THROGH","enableRemove":True}})

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

class MountInfo:
    def __init__(self):
        self.mounted_user_addresses = set()

class MountedPathInfo:
    def __init__(self, mount_root: str):
        self.mount_root = mount_root
        self.ref_count = 0

class NfsManager():
    def __init__(self):
        self._mounted_path_map: Dict[str, MountedPathInfo] = {}
        self._nfs_info_map: Dict[str, MountInfo] = {} # nfs address -> MountInfo
        self._lock = threading.RLock()

    def _do_mount_nfs(self, nfs_address: str, mount_root: str):
        check_call(f"sudo mkdir -p {mount_root}", shell=True)
        check_call(f"sudo mount -t nfs -o vers=4,minorversion=0,noresvport {nfs_address}:/ {mount_root}", shell=True)
        logging.info(f"successfully mounted nfs path {nfs_address} to {mount_root}")
        self._nfs_info_map[mount_root] = MountInfo()

    def _do_unmount_nfs(self, mount_root: str):
        check_call(f"sudo umount {mount_root}", shell=True)
        check_call(f"sudo rm -rf {mount_root}", shell=True)
        logging.info(f"successfully unmounted nfs path {mount_root}")
        del self._nfs_info_map[mount_root]

    def mount_nfs_dir(self, path: str) -> str:
        parse_result = urlparse(path)
        nfs_address = parse_result.netloc
        address_md5 = hashlib.md5(nfs_address.encode("utf-8")).hexdigest()[0:8]
        pid = os.getpid()
        mount_root = f"/mnt/ft_{pid}_nfs_{address_md5}"
        mounted_dir_path = f"{mount_root}/{parse_result.path}"
        with self._lock:
            if mounted_dir_path in self._mounted_path_map:
                self._mounted_path_map[mounted_dir_path].ref_count += 1
                logging.info(f"nfs path {path} already mounted to {mounted_dir_path}, skip")
                return mounted_dir_path
            if mount_root not in self._nfs_info_map:
                logging.info(f"first time mounting nfs path [{nfs_address}] for [{path}]")
                self._do_mount_nfs(nfs_address, mount_root)
            self._nfs_info_map[mount_root].mounted_user_addresses.add(mounted_dir_path)
            self._mounted_path_map[mounted_dir_path] = MountedPathInfo(mount_root)
            logging.info(f"nfs path {path} mounted to {mounted_dir_path}")
            return mounted_dir_path

    def unmount_nfs_path(self, path: str) -> None:
        with self._lock:
            if path not in self._mounted_path_map:
                return
            self._mounted_path_map[path].ref_count -= 1
            if self._mounted_path_map[path].ref_count > 0:
                logging.info(f"nfs path {path} still in use, skip actual unmount")
                return
            mount_root = self._mounted_path_map[path].mount_root
            self._nfs_info_map[mount_root].mounted_user_addresses.remove(path)
            if len(self._nfs_info_map[mount_root].mounted_user_addresses) == 0:
                self._do_unmount_nfs(mount_root)
                del self._mounted_path_map[path]

    def unmount_all(self) -> None:
        logging.info("unmount all nas paths")
        with self._lock:
            self._mounted_path_map = {}
            for mount_root in list(self._nfs_info_map.keys()):
                self._do_unmount_nfs(mount_root)

_nfs_manager = NfsManager()

def fetch_remote_file_to_local(path: str, mount_mode:MountRwMode = MountRwMode.RWMODE_RO):
    parse_result = urlparse(path)
    if parse_result.scheme == '':
        logging.info(f"Local path {path} use directly.")
        return path
    elif parse_result.scheme == 'nas':
        logging.info(f"try mount nas path {path}")
        return _nfs_manager.mount_nfs_dir(path)
    else:
        logging.info(f"try fuse path {path}")
        return _fuser.mount_dir(path, mount_mode)

def umount_file(path: str, force: bool = False):
    logging.info(f"umount file {path}")
    _fuser.umount_fuse_dir(path, force=force)
    _nfs_manager.unmount_nfs_path(path)
