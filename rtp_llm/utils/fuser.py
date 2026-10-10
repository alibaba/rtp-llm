import atexit
import errno
import functools
import hashlib
import logging
import os
import re
import threading
import time
import uuid
from enum import Enum
from subprocess import check_call
from typing import Dict, Optional, Tuple, Type
from urllib.parse import urlparse


class RetryableError(Exception):
    pass


def _requests_module():
    import requests

    return requests


def retry_with_timeout(
    timeout_seconds: int = 300,
    retry_interval: float = 1.0,
    exceptions: Optional[Tuple[Type[BaseException], ...]] = None,
):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            retry_exceptions = exceptions
            if retry_exceptions is None:
                requests = _requests_module()
                retry_exceptions = (
                    requests.exceptions.RequestException,
                    RetryableError,
                )
            start_time = time.time()
            while True:
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= timeout_seconds:
                        raise TimeoutError(
                            f"Function {func.__name__} timed out after {timeout_seconds} seconds"
                        ) from e
                    logging.info(
                        f"Retrying {func.__name__} after catching exception: {e}"
                    )
                    time.sleep(retry_interval)

        return wrapper

    return decorator


class MountRwMode(Enum):
    RWMODE_RO = 0  # 只读模式, 默认只读
    RWMODE_WO = 1  # 只写模式
    RWMODE_RW = 2  # 读写模式


def _fuse_mount_options(path: str) -> Optional[set[str]]:
    """Find the FUSE mount covering path in this process's mount namespace."""
    path = os.path.realpath(path)
    matches = []
    with open("/proc/self/mountinfo") as mounts:
        for line in mounts:
            mount, separator, filesystem = line.partition(" - ")
            fields, fs_fields = mount.split(), filesystem.split()
            if not separator or len(fields) < 6 or len(fs_fields) < 3:
                continue
            mount_path = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
            if path == mount_path or path.startswith(mount_path.rstrip("/") + "/"):
                matches.append(
                    (
                        len(mount_path),
                        fields[0],
                        fields[1],
                        fs_fields[0],
                        set(fields[5].split(",")) | set(fs_fields[2].split(",")),
                    )
                )
    if not matches:
        return None
    # Follow the visible mount tree, entering the closest child mount first.
    # A parent-directory overmount can hide a deeper mount that remains in
    # mountinfo, so choosing the longest path alone would reuse a hidden mount.
    # Parents outside a chroot may be absent from mountinfo; treat those mounts
    # as roots of the tree visible to this process.
    mount_ids = {entry[1] for entry in matches}
    roots = [
        entry for entry in matches if entry[2] not in mount_ids or entry[1] == entry[2]
    ]
    if not roots:
        return None
    visible = min(roots, key=lambda entry: entry[0])
    while True:
        children = [
            entry
            for entry in matches
            if entry[2] == visible[1] and entry[1] != visible[1]
        ]
        if not children:
            break
        visible = min(children, key=lambda entry: entry[0])
    _, _, _, fs_type, options = visible
    return (
        options
        if fs_type in ("fuse", "fuseblk") or fs_type.startswith("fuse.")
        else None
    )


@retry_with_timeout(
    timeout_seconds=10,
    retry_interval=0.1,
    exceptions=(FileNotFoundError, RetryableError),
)
def _read_external_mount_probe(path: str, expected: bytes) -> None:
    # Some FUSE implementations expose a file only after its upload completes.
    with open(path, "rb") as probe:
        if probe.read(len(expected) + 1) != expected:
            raise RetryableError(f"external fuse probe is not readable yet: {path}")


@retry_with_timeout(
    timeout_seconds=10,
    retry_interval=0.1,
    exceptions=(FileNotFoundError,),
)
def _rename_external_mount_probe(path: str, renamed: str) -> None:
    # Write-only mounts cannot use a read probe to wait for upload visibility.
    os.rename(path, renamed)


def _check_external_mount_access(
    path: str, mode: MountRwMode, options: set[str]
) -> None:
    if mode != MountRwMode.RWMODE_RO and "ro" in options:
        raise PermissionError(errno.EROFS, "external fuse mount is read-only", path)
    if not os.path.isdir(path):
        raise NotADirectoryError(path)
    if mode != MountRwMode.RWMODE_WO:
        with os.scandir(path) as entries:
            for entry in entries:
                if mode == MountRwMode.RWMODE_RW:
                    break
                if entry.is_file(follow_symlinks=False):
                    with open(entry.path, "rb") as sample:
                        sample.read(1)
                    break
    if mode == MountRwMode.RWMODE_RO:
        return

    # Probe only the borrowed mount, without touching existing files or chmod.
    probe_path = os.path.join(path, f".rtp_llm_fuse_probe_{uuid.uuid4().hex}")
    payload, created = b"rtp-llm fuse access probe", False
    try:
        with open(probe_path, "xb") as probe:
            created = True
            probe.write(payload)
        if mode == MountRwMode.RWMODE_RW:
            _read_external_mount_probe(probe_path, payload)
        renamed = probe_path + ".ready"
        _rename_external_mount_probe(probe_path, renamed)
        probe_path = renamed
    finally:
        if created:
            try:
                os.unlink(probe_path)
            except FileNotFoundError:
                pass


# Fuser is a wrapper class for c2 sidecar fuse.
# see documents at https://aliyuque.antfin.com/owt27z/ohohhg/xyardt2bwbyfmhn5
class Fuser:
    def __init__(self) -> None:
        from rtp_llm.aios.kmonitor.python_client.kmonitor.utils.hippo_helper import (
            HippoHelper,
        )

        if HippoHelper.host_fuse_port():
            self._fuse_uri = (
                f"http://{HippoHelper.host_ip}:{HippoHelper.host_fuse_port()}"
            )
            self._fuse_path_prefix = f"{HippoHelper.app_workdir}/fuse"
        else:
            self._fuse_uri = "http://0:28006"
            self._fuse_path_prefix = "/mnt/fuse"
        self._mount_src_map = {}  # Maps mount path to (original path, ref count)
        self.lock = threading.RLock()  # 使用重入锁
        atexit.register(self.umount_all)
        # Reusing an externally mounted directory must not require the sidecar.
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        with self.lock:
            if self._available is None:
                self._available = self._check_valid()
            return self._available

    def _check_valid(self) -> bool:
        requests = _requests_module()
        try:
            response = requests.post(
                f"{self._fuse_uri}/FuseService/mount",
                json={},  # 空请求体
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            # 验证响应状态码和错误码
            if response.status_code == 200:
                logging.info(
                    f"check fuse is valid:{self._fuse_uri}, response: {response}, {response.status_code}"
                )
                return True
            else:
                logging.warning(
                    f"fuse is not valid:{self._fuse_uri}, {response.status_code} != 200, response: {response}"
                )
                return False
        except requests.ConnectionError:
            logging.warning(f"fuse is not valid: connet {self._fuse_uri} error")
            return False
        except requests.Timeout:
            logging.warning(f"fuse is not valid: connet {self._fuse_uri} timeout")
            return False
        except Exception:
            logging.warning(f"fuse is not valid: connet {self._fuse_uri}  unknown err")
            return False

    def mount_or_reuse_dir(
        self,
        path: str,
        mount_mode: MountRwMode = MountRwMode.RWMODE_RO,
        enable_mnt_ref: bool = False,
    ) -> Optional[str]:
        mnt_path = os.path.join(
            self._fuse_path_prefix, hashlib.md5(path.encode("utf-8")).hexdigest()
        )
        with self.lock:
            if mnt_path not in self._mount_src_map:
                options = _fuse_mount_options(mnt_path)
                if options is not None:
                    # An unusable external mount must not trigger another mount
                    # on top of it. The caller handles the access error.
                    _check_external_mount_access(mnt_path, mount_mode, options)
                    logging.info(
                        "reuse existing fuse directory, skip mount: %s -> %s",
                        path,
                        mnt_path,
                    )
                    # Do not register it: the external owner handles unmounting.
                    return mnt_path

            # Keep reuse detection outside the original mount/retry operation.
            # Serialize callers so an in-progress mount cannot be borrowed.
            return self.mount_dir(path, mount_mode, enable_mnt_ref)

    @retry_with_timeout()
    def mount_dir(
        self,
        path: str,
        mount_mode: MountRwMode = MountRwMode.RWMODE_RO,
        enable_mnt_ref: bool = False,
    ) -> Optional[str]:
        mnt_path = os.path.join(
            self._fuse_path_prefix, hashlib.md5(path.encode("utf-8")).hexdigest()
        )
        req_json = {
            "uri": path,
            "mountDir": mnt_path,
            "rwMode": mount_mode.name,
            "enableMntRef": enable_mnt_ref,
        }
        if mount_mode in [MountRwMode.RWMODE_WO, MountRwMode.RWMODE_RW]:
            req_json.update(
                {"cacheOptions": {"writeMode": "WRITE_THROGH", "enableRemove": True}}
            )

        logging.info(f"mount request to {self._fuse_uri}/FuseService/mount: {req_json}")
        mount_result = (
            _requests_module()
            .post(f"{self._fuse_uri}/FuseService/mount", json=req_json, timeout=600)
            .json()
        )
        error_code = mount_result["errorCode"]
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
        req_json = {"mountDir": mnt_path}
        umount_result = (
            _requests_module()
            .post(f"{self._fuse_uri}/FuseService/umount", json=req_json, timeout=600)
            .json()
        )
        error_code = umount_result["errorCode"]
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
                logging.info(
                    f"Reference count for {mnt_path} is still {count}, skipping umount."
                )
                return True

            # Perform umount if reference count is zero
            self._perform_umount(mnt_path)
            del self._mount_src_map[mnt_path]  # Remove the entry once unmounted

    def umount_all(self, force: bool = True) -> None:
        # Only allow unmounting when there's no other operation ongoing
        with self.lock:
            for mnt_path in list(self._mount_src_map.keys()):
                self.umount_fuse_dir(mnt_path, force=force)


_fuser: Optional[Fuser] = None
_fuser_lock = threading.Lock()


def _get_fuser() -> Fuser:
    global _fuser
    if _fuser is None:
        with _fuser_lock:
            if _fuser is None:
                _fuser = Fuser()
    return _fuser


class MountInfo:
    def __init__(self):
        self.mounted_user_addresses = set()


class MountedPathInfo:
    def __init__(self, mount_root: str):
        self.mount_root = mount_root
        self.ref_count = 0


class NfsManager:
    def __init__(self):
        self._mounted_path_map: Dict[str, MountedPathInfo] = {}
        self._nfs_info_map: Dict[str, MountInfo] = {}  # nfs address -> MountInfo
        self._lock = threading.RLock()

    def _do_mount_nfs(self, nfs_address: str, mount_root: str):
        check_call(f"sudo mkdir -p {mount_root}", shell=True)
        check_call(
            f"sudo mount -t nfs -o vers=4,minorversion=0,noresvport {nfs_address}:/ {mount_root}",
            shell=True,
        )
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
                logging.info(
                    f"nfs path {path} already mounted to {mounted_dir_path}, skip"
                )
                return mounted_dir_path
            if mount_root not in self._nfs_info_map:
                logging.info(
                    f"first time mounting nfs path [{nfs_address}] for [{path}]"
                )
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
                try:
                    self._do_unmount_nfs(mount_root)
                except Exception:
                    logging.exception(f"failed to unmount nfs {mount_root}")


_nfs_manager = NfsManager()


def fetch_remote_file_to_local(
    path: str,
    mount_mode: MountRwMode = MountRwMode.RWMODE_RO,
    enable_mnt_ref: bool = False,
):
    parse_result = urlparse(path)
    if parse_result.scheme == "":
        logging.info(f"Local path {path} use directly.")
        return path
    elif parse_result.scheme == "nas":
        logging.info(f"try mount nas path {path}")
        return _nfs_manager.mount_nfs_dir(path)
    else:
        logging.info(f"try fuse path {path}")
        return _get_fuser().mount_or_reuse_dir(path, mount_mode, enable_mnt_ref)


def umount_file(path: str, force: bool = False):
    logging.info(f"umount file {path}")
    try:
        _get_fuser().umount_fuse_dir(path, force=force)
    finally:
        _nfs_manager.unmount_nfs_path(path)


def fuse_available() -> bool:
    return _get_fuser().available
