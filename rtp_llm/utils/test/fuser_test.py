import atexit
import builtins
import errno
import hashlib
import logging
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import requests

from rtp_llm.aios.kmonitor.python_client.kmonitor.utils.hippo_helper import HippoHelper
from rtp_llm.utils.fuser import (
    Fuser,
    MountRwMode,
    _fuse_mount_options,
    fetch_remote_file_to_local,
    retry_with_timeout,
    umount_file,
)
from rtp_llm.utils.jit_cache_store import resolve_remote


class TestFuser(unittest.TestCase):
    def setUp(self):
        self.fuser = Fuser()
        self.addCleanup(atexit.unregister, self.fuser.umount_all)
        mount_root = tempfile.TemporaryDirectory()
        self.addCleanup(mount_root.cleanup)
        self.fuser._fuse_path_prefix = mount_root.name
        self.mock_response = MagicMock()
        self.mock_response.json.return_value = {"errorCode": 0}
        self.external_mounts = {}
        mount_options = patch(
            "rtp_llm.utils.fuser._fuse_mount_options",
            side_effect=lambda path: self.external_mounts.get(path),
        )
        self.mount_options = mount_options.start()
        self.addCleanup(mount_options.stop)

    def tearDown(self):
        self.fuser._mount_src_map = {}

    def mount_path(self, uri):
        return (
            Path(self.fuser._fuse_path_prefix) / hashlib.md5(uri.encode()).hexdigest()
        )

    def external_mount(self, uri, options=("rw",)):
        path = self.mount_path(uri)
        path.mkdir()
        self.external_mounts[str(path)] = set(options)
        return path

    @patch("requests.post")
    def test_external_mount_is_readable_without_mount_or_unmount(self, mock_post):
        uri = "oss://bucket/model/"
        mount_path = self.external_mount(uri, options=("ro",))
        (mount_path / "config.json").write_text('{"model_type": "glm"}')

        with patch("rtp_llm.utils.fuser._fuser", self.fuser):
            for _ in range(2):
                local_path = fetch_remote_file_to_local(uri)
                self.assertEqual(local_path, str(mount_path))
                self.assertEqual(
                    (Path(local_path) / "config.json").read_text(),
                    '{"model_type": "glm"}',
                )
            umount_file(local_path)
            umount_file(local_path, force=True)
            self.fuser.umount_all()

        self.assertEqual(self.fuser._mount_src_map, {})
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_cleanup_only_unmounts_managed_directories(self, mock_post):
        external_uri = "oss://bucket/external/"
        external_path = self.external_mount(external_uri)
        mock_post.return_value = self.mock_response

        for force in (False, True):
            with self.subTest(force=force):
                self.assertEqual(
                    self.fuser.mount_or_reuse_dir(external_uri), str(external_path)
                )
                managed_path = self.fuser.mount_or_reuse_dir(
                    f"oss://bucket/managed/{force}"
                )
                Path(managed_path).mkdir()
                mock_post.reset_mock()

                self.fuser.umount_all(force=force)

                mock_post.assert_called_once_with(
                    f"{self.fuser._fuse_uri}/FuseService/umount",
                    json={"mountDir": managed_path},
                    timeout=600,
                )
                self.assertEqual(self.fuser._mount_src_map, {})
                self.assertTrue(external_path.is_dir())

    @patch("requests.post")
    def test_removed_external_mount_falls_back_to_mount(self, mock_post):
        uri = "oss://bucket/model/"
        mount_path = self.external_mount(uri)
        self.assertEqual(self.fuser.mount_or_reuse_dir(uri), str(mount_path))
        mock_post.assert_not_called()

        mount_path.rmdir()
        self.external_mounts.clear()
        mock_post.return_value = self.mock_response
        self.assertEqual(self.fuser.mount_or_reuse_dir(uri), str(mount_path))

        mock_post.assert_called_once()
        request = mock_post.call_args.kwargs["json"]
        self.assertEqual(request["uri"], uri)
        self.assertEqual(request["mountDir"], str(mount_path))
        self.assertEqual(self.fuser._mount_src_map[str(mount_path)], (uri, 1))

    @patch("requests.post")
    def test_existing_directory_is_reused_for_all_mount_options(self, mock_post):
        uri = "oss://bucket/cache/"
        mount_path = self.external_mount(uri)
        for mode in MountRwMode:
            for enable_ref in (False, True):
                with self.subTest(mode=mode, enable_ref=enable_ref):
                    local_path = self.fuser.mount_or_reuse_dir(uri, mode, enable_ref)
                    self.assertEqual(local_path, str(mount_path))
                    self.assertEqual(self.fuser._mount_src_map, {})
                    self.assertEqual(list(mount_path.iterdir()), [])
                    self.fuser.umount_fuse_dir(local_path, force=True)
        self.fuser.umount_all()
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_missing_directory_preserves_explicit_mount_options(self, mock_post):
        uri = "oss://bucket/cache/"
        mock_post.return_value = self.mock_response
        for mode, enable_ref in (
            (MountRwMode.RWMODE_RO, False),
            (MountRwMode.RWMODE_RO, True),
            (MountRwMode.RWMODE_WO, False),
            (MountRwMode.RWMODE_WO, True),
            (MountRwMode.RWMODE_RW, False),
            (MountRwMode.RWMODE_RW, True),
        ):
            with self.subTest(mode=mode, enable_ref=enable_ref):
                self.fuser._mount_src_map.clear()
                mock_post.reset_mock()
                local_path = self.fuser.mount_or_reuse_dir(uri, mode, enable_ref)
                mock_post.assert_called_once()
                request = mock_post.call_args.kwargs["json"]
                self.assertEqual(request["rwMode"], mode.name)
                self.assertEqual(request["enableMntRef"], enable_ref)
                self.assertEqual(self.fuser._mount_src_map[local_path], (uri, 1))

    @patch("requests.post")
    def test_remote_jit_cache_uses_external_mount_for_publish_and_restore(
        self, mock_post
    ):
        uri = "oss://bucket/jit-cache/"
        mount_path = self.external_mount(uri)
        local_root = Path(self.fuser._fuse_path_prefix)
        kernel = local_root / "kernel.cubin"
        kernel.write_bytes(b"compiled kernel")

        with patch("rtp_llm.utils.fuser._fuser", self.fuser):
            store = resolve_remote(uri, "v1", "test-scope")
            self.assertIsNotNone(store)
            try:
                self.assertEqual(store.remote_root, mount_path / "v1/test-scope")
                store.publish_snapshot({"triton/kernel.cubin": kernel})
                restored = store.prepare_restore(local_root / "restore-staging")
                self.assertIsNotNone(restored)
                self.assertEqual(
                    (restored.staging / "triton/kernel.cubin").read_bytes(),
                    kernel.read_bytes(),
                )
            finally:
                store.close()
            self.fuser.umount_all()

        self.assertTrue(mount_path.is_dir())
        self.assertEqual(self.fuser._mount_src_map, {})
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_remote_jit_cache_unusable_external_mount_fails_open(self, mock_post):
        uri = "oss://bucket/jit-cache/"
        mount_path = self.external_mount(uri)
        with patch("rtp_llm.utils.fuser._fuser", self.fuser), patch(
            "rtp_llm.utils.jit_cache_store.os.access", return_value=False
        ), self.assertLogs(level="WARNING") as logs:
            self.assertIsNone(resolve_remote(uri, "v1", "test-scope"))

        self.assertTrue(any("JIT_CACHE_FAIL_OPEN" in line for line in logs.output))
        self.assertTrue(mount_path.is_dir())
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_remote_jit_cache_missing_directory_is_mounted_and_released(
        self, mock_post
    ):
        uri = "oss://bucket/jit-cache/"
        mount_path = self.mount_path(uri)

        def mount_response(url, **kwargs):
            if url.endswith("/FuseService/mount"):
                mount_path.mkdir()
            return self.mock_response

        mock_post.side_effect = mount_response
        with patch("rtp_llm.utils.fuser._fuser", self.fuser):
            store = resolve_remote(uri, "v1", "test-scope")
            self.assertIsNotNone(store)
            try:
                mock_post.assert_called_once()
                request = mock_post.call_args.kwargs["json"]
                self.assertEqual(request["rwMode"], "RWMODE_RW")
                self.assertTrue(request["enableMntRef"])
                self.assertEqual(request["cacheOptions"]["writeMode"], "WRITE_THROGH")
                self.assertEqual(self.fuser._mount_src_map[str(mount_path)], (uri, 1))
            finally:
                store.close()

        self.assertEqual(self.fuser._mount_src_map, {})
        self.assertEqual(mock_post.call_count, 2)
        self.assertTrue(mock_post.call_args.args[0].endswith("/FuseService/umount"))

    @patch("requests.post")
    def test_managed_mount_keeps_reference_count(self, mock_post):
        uri = "oss://bucket/model/"
        mock_post.return_value = self.mock_response
        local_path = self.fuser.mount_or_reuse_dir(uri)
        Path(local_path).mkdir()
        self.external_mounts[local_path] = {"rw"}
        with patch("rtp_llm.utils.fuser._check_external_mount_access") as check:
            self.assertEqual(self.fuser.mount_or_reuse_dir(uri), local_path)
            check.assert_not_called()
        self.assertEqual(self.fuser._mount_src_map[local_path], (uri, 2))
        self.assertEqual(mock_post.call_count, 2)

        self.fuser.umount_fuse_dir(local_path)
        self.assertEqual(self.fuser._mount_src_map[local_path], (uri, 1))
        self.assertEqual(mock_post.call_count, 2)
        self.fuser.umount_fuse_dir(local_path)
        self.assertNotIn(local_path, self.fuser._mount_src_map)
        self.assertTrue(mock_post.call_args.args[0].endswith("/FuseService/umount"))

    @patch("requests.post")
    def test_file_at_mount_path_is_not_reused(self, mock_post):
        uri = "oss://bucket/model/"
        self.mount_path(uri).write_text("not a directory")
        mock_post.return_value = self.mock_response
        self.fuser.mount_or_reuse_dir(uri)
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_plain_directory_is_not_treated_as_a_mount(self, mock_post):
        uri = "oss://bucket/model/"
        mount_path = self.mount_path(uri)
        mount_path.mkdir()
        (mount_path / "stale.json").write_text("stale local data")
        mock_post.return_value = self.mock_response

        with patch("rtp_llm.utils.fuser._check_external_mount_access") as check:
            self.assertEqual(self.fuser.mount_or_reuse_dir(uri), str(mount_path))
            check.assert_not_called()
        mock_post.assert_called_once()
        self.assertEqual(self.fuser._mount_src_map[str(mount_path)], (uri, 1))

    @patch("requests.post")
    def test_readonly_weights_are_checked_without_writes(self, mock_post):
        uri = "oss://bucket/model/"
        mount_path = self.external_mount(uri, options=("ro",))
        weights = mount_path / "weights.bin"
        weights.write_bytes(b"weights")

        with patch("rtp_llm.utils.fuser.open", wraps=builtins.open) as open_file, patch(
            "rtp_llm.utils.fuser.os.rename"
        ) as rename, patch("rtp_llm.utils.fuser.os.unlink") as unlink, patch(
            "rtp_llm.utils.fuser.os.chmod"
        ) as chmod:
            self.assertEqual(self.fuser.mount_or_reuse_dir(uri), str(mount_path))
            open_file.assert_called_once_with(str(weights), "rb")
            rename.assert_not_called()
            unlink.assert_not_called()
            chmod.assert_not_called()

        self.assertEqual(weights.read_bytes(), b"weights")
        self.assertEqual(list(mount_path.iterdir()), [weights])
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_readonly_mount_rejects_write_modes_without_remount(self, mock_post):
        uri = "oss://bucket/cache/"
        mount_path = self.external_mount(uri, options=("ro",))
        for mode in (MountRwMode.RWMODE_RW, MountRwMode.RWMODE_WO):
            with self.subTest(mode=mode), self.assertRaises(PermissionError) as error:
                self.fuser.mount_or_reuse_dir(uri, mode)
            self.assertEqual(error.exception.errno, errno.EROFS)
        self.assertEqual(list(mount_path.iterdir()), [])
        self.assertEqual(self.fuser._mount_src_map, {})
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_unreadable_weights_do_not_trigger_remount(self, mock_post):
        uri = "oss://bucket/model/"
        mount_path = self.external_mount(uri, options=("ro",))
        weights = mount_path / "weights.bin"
        weights.write_bytes(b"weights")
        with patch(
            "rtp_llm.utils.fuser.open",
            side_effect=PermissionError(errno.EACCES, "weights are not readable"),
        ), self.assertRaises(PermissionError):
            self.fuser.mount_or_reuse_dir(uri)
        self.assertEqual(self.fuser._mount_src_map, {})
        self.assertEqual(list(mount_path.iterdir()), [weights])
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_external_mount_access_errors_do_not_remount(self, mock_post):
        uri = "oss://bucket/cache/"
        mount_path = self.external_mount(uri)

        for operation in ("list", "write", "read", "rename", "delete"):

            def open_file(path, mode="r", *args, **kwargs):
                if (operation == "write" and mode == "xb") or (
                    operation == "read" and mode == "rb"
                ):
                    raise PermissionError(errno.EACCES, "access denied", str(path))
                return builtins.open(path, mode, *args, **kwargs)

            real_operation = {
                "list": "scandir",
                "rename": "rename",
                "delete": "unlink",
            }.get(operation)
            with self.subTest(operation=operation), patch(
                "rtp_llm.utils.fuser.open", side_effect=open_file
            ):
                if real_operation:
                    with patch(
                        f"rtp_llm.utils.fuser.os.{real_operation}",
                        side_effect=PermissionError(errno.EACCES, "access denied"),
                    ), self.assertRaises(PermissionError):
                        self.fuser.mount_or_reuse_dir(uri, MountRwMode.RWMODE_RW)
                else:
                    with self.assertRaises(PermissionError):
                        self.fuser.mount_or_reuse_dir(uri, MountRwMode.RWMODE_RW)
            if operation == "delete":
                # The deliberately denied deletion leaves only our probe behind.
                for probe in mount_path.iterdir():
                    self.assertTrue(probe.name.startswith(".rtp_llm_fuse_probe_"))
                    probe.unlink()
            self.assertEqual(list(mount_path.iterdir()), [])
        self.assertEqual(self.fuser._mount_src_map, {})
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_disconnected_external_mount_does_not_remount(self, mock_post):
        uri = "oss://bucket/model/"
        self.external_mount(uri)
        with patch(
            "rtp_llm.utils.fuser.os.scandir",
            side_effect=OSError(errno.ENOTCONN, "Transport endpoint is not connected"),
        ), self.assertRaises(OSError) as error:
            self.fuser.mount_or_reuse_dir(uri)
        self.assertEqual(error.exception.errno, errno.ENOTCONN)
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_mountinfo_error_does_not_trigger_mount(self, mock_post):
        self.mount_options.side_effect = PermissionError("cannot read mountinfo")
        with self.assertRaises(PermissionError):
            self.fuser.mount_or_reuse_dir("oss://bucket/model/")
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_remote_jit_cache_readonly_external_mount_fails_open(self, mock_post):
        uri = "oss://bucket/jit-cache/"
        mount_path = self.external_mount(uri, options=("ro",))
        with patch("rtp_llm.utils.fuser._fuser", self.fuser), self.assertLogs(
            level="WARNING"
        ) as logs:
            self.assertIsNone(resolve_remote(uri, "v1", "test-scope"))
        self.assertTrue(any("JIT_CACHE_FAIL_OPEN" in line for line in logs.output))
        self.assertEqual(list(mount_path.iterdir()), [])
        mock_post.assert_not_called()

    @patch("requests.post")
    @patch("rtp_llm.utils.fuser.time.sleep")
    def test_external_rw_probe_waits_for_file_visibility(self, mock_sleep, mock_post):
        uri = "oss://bucket/jit-cache/"
        mount_path = self.external_mount(uri)
        reads = 0

        def open_file(path, mode="r", *args, **kwargs):
            nonlocal reads
            if mode == "rb":
                reads += 1
                if reads == 1:
                    raise FileNotFoundError("upload still pending")
            return builtins.open(path, mode, *args, **kwargs)

        with patch("rtp_llm.utils.fuser.open", side_effect=open_file):
            self.assertEqual(
                self.fuser.mount_or_reuse_dir(uri, MountRwMode.RWMODE_RW),
                str(mount_path),
            )
        self.assertEqual(reads, 2)
        self.assertEqual(list(mount_path.iterdir()), [])
        mock_sleep.assert_called_once()
        mock_post.assert_not_called()

    @patch("requests.post")
    @patch("rtp_llm.utils.fuser.time.sleep")
    def test_external_wo_probe_waits_for_file_visibility(self, mock_sleep, mock_post):
        uri = "oss://bucket/write-only/"
        mount_path = self.external_mount(uri)
        original_rename, attempts = os.rename, 0

        def rename_file(source, destination):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise FileNotFoundError("upload still pending")
            return original_rename(source, destination)

        with patch("rtp_llm.utils.fuser.os.rename", side_effect=rename_file), patch(
            "rtp_llm.utils.fuser.open", wraps=builtins.open
        ) as open_file, patch("rtp_llm.utils.fuser.os.scandir") as scandir:
            self.assertEqual(
                self.fuser.mount_or_reuse_dir(uri, MountRwMode.RWMODE_WO),
                str(mount_path),
            )
            scandir.assert_not_called()
            open_file.assert_called_once()
            self.assertEqual(open_file.call_args.args[1], "xb")
        self.assertEqual(attempts, 2)
        self.assertEqual(list(mount_path.iterdir()), [])
        self.assertEqual(self.fuser._mount_src_map, {})
        mock_sleep.assert_called_once()
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_external_rw_probe_timeout_cleans_up_without_remount(self, mock_post):
        uri = "oss://bucket/jit-cache/"
        mount_path = self.external_mount(uri)

        def open_file(path, mode="r", *args, **kwargs):
            if mode == "rb":
                raise FileNotFoundError("upload still pending")
            return builtins.open(path, mode, *args, **kwargs)

        with patch("rtp_llm.utils.fuser.open", side_effect=open_file), patch(
            "rtp_llm.utils.fuser.time.time", side_effect=(0, 11)
        ), self.assertRaises(TimeoutError):
            self.fuser.mount_or_reuse_dir(uri, MountRwMode.RWMODE_RW)
        self.assertEqual(list(mount_path.iterdir()), [])
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_active_mount_does_not_run_reuse_checks(self, mock_post):
        uri = "oss://bucket/model/"
        self.external_mount(uri, options=("ro",))
        mock_post.return_value = self.mock_response
        with patch("rtp_llm.utils.fuser._check_external_mount_access") as check:
            local_path = self.fuser.mount_dir(uri, MountRwMode.RWMODE_RW, True)
            check.assert_not_called()
        self.mount_options.assert_not_called()
        mock_post.assert_called_once()
        self.assertEqual(self.fuser._mount_src_map[local_path], (uri, 1))

    @patch("requests.post")
    def test_mount_prefixes_can_be_reused_without_sidecar_probe(self, mock_post):
        uri = "oss://bucket/model/"
        suffix = hashlib.md5(uri.encode()).hexdigest()
        for port, prefix in ((None, "/mnt/fuse"), ("12345", "/work/app/fuse")):
            with self.subTest(port=port), patch.object(
                HippoHelper, "host_fuse_port", return_value=port
            ), patch.object(HippoHelper, "app_workdir", "/work/app"), patch(
                "rtp_llm.utils.fuser._check_external_mount_access"
            ) as check, patch(
                "rtp_llm.utils.fuser._fuser", None
            ):
                self.external_mounts[f"{prefix}/{suffix}"] = {"ro"}
                self.assertEqual(fetch_remote_file_to_local(uri), f"{prefix}/{suffix}")
                check.assert_called_once_with(
                    f"{prefix}/{suffix}", MountRwMode.RWMODE_RO, {"ro"}
                )
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_availability_is_probed_on_demand_and_cached(self, mock_post):
        for status, expected in ((200, True), (500, False)):
            with self.subTest(status=status):
                mock_post.reset_mock()
                mock_post.return_value.status_code = status
                fuser = Fuser()
                self.addCleanup(atexit.unregister, fuser.umount_all)
                mock_post.assert_not_called()
                self.assertEqual(fuser.available, expected)
                self.assertEqual(fuser.available, expected)
                mock_post.assert_called_once()

    @patch("rtp_llm.utils.fuser._get_fuser")
    @patch("rtp_llm.utils.fuser._nfs_manager.mount_nfs_dir")
    def test_local_and_nas_paths_keep_existing_behavior(self, mock_nfs, mock_fuser):
        self.assertEqual(fetch_remote_file_to_local("/models/glm"), "/models/glm")
        mock_nfs.assert_not_called()
        mock_nfs.return_value = "/mounted/nas/model"
        self.assertEqual(
            fetch_remote_file_to_local("nas://server/model"), "/mounted/nas/model"
        )
        mock_nfs.assert_called_once_with("nas://server/model")
        mock_fuser.assert_not_called()

    @patch("requests.post")
    def test_mount_dir_success(self, mock_post):
        # Mock successful HTTP response
        mock_post.return_value = self.mock_response

        # Call the method
        mount_path = self.fuser.mount_dir("/path/to/dir")

        # Assert the response was as expected
        self.assertIsNotNone(mount_path)
        self.assertIn(mount_path, self.fuser._mount_src_map)

    @patch("requests.post")
    def test_mount_dir_rw_payload(self, mock_post):
        mock_post.return_value = self.mock_response

        self.fuser.mount_dir(
            "oss://bucket/jit-cache/", MountRwMode.RWMODE_RW, enable_mnt_ref=True
        )

        request = mock_post.call_args.kwargs["json"]
        self.assertEqual(request["rwMode"], "RWMODE_RW")
        self.assertTrue(request["enableMntRef"])
        self.assertEqual(
            request["cacheOptions"],
            {"writeMode": "WRITE_THROGH", "enableRemove": True},
        )

    def test_mount_dir_with_retries(self):
        # 准备side effects列表
        side_effects = [
            requests.exceptions.ConnectionError("Unable to connect"),
            Mock(
                status_code=200, json=lambda: {"errorCode": 1}
            ),  # 第二次请求返回errorCode 1
            Mock(status_code=200, json=lambda: {"errorCode": 0}),  # 第三次请求成功
        ]

        # Mock requests.post并设置side effects
        with patch("requests.post", side_effect=side_effects) as mock_post, patch(
            "rtp_llm.utils.fuser.time.sleep"
        ):
            mount_path = self.fuser.mount_dir("test-path")
            self.assertEqual(mock_post.call_count, 3)

        self.assertIsNotNone(mount_path)

    @patch("rtp_llm.utils.fuser.time.sleep")
    def test_failed_mount_directory_does_not_bypass_retry(self, mock_sleep):
        uri = "oss://bucket/model/"
        mount_path = self.mount_path(uri)

        def mount_response(*args, **kwargs):
            if not mount_path.exists():
                mount_path.mkdir()
                self.external_mounts[str(mount_path)] = {"rw"}
                return Mock(json=lambda: {"errorCode": 1})
            return self.mock_response

        with patch("requests.post", side_effect=mount_response) as mock_post:
            self.assertEqual(self.fuser.mount_or_reuse_dir(uri), str(mount_path))
            self.assertEqual(mock_post.call_count, 2)
            self.assertEqual(self.fuser._mount_src_map[str(mount_path)], (uri, 1))

    @patch("requests.post")
    def test_umount_fuse_dir_success(self, mock_post):
        # Mock successful HTTP response and a previous successful mount
        self.fuser._mount_src_map["/mnt/fuse/dummyhash"] = ("/path/to/dir", 1)
        mock_post.return_value = self.mock_response

        # Call the method
        self.fuser.umount_fuse_dir("/mnt/fuse/dummyhash")

        # Assert the response was as expected
        self.assertNotIn("/mnt/fuse/dummyhash", self.fuser._mount_src_map)

    @patch("requests.post")
    def test_umount_fuse_dir_failure(self, mock_post):
        # Mock failure HTTP response
        self.mock_response.json.return_value = {"errorCode": 1}
        self.fuser._mount_src_map["/mnt/fuse/dummyhash"] = ("/path/to/dir", 1)
        mock_post.return_value = self.mock_response

        with self.assertRaises(Exception) as cm:
            self.fuser.umount_fuse_dir("/mnt/fuse/dummyhash")

        # Assert the response was as expected
        self.assertIn("/mnt/fuse/dummyhash", self.fuser._mount_src_map)

    def test_umount_all(self):
        # Setup some mounts
        self.fuser._mount_src_map = {
            "/mnt/fuse/hash1": ("/path/to/dir1", 1),
            "/mnt/fuse/hash2": ("/path/to/dir2", 1),
        }

        # Mock the umount_fuse_dir method to just remove the key from the map
        def mock_umount(mnt_path: str, force: bool = False):
            self.fuser._mount_src_map.pop(mnt_path, None)

        with patch.object(
            Fuser, "umount_fuse_dir", wraps=Fuser.umount_fuse_dir
        ) as mock_func:
            mock_func.side_effect = lambda mnt_path, force: mock_umount(mnt_path, force)
            # Call the method
            self.fuser.umount_all()

        # Assert everything was umounted
        self.assertEqual(self.fuser._mount_src_map, {})


class FuseMountInfoTest(unittest.TestCase):
    def test_mount_namespace_and_options(self):
        root = "1 0 8:1 / / rw - ext4 /dev/root rw\n"
        fuse = "2 1 0:99 / /mnt/fuse rw - fuse.ossfs oss rw\n"
        cases = (
            ("ordinary directory", root, "/mnt/fuse/hash", None),
            ("exact mount", root + fuse, "/mnt/fuse", {"rw"}),
            ("parent mount", root + fuse, "/mnt/fuse/hash", {"rw"}),
            ("sibling prefix", root + fuse, "/mnt/fuse-other/hash", None),
            (
                "bind mount of fuse subdirectory",
                root + "2 1 0:99 /models/glm /mnt/fuse/hash ro - fuse.ossfs oss rw\n",
                "/mnt/fuse/hash",
                {"ro", "rw"},
            ),
            (
                "read-only superblock",
                root + "2 1 0:99 / /mnt/fuse rw - fuse.ossfs oss ro\n",
                "/mnt/fuse/hash",
                {"ro", "rw"},
            ),
            (
                "escaped mount path",
                root + "2 1 0:99 / /mnt/my\\040fuse\\134dir rw - fuse.ossfs oss rw\n",
                "/mnt/my fuse\\dir/hash",
                {"rw"},
            ),
            (
                "non-fuse child hides parent",
                root + fuse + "3 2 0:1 / /mnt/fuse/hash rw - tmpfs tmpfs rw\n",
                "/mnt/fuse/hash",
                None,
            ),
            (
                "stacked non-fuse mount in reverse record order",
                root + "3 2 0:1 / /mnt/fuse rw - tmpfs tmpfs rw\n" + fuse,
                "/mnt/fuse/hash",
                None,
            ),
            (
                "ancestor mount hides a deeper fuse sibling",
                root
                + "2 1 0:99 / /mnt/fuse/hash rw - fuse.ossfs oss rw\n"
                + "3 1 0:1 / /mnt/fuse rw - tmpfs tmpfs rw\n",
                "/mnt/fuse/hash",
                None,
            ),
            (
                "ancestor overmount hides a deeper fuse child",
                root
                + fuse
                + "3 2 0:100 / /mnt/fuse/hash rw - fuse.ossfs oss rw\n"
                + "4 2 0:1 / /mnt/fuse rw - tmpfs tmpfs rw\n",
                "/mnt/fuse/hash",
                None,
            ),
            (
                "fuse child on the visible overmount",
                root
                + fuse
                + "3 2 0:100 / /mnt/fuse/hash ro - fuse.ossfs old ro\n"
                + "4 2 0:1 / /mnt/fuse rw - tmpfs tmpfs rw\n"
                + "5 4 0:101 / /mnt/fuse/hash rw - fuse.ossfs new rw\n",
                "/mnt/fuse/hash",
                {"rw"},
            ),
            ("malformed entry", root + "invalid mount entry\n", "/mnt/fuse", None),
        )
        for label, contents, path, expected in cases:
            with self.subTest(label=label), patch(
                "rtp_llm.utils.fuser.open", mock_open(read_data=contents)
            ):
                self.assertEqual(_fuse_mount_options(path), expected)


class RetryDecoratorTest(unittest.TestCase):

    def test_retry_decorator_timeout(self):
        # Mock function that always raises an exception
        @retry_with_timeout(
            timeout_seconds=5, retry_interval=1, exceptions=(ValueError,)
        )
        def mock_function():
            raise ValueError("Deliberate exception for testing.")

        # Start the clock just before running the function
        start_time = time.time()

        # Run the function and expect a TimeoutError
        with self.assertRaises(TimeoutError) as cm:
            mock_function()

        # Stop the clock immediately after the exception is raised
        end_time = time.time()

        # Check if the TimeoutError contains the correct message
        self.assertIn("timed out after 5 seconds", str(cm.exception))

        # Check if the elapsed time is greater than or equal to the timeout
        self.assertTrue(end_time - start_time >= 5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
