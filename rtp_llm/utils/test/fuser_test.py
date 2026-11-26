import errno
import hashlib
import logging
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import requests

import rtp_llm.utils.fuser as fuser
from rtp_llm.utils.fuser import Fuser, MountRwMode, _redact_uri, retry_with_timeout


class TestFuser(unittest.TestCase):
    def setUp(self):
        self.fuser = Fuser()
        self.mock_response = MagicMock()
        self.mock_response.json.return_value = {"errorCode": 0}

    def tearDown(self):
        self.fuser._mount_src_map = {}

    def external_mount_path(self, uri: str) -> Path:
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        self.fuser._fuse_path_prefix = root.name
        path = Path(root.name) / hashlib.md5(uri.encode()).hexdigest()
        path.mkdir()
        return path

    def test_visible_mount_determines_fuse_access(self):
        mounts = (
            "1 0 8:1 / / rw - ext4 /dev/root rw\n"
            "2 1 0:2 / /models ro - fuse.ossfs oss ro\n"
            "3 2 0:3 / /models/local rw - tmpfs tmpfs rw\n"
        )
        with patch("builtins.open", mock_open(read_data=mounts)):
            self.assertEqual(fuser._fuse_mount_options("/models/weights"), {"ro"})
            self.assertIsNone(fuser._fuse_mount_options("/models/local/weights"))

    @patch("requests.post")
    def test_borrowed_mount_is_not_released(self, post):
        uri = "oss://bucket/model"
        path = self.external_mount_path(uri)
        with patch.object(fuser, "_fuse_mount_options", return_value={"ro"}):
            self.assertEqual(self.fuser.mount_or_reuse_dir(uri), str(path))
            self.fuser.umount_fuse_dir(str(path), force=True)
        self.assertTrue(path.is_dir())
        post.assert_not_called()

    @patch("requests.post")
    def test_readonly_mount_rejects_write(self, post):
        uri = "oss://bucket/model"
        self.external_mount_path(uri)
        with patch.object(
            fuser, "_fuse_mount_options", return_value={"ro"}
        ), self.assertRaises(PermissionError) as error:
            self.fuser.mount_or_reuse_dir(uri, MountRwMode.RWMODE_RW)
        self.assertEqual(error.exception.errno, errno.EROFS)
        post.assert_not_called()

    @patch("requests.post")
    def test_managed_mount_lasts_until_last_release(self, post):
        uri = "oss://bucket/model"
        post.return_value = self.mock_response
        with patch.object(fuser, "_fuse_mount_options", return_value=None) as options:
            path = self.fuser.mount_or_reuse_dir(uri)
            options.return_value = {"ro"}
            self.assertEqual(
                self.fuser.mount_or_reuse_dir(uri, MountRwMode.RWMODE_RW), path
            )
        post.reset_mock()
        self.fuser.umount_fuse_dir(path)
        post.assert_not_called()
        self.fuser.umount_fuse_dir(path)
        post.assert_called_once_with(
            f"{self.fuser._fuse_uri}/FuseService/umount",
            json={"mountDir": path},
            timeout=600,
        )

    @patch("requests.post")
    def test_mount_dir_success(self, mock_post):
        # Mock successful HTTP response
        mock_post.return_value = self.mock_response

        # Call the method
        mount_path = self.fuser.mount_dir("/path/to/dir")

        # Assert the response was as expected
        self.assertIsNotNone(mount_path)
        self.assertIn(mount_path, self.fuser._mount_src_map)

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
        with patch("requests.post", side_effect=side_effects) as mock_post:
            fuser = Fuser()
            mount_path = fuser.mount_dir("test-path")
            self.assertEqual(mock_post.call_count, 3)
            fuser._mount_src_map = {}

        self.assertIsNotNone(mount_path)

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

    def test_module_umount_all_releases_fuse_and_nfs(self):
        # The shutdown hook must force-unmount both FUSE and NFS (os._exit path).
        # umount_all() only touches an already-constructed Fuser; force one so
        # the hook has an instance to act on regardless of test ordering.
        fuser._get_fuser()
        with patch.object(fuser._fuser, "umount_all") as fuse_umount, patch.object(
            fuser._nfs_manager, "unmount_all"
        ) as nfs_umount:
            fuser.umount_all()
        fuse_umount.assert_called_once_with(force=True)
        nfs_umount.assert_called_once_with()

    def test_redact_oss_uri_credentials(self):
        uri = (
            "oss://bucket/prefix?OSS_ACCESS_ID=test-id"
            "&OSS_ACCESS_KEY=test-secret"
            "&OSS_ENDPOINT=oss-cn-shanghai.aliyuncs.com"
        )

        redacted = _redact_uri(uri)

        self.assertNotIn("test-id", redacted)
        self.assertNotIn("test-secret", redacted)
        self.assertIn("OSS_ENDPOINT=oss-cn-shanghai.aliyuncs.com", redacted)
        self.assertIn("OSS_ACCESS_ID=", redacted)
        self.assertIn("OSS_ACCESS_KEY=", redacted)


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
