import time
import unittest
import requests
import logging
from unittest.mock import patch, MagicMock, Mock
from maga_transformer.utils.fuser import Fuser, retry_with_timeout

class TestFuser(unittest.TestCase):
    def setUp(self):
        self.fuser = Fuser()
        self.mock_response = MagicMock()
        self.mock_response.json.return_value = {'errorCode': 0}

    def tearDown(self):
        self.fuser._mount_src_map = {}

    @patch('requests.post')
    def test_mount_dir_success(self, mock_post):
        # Mock successful HTTP response
        mock_post.return_value = self.mock_response

        # Call the method
        mount_path = self.fuser.mount_dir('/path/to/dir')

        # Assert the response was as expected
        self.assertIsNotNone(mount_path)
        self.assertIn(mount_path, self.fuser._mount_src_map)

    def test_mount_dir_with_retries(self):
        # 准备side effects列表
        side_effects = [
            requests.exceptions.ConnectionError("Unable to connect"),
            Mock(status_code=200, json=lambda: {'errorCode': 1}),  # 第二次请求返回errorCode 1
            Mock(status_code=200, json=lambda: {'errorCode': 0})   # 第三次请求成功
        ]

        # Mock requests.post并设置side effects
        with patch('requests.post', side_effect=side_effects) as mock_post:
            fuser = Fuser()
            mount_path = fuser.mount_dir('test-path')
            self.assertEqual(mock_post.call_count, 3)
            fuser._mount_src_map = {}

        self.assertIsNotNone(mount_path)

    @patch('requests.post')
    def test_umount_fuse_dir_success(self, mock_post):
        # Mock successful HTTP response and a previous successful mount
        self.fuser._mount_src_map['/mnt/fuse/dummyhash'] = ('/path/to/dir', 1)
        mock_post.return_value = self.mock_response

        # Call the method
        self.fuser.umount_fuse_dir('/mnt/fuse/dummyhash')

        # Assert the response was as expected
        self.assertNotIn('/mnt/fuse/dummyhash', self.fuser._mount_src_map)

    @patch('requests.post')
    def test_umount_fuse_dir_failure(self, mock_post):
        # Mock failure HTTP response
        self.mock_response.json.return_value = {'errorCode': 1}
        self.fuser._mount_src_map['/mnt/fuse/dummyhash'] = ('/path/to/dir',1)
        mock_post.return_value = self.mock_response

        with self.assertRaises(Exception) as cm:
            self.fuser.umount_fuse_dir('/mnt/fuse/dummyhash')

        # Assert the response was as expected
        self.assertIn('/mnt/fuse/dummyhash', self.fuser._mount_src_map)

    def test_umount_all(self):
        # Setup some mounts
        self.fuser._mount_src_map = {
            '/mnt/fuse/hash1': ('/path/to/dir1', 1),
            '/mnt/fuse/hash2': ('/path/to/dir2', 1),
        }

        # Mock the umount_fuse_dir method to just remove the key from the map
        def mock_umount(mnt_path: str, force: bool = False):
            self.fuser._mount_src_map.pop(mnt_path, None)


        with patch.object(Fuser, 'umount_fuse_dir', wraps=Fuser.umount_fuse_dir) as mock_func:
            mock_func.side_effect = lambda mnt_path, force: mock_umount(mnt_path, force)
            # Call the method
            self.fuser.umount_all()

        # Assert everything was umounted
        self.assertEqual(self.fuser._mount_src_map, {})

class RetryDecoratorTest(unittest.TestCase):

    def test_retry_decorator_timeout(self):
        # Mock function that always raises an exception
        @retry_with_timeout(timeout_seconds=5, retry_interval=1, exceptions=(ValueError,))
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
