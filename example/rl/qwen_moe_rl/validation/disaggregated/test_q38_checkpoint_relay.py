"""Verify bounded file access and fail-closed response integrity."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from q38_checkpoint_relay import MAX_RANGE_BYTES, Range, fetch, read_local


class RelayTest(unittest.TestCase):
    def test_read_exact_offset_and_reject_changed_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'model-00001-of-00213.safetensors'
            path.write_bytes(bytes(range(64)))
            selected = Range(path.name, 64, 23, 15)
            self.assertEqual(read_local(root, selected), bytes(range(23, 38)))
            path.write_bytes(bytes(range(63)))
            with self.assertRaises(ValueError):
                read_local(root, selected)

    def test_does_not_follow_symlinks_or_allow_arbitrary_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'unapproved').write_bytes(b'text')
            (root/'config.json').symlink_to(root/'unapproved')
            with self.assertRaises(OSError):
                read_local(root, Range('config.json', 4, 0, 4))
        for name in ('../config.json', 'unapproved', 'model-00000-of-00001.safetensors'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                Range(name, 4, 0, 4)

    def test_invalid_coordinates_are_rejected_before_network_use(self):
        for offset, size, length in [(-1, 4, 2), (3, 4, 2), (0, 4, 0),
                                     (0, MAX_RANGE_BYTES+1, MAX_RANGE_BYTES+1)]:
            with self.subTest(offset=offset, size=size, length=length), self.assertRaises(ValueError):
                Range('config.json', size, offset, length)

    @patch('q38_checkpoint_relay.HTTPConnection')
    def test_receiver_rejects_wrong_range_and_corrupt_bytes(self, connection):
        selected = Range('model-00001-of-00213.safetensors', 64, 10, 4)
        reply = connection.return_value.getresponse.return_value
        reply.status = 206
        headers = {'Content-Length': '4', 'Content-Range': 'bytes 11-14/64',
                   'X-Checkpoint-SHA256': 'a'*64}
        reply.getheader.side_effect = headers.get
        reply.read.return_value = b'bad!'
        with self.assertRaisesRegex(ValueError, 'unexpected checkpoint range'):
            fetch('10.0.0.1', 12345, selected, local_ip='10.0.0.2')
        reply.read.assert_not_called()
        headers['Content-Range'] = selected.content_range
        with self.assertRaisesRegex(ValueError, 'integrity'):
            fetch('10.0.0.1', 12345, selected, local_ip='10.0.0.2')
        self.assertEqual(connection.return_value.close.call_count, 2)


if __name__ == '__main__':
    unittest.main()
