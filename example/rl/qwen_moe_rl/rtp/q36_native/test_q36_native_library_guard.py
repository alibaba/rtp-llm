import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import q36_native_library_guard as guard


class NativeLibraryGuardTest(unittest.TestCase):
    def test_exact_native_set_and_reject_mixed_same_named_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qualified = root / 'native'
            alias = root / 'innocent-name'
            qualified.mkdir(); alias.mkdir()
            libraries = {'libcudnn.so.9': b'old-cudnn', 'libnccl.so.2': b'old-nccl'}
            pins = {}
            for name, data in libraries.items():
                (qualified / name).write_bytes(data)
                pins[name] = (hashlib.sha256(data).hexdigest(), len(data))
            maps = root / 'maps'
            map_files = root / 'map_files'; map_files.mkdir()
            def line(path, address):
                stat = path.stat()
                (map_files / address).symlink_to(path)
                return ('%s r-xp 0 %x:%x %d %s\n' %
                        (address, os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino, path))
            maps.write_text(''.join(line(qualified / name, f'{i}000-{i}fff') for i, name in enumerate(libraries)))
            with patch.object(guard, 'NATIVE_LIBRARIES', pins):
                guard.verify_native_search_path(str(qualified))
                self.assertEqual(len(guard.verify_native_mappings(92000, maps, map_files)), 2)
                with self.assertRaisesRegex(RuntimeError, 'runtime'):
                    guard.verify_native_mappings(92801, maps, map_files)
                (alias / 'libcudnn.so.9').write_bytes(b'new-cudnn')
                with self.assertRaisesRegex(RuntimeError, 'digest'):
                    guard.verify_native_search_path(str(alias) + ':' + str(qualified))
                (map_files / '0000-0fff').unlink()
                (map_files / '1000-1fff').unlink()
                maps.write_text(line(alias / 'libcudnn.so.9', '0000-0fff') + line(qualified / 'libnccl.so.2', '1000-1fff'))
                with self.assertRaisesRegex(RuntimeError, 'digest'):
                    guard.verify_native_mappings(92000, maps, map_files)


if __name__ == '__main__':
    unittest.main()
