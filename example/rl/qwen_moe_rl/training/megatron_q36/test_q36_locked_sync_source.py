import hashlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import q36_nonshared_worker_bootstrap as bootstrap


class LockedSyncSourceTest(unittest.TestCase):
    def test_exact_origin_hash_and_cached_module_are_required(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            source = Path(a) / 'q36_rllm_sync_compat.py'
            source.write_text('value = 7\n')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            hostile = Path(b) / source.name
            hostile.write_text('value = 8\n')
            with patch.dict(os.environ, {'Q36_CODE_DIR': a}), patch.object(sys, 'path', [b, a] + sys.path):
                with patch.dict(sys.modules):
                    sys.modules.pop('q36_rllm_sync_compat', None)
                    with self.assertRaisesRegex(RuntimeError, 'source mismatch'):
                        bootstrap._locked_import('q36_rllm_sync_compat', digest)
            with patch.dict(os.environ, {'Q36_CODE_DIR': a}), patch.object(sys, 'path', [a] + sys.path):
                with patch.dict(sys.modules):
                    sys.modules.pop('q36_rllm_sync_compat', None)
                    self.assertEqual(bootstrap._locked_import('q36_rllm_sync_compat', digest).value, 7)
                    source.write_text('value = 9\n')
                    with self.assertRaisesRegex(RuntimeError, 'source mismatch'):
                        bootstrap._locked_import('q36_rllm_sync_compat', digest)


if __name__ == '__main__': unittest.main()
