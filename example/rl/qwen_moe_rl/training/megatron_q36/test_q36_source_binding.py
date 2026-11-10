import hashlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import q36_source_binding as binding


class SourceBindingTest(unittest.TestCase):
    def test_qualified_path_and_reject_shadow_or_preloaded_module(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); good = root / 'reviewed'; bad = root / 'shadow'
            good.mkdir(); bad.mkdir()
            source = good / 'rtp_verl_backend.py'; source.write_text('value = 1\n')
            (bad / source.name).write_text('value = 0\n')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            with patch.dict(os.environ, {'Q36_ADAPTER_DIR': str(good)}), \
                    patch.dict(binding.SOURCE_SHA, {'rtp_verl_backend': digest}), \
                    patch.object(sys, 'path', [str(good)]):
                binding.verify_source('rtp_verl_backend')
                with patch.object(sys, 'path', [str(bad), str(good)]):
                    with self.assertRaisesRegex(RuntimeError, 'resolved outside'):
                        binding.verify_source('rtp_verl_backend')
                with patch.dict(sys.modules, {'rtp_verl_backend': types.SimpleNamespace(__file__=str(bad / source.name))}):
                    with self.assertRaisesRegex(RuntimeError, 'loaded adapter'):
                        binding.verify_source('rtp_verl_backend')
                source.write_text('value = 2\n')
                with self.assertRaisesRegex(RuntimeError, 'digest mismatch'):
                    binding.verify_source('rtp_verl_backend')


if __name__ == '__main__':
    unittest.main()
