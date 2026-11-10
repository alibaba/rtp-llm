import hashlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import q36_native_source_guard as guard


class NativeSourceGuardTest(unittest.TestCase):
    def test_qualified_hook_and_reject_conflicting_or_cached_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); good = root / 'qualified'; bad = root / 'deploy-arm-rope'
            good.mkdir(); bad.mkdir()
            expected = good / 'rtp_arm_bf16_import_hook.py'; expected.write_text('qualified\n')
            wrong = bad / expected.name; wrong.write_text('different\n')
            env = {'Q36_ARM_ROPE_DIR': str(good), 'Q36_ARM_ROPE_SHA256': hashlib.sha256(expected.read_bytes()).hexdigest()}
            with patch.dict(os.environ, env), patch.object(sys, 'path', [str(good), str(bad)]):
                guard.verify_rope_source()
                with patch.object(sys, 'path', [str(bad), str(good)]):
                    with self.assertRaisesRegex(RuntimeError, 'resolves outside'):
                        guard.verify_rope_source()
                with patch.dict(sys.modules, {'rtp_arm_bf16_import_hook': types.SimpleNamespace(__file__=str(wrong))}):
                    with self.assertRaisesRegex(RuntimeError, 'loaded ARM'):
                        guard.verify_rope_source()
                expected.write_text('tampered\n')
                with self.assertRaisesRegex(RuntimeError, 'digest mismatch'):
                    guard.verify_rope_source()


if __name__ == '__main__':
    unittest.main()
