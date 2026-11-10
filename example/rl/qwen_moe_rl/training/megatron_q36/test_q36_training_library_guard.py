import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import q36_training_library_guard as guard


class OverlayIdentityTest(unittest.TestCase):
    def test_install_is_passive_until_assigned_worker_build(self):
        class Engine:
            def _build_megatron_module(self): return 'built'
        module = types.SimpleNamespace(MegatronEngine=Engine)
        with patch.object(guard, 'verify_runtime') as verify:
            guard.install(module)
            verify.assert_not_called()
            self.assertEqual(Engine()._build_megatron_module(), 'built')
            verify.assert_called_once()

    def test_reject_missing_extra_and_nonidentical_torch_overlay_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / 'base'; overlay = root / 'isolated/pyoverlay/torch'
            base.mkdir(); overlay.mkdir(parents=True)
            (base / '__init__.py').write_text('torch')
            os.link(base / '__init__.py', overlay / '__init__.py')
            env = {'Q36_BASE_TORCH_DIR': str(base), 'Q36_CUDNN_ROOT': str(root / 'isolated')}
            self.assertEqual(guard.validate_overlay(env), 1)
            (overlay / 'extra.py').write_text('extra')
            with self.assertRaisesRegex(RuntimeError, 'file set'):
                guard.validate_overlay(env)
            (overlay / 'extra.py').unlink()
            (overlay / '__init__.py').unlink()
            (overlay / '__init__.py').write_text('torch')
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                guard.validate_overlay(env)
            (overlay / '__init__.py').unlink()
            with self.assertRaisesRegex(RuntimeError, 'file set'):
                guard.validate_overlay(env)


if __name__ == '__main__':
    unittest.main()
