"""Exercise the real post-sync probe method in both qualification modes."""
import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch


source = Path(__file__).with_name('rtp_native_server.py')
tree = ast.parse(source.read_text())
method = next(item for cls in tree.body if isinstance(cls, ast.ClassDef)
              for item in cls.body if isinstance(item, ast.AsyncFunctionDef)
              and item.name == 'probe_synced_policy')
helper = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
              and item.name == '_require_real_update_for_probe')
module = ast.fix_missing_locations(ast.Module(body=[helper, method], type_ignores=[]))


class PostSyncGateTest(unittest.TestCase):
    def test_unchanged_is_allowed_normally_and_rejected_in_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('qualified-test-model')
            values = [-0.5]
            score_sha = hashlib.sha256(struct.pack('<f', *values)).hexdigest()
            baseline = {'prompt_ids': [1], 'response_ids': [1], 'model_version': 'v0',
                        'full_source_sha256': 'same', 'score_sha256': score_sha}
            (root / 'rtp_probe_rank0_step0.json').write_text(json.dumps(baseline))
            (root / 'rtp_sync_rank0_step1.json').write_text(json.dumps({'model_version': 'v1',
                                                                        'full_source_sha256': 'same'}))
            records = []
            ns = {'json': json, 'os': os, 'struct': struct, 'hashlib': hashlib, 'Path': Path,
                  'Q36_ZERO_UPDATE_CONFIG_SHA': hashlib.sha256((root / 'config.json').read_bytes()).hexdigest(),
                  '_audit_root': lambda: root, '_write_json': lambda p, r: records.append(r)}
            exec(compile(module, str(source), 'exec'), ns)
            class Tokenizer:
                def encode(self, *args, **kwargs): return [1]
            class Server:
                tokenizer = Tokenizer()
                replica_rank = 0
                tp_size = 4
                model_path = str(root)
                async def score_tokens(self, *args):
                    return {'global_step': 1, 'model_version': 'v1', 'log_probs': values}
            server = Server()
            with patch.dict(os.environ, {'RLLM_REQUIRE_REAL_UPDATE': '0', 'Q36_ZERO_UPDATE_PROFILE': '1'}, clear=True):
                result = asyncio.run(ns['probe_synced_policy'](server, 1, 'v1'))
                self.assertFalse(result['changed_from_step0'])
                self.assertFalse(result['scores_changed_from_step0'])
            for env in ({}, {'RLLM_REQUIRE_REAL_UPDATE': '0'},
                        {'RLLM_REQUIRE_REAL_UPDATE': '1', 'Q36_ZERO_UPDATE_PROFILE': '1'}):
                with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                    with self.assertRaisesRegex(RuntimeError, 'weights did not change'):
                        asyncio.run(ns['probe_synced_policy'](server, 1, 'v1'))
            for env in ({'RLLM_REQUIRE_REAL_UPDATE': '2', 'Q36_ZERO_UPDATE_PROFILE': '1'},):
                with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                    with self.assertRaisesRegex(ValueError, 'invalid'):
                        asyncio.run(ns['probe_synced_policy'](server, 1, 'v1'))
            with patch.dict(os.environ, {'RLLM_REQUIRE_REAL_UPDATE': '0', 'Q36_ZERO_UPDATE_PROFILE': '1'}, clear=True):
                (root / 'config.json').write_text('different-model')
                with self.assertRaisesRegex(RuntimeError, 'weights did not change'):
                    asyncio.run(ns['probe_synced_policy'](server, 1, 'v1'))
                (root / 'config.json').write_text('qualified-test-model')
                (root / 'rtp_probe_rank0_step0.json').write_text(json.dumps({**baseline, 'model_version': 'v1'}))
                with self.assertRaisesRegex(RuntimeError, 'initial engine version'):
                    asyncio.run(ns['probe_synced_policy'](server, 1, 'v1'))
                (root / 'rtp_probe_rank0_step0.json').write_text(json.dumps({**baseline, 'prompt_ids': [2]}))
                with self.assertRaisesRegex(RuntimeError, 'tokenization differs'):
                    asyncio.run(ns['probe_synced_policy'](server, 1, 'v1'))
            # Non-Q36 models retain the original strict behavior, including
            # when they inherit both the Q36 profile and an invalid mode.
            (root / 'config.json').write_text('different-model')
            (root / 'rtp_probe_rank0_step0.json').write_text(json.dumps(baseline))
            (root / 'rtp_sync_rank0_step1.json').write_text(json.dumps({'model_version': 'v1',
                                                                        'full_source_sha256': 'changed'}))
            values[:] = [-0.7]
            with patch.dict(os.environ, {'RLLM_REQUIRE_REAL_UPDATE': 'invalid', 'Q36_ZERO_UPDATE_PROFILE': '1'}, clear=True):
                result = asyncio.run(ns['probe_synced_policy'](server, 1, 'v1'))
                self.assertTrue(result['changed_from_step0'])
            self.assertEqual(len(records), 2)


if __name__ == '__main__':
    unittest.main()
