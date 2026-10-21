import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('release_bundle', Path(__file__).with_name('bundle.py'))
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class ReleaseLauncherTest(unittest.TestCase):
    def test_independent_roles_and_explicit_pool_capacity(self):
        with tempfile.TemporaryDirectory() as root:
            args = bundle.command({'RTP_LLM_MOCK_BUNDLE': '1', 'MOCK_BUNDLE_ROLE': 'mock',
                'POD_IP': '10.1.2.3', 'FLEXLB_CONFIG': '{}', 'MOCK_BUNDLE_RUN_DIR': root})
            def value(flag):
                return args[args.index('--' + flag) + 1]
            self.assertEqual(value('prefill-block-size'), '512')
            self.assertEqual(value('decode-block-size'), '128')
            self.assertEqual(value('prefill-cache-blocks'), '45083')
            self.assertEqual(value('decode-cache-blocks'), '932802')
            self.assertNotIn('--whale', args)  # Not a flag in the frozen release.
            self.assertEqual(json.loads((Path(root) / 'master-config.json').read_text()), {})

    def test_invalid_test_boundary_or_port_range_is_rejected(self):
        for overrides in [{'RTP_LLM_MOCK_BUNDLE':'0'}, {'MOCK_BUNDLE_ROLE':'bundle'},
                          {'POD_IP':'127.0.0.1'}, {'MOCK_GRPC_BASE_PORT':'65535'}]:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as root:
                env = {'RTP_LLM_MOCK_BUNDLE':'1','MOCK_BUNDLE_ROLE':'mock','POD_IP':'10.1.2.3',
                       'FLEXLB_CONFIG':'{}','MOCK_BUNDLE_RUN_DIR':root, **overrides}
                with self.assertRaises(ValueError):
                    bundle.command(env)


if __name__ == '__main__':
    unittest.main()
