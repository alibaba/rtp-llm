import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import master_vip
import vit


class PerEngineRolesTest(unittest.TestCase):
    def test_vit_uses_real_metadata_and_own_http_grpc_pair(self):
        env = dict(RTP_LLM_MOCK_BUNDLE="1", ROLE_TYPE="VIT", START_PORT="22160",
                   TOKENIZER_PATH="oss://actual-model", VIT_MAX_CONCURRENT_REQUESTS="3")
        with patch.object(vit, "prepare_assets", return_value=Path("/model")) as prepare:
            args = vit.command(env)
        prepare.assert_called_once()
        for key, value in (("--http-port", "22160"), ("--port", "22161"),
                           ("--model-path", "/model"), ("--concurrency", "3")):
            self.assertEqual(args[args.index(key) + 1], value)

    def test_vit_rejects_role_model_and_port_errors_before_start(self):
        base = dict(RTP_LLM_MOCK_BUNDLE="1", ROLE_TYPE="VIT", TOKENIZER_PATH="/model")
        for override in ({"ROLE_TYPE": "PREFILL"}, {"START_PORT": "65535"},
                         {"TOKENIZER_PATH": ""}, {"RTP_LLM_MOCK_BUNDLE": "0"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                vit.command({**base, **override})

    def test_master_requires_real_vip_adapter_and_keeps_native_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "build-manifest.json"
            manifest.write_text(json.dumps({"components": {"master": {"vipserver_packaged": True}}}))
            env = dict(RTP_LLM_MOCK_BUNDLE="1", MOCK_BUNDLE_ROOT=directory,
                       MODEL_SERVICE_CONFIG='{"role_endpoints": []}', START_PORT="28080")
            args = master_vip.command(env)
            self.assertIn("--server.port=28080", args)
            self.assertIn("--management.server.port=28081", args)
            with self.assertRaisesRegex(ValueError, "actual VIP"):
                master_vip.command({**env, "MODEL_SERVICE_CONFIG": '{"discovery_file":"fake"}'})
            manifest.write_text('{"components":{"master":{"vipserver_packaged":false}}}')
            with self.assertRaisesRegex(ValueError, "VipServer"):
                master_vip.command(env)
