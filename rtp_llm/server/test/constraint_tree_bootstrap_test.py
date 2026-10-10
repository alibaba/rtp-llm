import importlib.util
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

# Keep this control-plane test independent of CUDA/model imports.
spec = importlib.util.spec_from_file_location(
    "constraint_tree_bootstrap",
    Path(__file__).resolve().parents[1] / "constraint_tree_bootstrap.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ConstraintTreeBootstrap = module.ConstraintTreeBootstrap


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"CONSTRAINT_TREE_MASTER_ENDPOINT": ""})
        env.start()
        self.addCleanup(env.stop)

    def test_enabled_only_for_required_inference_roles_and_requires_master(self):
        host = Mock()
        with patch.dict(os.environ, {"CONSTRAINT_TREE_REQUIRED": "false"}):
            self.assertIsNone(ConstraintTreeBootstrap.from_env(host, 12345, "PDFUSION"))
        with patch.dict(
            os.environ,
            {"CONSTRAINT_TREE_REQUIRED": "true", "MODEL_SERVICE_CONFIG": "{}"},
        ):
            self.assertIsNone(ConstraintTreeBootstrap.from_env(host, 12345, "PREFILL"))
            with self.assertRaisesRegex(ValueError, "CONSTRAINT_TREE_MASTER_ENDPOINT"):
                ConstraintTreeBootstrap.from_env(host, 12345, "PDFUSION")
            with patch.dict(
                os.environ,
                {
                    "MODEL_SERVICE_CONFIG": json.dumps(
                        {
                            "service_id": "aigc.text-generation.generation.engine_service",
                            "master_endpoint": {"address": "master.vip"},
                        }
                    )
                },
            ):
                bootstrap = ConstraintTreeBootstrap.from_env(
                    host, 12345, SimpleNamespace(name="DECODE")
                )
                self.assertEqual(12345, bootstrap.body["http_port"])
                self.assertEqual("DECODE", bootstrap.body["role"])

    def test_tree_vip_only_needs_no_model_service_config(self):
        for config in (None, "", "{}"):
            with self.subTest(config=config), patch.dict(
                os.environ,
                {
                    "CONSTRAINT_TREE_REQUIRED": "true",
                    "CONSTRAINT_TREE_MASTER_ENDPOINT": "tree.master.vip",
                },
            ):
                if config is None:
                    os.environ.pop("MODEL_SERVICE_CONFIG", None)
                else:
                    os.environ["MODEL_SERVICE_CONFIG"] = config
                host = Mock()
                bootstrap = ConstraintTreeBootstrap.from_env(host, 23495, "PDFUSION")
                self.assertEqual(
                    {"http_port": 23495, "role": "PDFUSION"}, bootstrap.body
                )
                host.get_master_addr.assert_not_called()

    def test_legacy_routing_master_still_requires_explicit_service(self):
        with patch.dict(
            os.environ,
            {
                "CONSTRAINT_TREE_REQUIRED": "true",
                "MODEL_SERVICE_CONFIG": '{"master_endpoint":{"address":"routing.vip"}}',
            },
        ):
            with self.assertRaisesRegex(ValueError, "service_id"):
                ConstraintTreeBootstrap.from_env(Mock(), 23495, "PDFUSION")

    def test_tree_vip_is_independent_of_inference_routing_and_refreshed(self):
        config = json.dumps({"service_id": "service"})
        discovery = Mock()
        discovery.get_host_list_by_domain_now.side_effect = [
            [],
            [SimpleNamespace(ip="127.0.0.1", port=7001)],
            [SimpleNamespace(ip="127.0.0.2", port=7002)],
        ]
        host_service = Mock()
        with patch.dict(
            os.environ,
            {
                "CONSTRAINT_TREE_REQUIRED": "true",
                "CONSTRAINT_TREE_MASTER_ENDPOINT": " tree.master.vip ",
                "MODEL_SERVICE_CONFIG": config,
            },
        ), patch.dict(sys.modules, {"rtp_llm.vipserver": discovery}):
            bootstrap = ConstraintTreeBootstrap.from_env(
                host_service, 23495, "PDFUSION"
            )
            discovery.get_host_list_by_domain_now.assert_not_called()
            self.assertIsNone(bootstrap.master_address())
            self.assertEqual("127.0.0.1:7001", bootstrap.master_address())
            self.assertEqual("127.0.0.2:7002", bootstrap.master_address())
            discovery.get_host_list_by_domain_now.assert_called_with("tree.master.vip")
            host_service.get_master_addr.assert_not_called()
            self.assertEqual(config, os.environ["MODEL_SERVICE_CONFIG"])

    def test_dedicated_vip_takes_precedence_without_changing_flexlb_config(self):
        config = json.dumps(
            {"service_id": "service", "master_endpoint": {"address": "routing.vip"}}
        )
        host_service = Mock()
        with patch.dict(
            os.environ,
            {
                "CONSTRAINT_TREE_REQUIRED": "true",
                "CONSTRAINT_TREE_MASTER_ENDPOINT": "tree.vip",
                "MODEL_SERVICE_CONFIG": config,
            },
        ), patch.object(
            ConstraintTreeBootstrap, "_master_from_vip", return_value="tree:7001"
        ) as resolve:
            bootstrap = ConstraintTreeBootstrap.from_env(
                host_service, 23495, "PDFUSION"
            )
            self.assertEqual("tree:7001", bootstrap.master_address())
            resolve.assert_called_once_with("tree.vip")
            host_service.get_master_addr.assert_not_called()
            self.assertEqual(config, os.environ["MODEL_SERVICE_CONFIG"])

    def test_missing_tree_and_routing_endpoint_fails_at_startup(self):
        with patch.dict(
            os.environ,
            {
                "CONSTRAINT_TREE_REQUIRED": "true",
                "MODEL_SERVICE_CONFIG": '{"service_id":"service"}',
            },
        ):
            with self.assertRaisesRegex(ValueError, "CONSTRAINT_TREE_MASTER_ENDPOINT"):
                ConstraintTreeBootstrap.from_env(Mock(), 23495, "PDFUSION")

    def test_needs_both_discovery_and_native_readiness(self):
        bootstrap = ConstraintTreeBootstrap(
            lambda: "master:123", "service", 23495, "PDFUSION"
        )
        session = Mock()
        for discovered, code, expected in (
            (False, 200, False),
            (True, 503, False),
            (True, 200, True),
        ):
            session.post.return_value.json.return_value = {"discovered": discovered}
            session.get.return_value.status_code = code
            session.get.return_value.json.return_value = "ok"
            self.assertEqual(expected, bootstrap._register_once(session))
        session.post.assert_called_with(
            "http://master:123" + bootstrap.PATH, json=bootstrap.body, timeout=(2, 5)
        )
        session.get.return_value.json.return_value = {"error": "not ready"}
        self.assertFalse(bootstrap._register_once(session))

    def test_real_http_redirect_failure_retry_then_handoff(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.path == ConstraintTreeBootstrap.PATH:
                    self.send_response(307)
                    self.send_header("Location", "/leader")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                seen.append(json.loads(body))
                self.send_response(503 if len(seen) == 1 else 200)
                self.end_headers()
                self.wfile.write(b'{"discovered":true}')

            def do_GET(self):
                self.send_response(503 if len(seen) < 3 else 200)
                self.end_headers()
                self.wfile.write(b'"ok"')

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        bootstrap = ConstraintTreeBootstrap(
            lambda: f"127.0.0.1:{server.server_port}",
            "service",
            server.server_port,
            "PDFUSION",
            interval=0.01,
        )
        try:
            bootstrap.start()
            bootstrap._thread.join(timeout=5)
            self.assertFalse(bootstrap._thread.is_alive())
            self.assertEqual(3, len(seen))
            self.assertTrue(all(body == bootstrap.body for body in seen))
        finally:
            bootstrap.stop()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_master_unavailable_and_shutdown_are_bounded(self):
        bootstrap = ConstraintTreeBootstrap(
            lambda: None, "service", 12345, "DECODE", interval=100
        )
        with requests.Session() as session:
            with self.assertRaisesRegex(RuntimeError, "not yet discoverable"):
                bootstrap._register_once(session)
        bootstrap.start()
        bootstrap.stop()
        self.assertFalse(bootstrap._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
