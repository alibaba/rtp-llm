"""Source migration checks; these do not substitute for native integration tests."""
import importlib.util
import os
from pathlib import Path
import re
import sys
import types
import unittest

ROOT = Path(os.environ.get("NAVI_MIGRATION_SOURCE", Path(__file__).resolve().parents[1]))
VENDOR = ROOT / "vendor"


def load_util():
    package = types.ModuleType("migration_util_test")
    package.__path__ = [str(ROOT / "navi_sched")]
    package._navi_sched_core = types.SimpleNamespace()
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        package.__name__ + ".util", ROOT / "navi_sched/util.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Fields:
    def __init__(self, **values):
        self.__dict__.update(values)


class MigrationVipOnlyTest(unittest.TestCase):
    def setUp(self):
        self.util = load_util()
        self.navi = types.SimpleNamespace(
            NaviRuntimeConfig=lambda: Fields(local_nodes=[], cm2_subscriptions=[]),
            NaviLocalNode=lambda: Fields(biz_name="", ip="", grpc_port=0),
            NaviCm2Subscription=lambda: Fields(zk_host="", zk_path=""),
            NaviVipSubscription=lambda: Fields(
                jmenv_domain="", services=[], service_key="", grpc_port_diff=0
            ),
        )

    def test_vip_metadata_is_preserved(self):
        fields = {
            "jmenv_domain": "jmenv.test",
            "services": ["master.test", "agent.test"],
            "service_key": "routing",
            "grpc_port_diff": 2,
        }
        runtime = self.util._makeRuntimeDetails(
            self.navi, {"vip_subscription": fields}
        )
        self.assertEqual(fields, vars(runtime.vip_subscription))

    def test_removed_discovery_options_are_rejected(self):
        for key, value in [
            ("local_nodes", [{"biz_name": "agent", "ip": "127.0.0.1", "grpc_port": 1234}]),
            ("cm2_subscriptions", [{"zk_host": "zk.test", "zk_path": "/cluster"}]),
        ]:
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "unknown runtime_config option: " + key):
                    self.util._makeRuntimeDetails(self.navi, {key: value})

    def test_only_vip_backend_is_registered(self):
        folder = VENDOR / "network/gig/multi_call/subscribe"
        factory = (folder / "SubscribeServiceFactory.cpp").read_text()
        self.assertIn("createVipSub", factory)
        self.assertIsNone(re.search("create(?:Cm2|Istio|Local)Sub", factory))
        for name in ["CM2SubscribeService", "IstioSubscribeService", "LocalSubscribeService", "XdsClient", "XdsStore", "GrpcAds"]:
            self.assertEqual([], list(folder.glob(name + ".*")), name)
        self.assertFalse((VENDOR / "apps/facility/cm2").exists())

    def test_removed_modules_and_build_references_are_absent(self):
        for module in ["apps/facility/cm2", "future_lite", "tair"]:
            self.assertFalse((VENDOR / module).exists(), module)
        forbidden = re.compile(
            r"rtp_llm/flexlb/navi_sched/vendor/(?:apps/facility/cm2|future_lite|tair)(?:/|:)"
            r"|future_lite/|future_lite::"
        )
        offenders = []
        for path in VENDOR.rglob("*"):
            if path.is_file() and (
                path.name in {"BUILD", "BUILD.bazel"}
                or path.suffix in {".h", ".cpp", ".cc", ".bzl"}
            ):
                if forbidden.search(path.read_text()):
                    offenders.append(str(path.relative_to(VENDOR)))
        self.assertEqual([], offenders)

    def test_allocator_headers_and_link_targets_are_absent(self):
        self.assertFalse((VENDOR / "memory/lockless_allocator").exists())
        offenders = []
        for path in VENDOR.rglob("*"):
            if path.is_file() and path.suffix in {".h", ".cpp", ".cc"}:
                if '#include "lockless_allocator/' in path.read_text():
                    offenders.append(str(path.relative_to(VENDOR)))
        self.assertEqual([], offenders)
        self.assertNotIn("wheel_lockless_allocator", (ROOT / "BUILD").read_text())

    def test_vendor_has_no_empty_directories(self):
        empty = [
            str(path.relative_to(VENDOR))
            for path in VENDOR.rglob("*")
            if path.is_dir() and not any(path.iterdir())
        ]
        self.assertEqual([], empty)

    def test_flat_vendor_layout_and_local_file_util(self):
        self.assertFalse((VENDOR / "alibaba").exists())
        self.assertFalse((VENDOR / "aios").exists())
        self.assertFalse((VENDOR / "filesystem").exists())
        header = VENDOR / "autil/autil/FileUtil.h"
        self.assertTrue(header.is_file())
        declared = set(re.findall(r"static (?:bool|std::string) (\w+)\(", header.read_text()))
        used = set()
        for path in VENDOR.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix in {".h", ".cpp", ".cc"}:
                source = path.read_text()
                self.assertIsNone(re.search(r"fslib/|fslib::", source), str(path))
                used.update(re.findall(r"\bFileUtil::(\w+)\(", source))
            if path.name in {"BUILD", "BUILD.bazel"}:
                self.assertNotIn("vendor/filesystem", path.read_text(), str(path))
                self.assertNotIn("fslib", path.read_text(), str(path))
        self.assertEqual(declared, used, "FileUtil must contain only methods used by vendored callers")

    def test_vip_client_has_no_sdk_or_easy_dependency(self):
        folder = VENDOR / "network/gig/multi_call/subscribe"
        interface = (folder / "VipSubscriptionClient.h").read_text()
        self.assertIn("virtual bool init(", interface)
        self.assertIn("virtual bool queryAllIp(", interface)
        self.assertEqual(2, len(re.findall(r"\)\s*=\s*0;", interface)))
        self.assertIn("setVipClientFactory", (folder / "SubscribeServiceFactory.h").read_text())
        for module in ["easy", "vipserver"]:
            self.assertFalse((VENDOR / "third_party" / module).exists())
        forbidden = re.compile(r"@vipserver\b|//[^\s\"]*(?:/easy:|/vipserver:)|VipClientApi|middleware::vipclient|#include\s*[<\"](?:vipclient|iphost|option)\.h[>\"]")
        for path in VENDOR.rglob("*"):
            if path.is_file() and (path.name in {"BUILD", "BUILD.bazel"} or path.suffix in {".h", ".cpp", ".cc", ".bzl"}):
                self.assertIsNone(forbidden.search(path.read_text()), str(path))


if __name__ == "__main__":
    unittest.main()
