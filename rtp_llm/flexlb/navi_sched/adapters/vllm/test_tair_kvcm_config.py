import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location("tair_query_config_under_test", Path(__file__).with_name("VLLMTairKvcmConfig.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

FullAttentionSpec = type("FullAttentionSpec", (), {})
MambaSpec = type("MambaSpec", (), {})


class TypedConfig(SimpleNamespace):
    def valid(self):
        self.validation_calls = getattr(self, "validation_calls", 0) + 1
        return True


class TairKvcmConfigTest(unittest.TestCase):
    def setup_source(self):
        remote = {
            "manager_endpoint": "spectrum://v-actual:6382", "instance_id": "actual_native_4096",
            "hash_block_size": 4096, "node_count": 4, "discovery_refresh_interval_ms": 30000,
            "groups": [{"group_id": group, "spec_type": kind, "object_size_bytes": size,
                        "location_spec_names": [f"real_group_{group}_N{rank}" for rank in range(4)]}
                       for group, kind, size in [(1, "MambaSpec", 1024), (0, "FullAttentionSpec", 512)]],
        }
        owner = SimpleNamespace(
            peer_info={"kvcm_status": {"mode": "report-event-and-cache-transfer"}},
            rpc=SimpleNamespace(refresh_peer_info=Mock(return_value={"kvcm_status": {"mode": "report-event-and-cache-transfer"}, "tair_kvcm_query_config": remote})),
        )
        other = SimpleNamespace(rpc=SimpleNamespace(refresh_peer_info=Mock()))
        source = SimpleNamespace(
            _client_context=SimpleNamespace(client=SimpleNamespace(_clients=[owner, other])),
            hash_block_size=4096, _group_block_bytes={0: (512, []), 1: (1024, [])},
            kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=FullAttentionSpec()),
                                                            SimpleNamespace(kv_cache_spec=MambaSpec())]),
        )
        policy = SimpleNamespace(block_size=4096, group_roles=("full", "tail"), group_block_scales=(1, 2))
        navi = SimpleNamespace(NaviSchedTairKvcmConfig=TypedConfig, NaviSchedTairKvcmGroupConfig=SimpleNamespace)
        return source, policy, navi, remote, owner, other

    def test_registered_capability_refresh_replaces_stale_connection_metadata(self):
        source, policy, navi, remote, owner, other = self.setup_source()
        owner.peer_info.clear()
        result = module.load_tair_kvcm_config(source, policy, navi)
        self.assertIsNotNone(result)
        self.assertEqual(remote["instance_id"], result.instance_id)
        owner.rpc.refresh_peer_info.assert_called_once_with()
        other.rpc.refresh_peer_info.assert_not_called()

    def test_fresh_disabled_capability_overrides_stale_enabled_connection(self):
        source, policy, navi, remote, owner, other = self.setup_source()
        owner.rpc.refresh_peer_info.return_value = {"kvcm_status": {"mode": "disabled"}}
        self.assertIsNone(module.load_tair_kvcm_config(source, policy, navi))
        owner.rpc.refresh_peer_info.assert_called_once_with()
        other.rpc.refresh_peer_info.assert_not_called()

    def test_actual_native_layout_becomes_typed_handshake_config_once(self):
        source, policy, navi, remote, owner, other = self.setup_source()
        result = module.load_tair_kvcm_config(source, policy, navi)
        self.assertEqual(result.manager_endpoint, remote["manager_endpoint"])
        self.assertEqual(result.instance_id, remote["instance_id"])
        self.assertEqual(result.discovery_refresh_interval_us, 30000000)
        self.assertEqual([group.group_id for group in result.groups], [0, 1])
        self.assertEqual([group.hash_block_scale for group in result.groups], [1, 2])
        self.assertEqual([group.tail for group in result.groups], [False, True])
        self.assertEqual(result.groups[1].location_spec_names, [f"real_group_1_N{rank}" for rank in range(4)])
        self.assertEqual(result.validation_calls, 1)
        owner.rpc.refresh_peer_info.assert_called_once_with()
        other.rpc.refresh_peer_info.assert_not_called()

    def test_disabled_or_report_only_follows_current_peer_metadata(self):
        for mode in ["disabled", "report-event-only", None]:
            source, policy, navi, remote, owner, other = self.setup_source()
            owner.peer_info["kvcm_status"]["mode"] = mode
            owner.rpc.refresh_peer_info.return_value["kvcm_status"]["mode"] = mode
            self.assertIsNone(module.load_tair_kvcm_config(source, policy, navi))
            owner.rpc.refresh_peer_info.assert_called_once_with()
            other.rpc.refresh_peer_info.assert_not_called()

    def test_missing_client_does_not_query(self):
        source, policy, navi, remote, owner, other = self.setup_source()
        source._client_context = None
        self.assertIsNone(module.load_tair_kvcm_config(source, policy, navi))
        owner.rpc.refresh_peer_info.assert_not_called()

    def test_metadata_failure_does_not_fail_engine_initialization(self):
        source, policy, navi, remote, owner, other = self.setup_source()
        owner.rpc.refresh_peer_info.side_effect = TimeoutError("isolated test")
        with patch.object(module.logger, "exception") as error:
            self.assertIsNone(module.load_tair_kvcm_config(source, policy, navi))
            error.assert_called_once()

    def test_inconsistent_native_layout_is_rejected(self):
        source, policy, navi, remote, owner, other = self.setup_source()
        original = copy.deepcopy(remote)
        for mutate in [lambda r: r.update(hash_block_size=512),
                       lambda r: r["groups"].pop(),
                       lambda r: r["groups"][0].update(spec_type="FullAttentionSpec"),
                       lambda r: r["groups"][1].update(object_size_bytes=999)]:
            remote.clear()
            remote.update(copy.deepcopy(original))
            mutate(remote)
            with self.assertRaises(ValueError):
                module._capture_tair_kvcm_config(source, policy, navi)

    def test_missing_registered_layout_returns_unknown(self):
        source, policy, navi, remote, owner, other = self.setup_source()
        owner.rpc.refresh_peer_info.return_value = {"kvcm_status": {"mode": "report-event-and-cache-transfer"}}
        with patch.object(module.logger, "warning") as warning:
            self.assertIsNone(module.load_tair_kvcm_config(source, policy, navi))
            warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
