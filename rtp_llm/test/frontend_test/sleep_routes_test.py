"""HTTP sleep routes and backend control-address discovery contracts."""

import unittest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rtp_llm.frontend.sleep_routes import register_sleep_routes
from rtp_llm.frontend.worker_address_utils import (
    SLEEP_CONTROL_ADDRESSES_ENV,
    SLEEP_INFER_CONTROL_ADDRESSES_ENV,
    get_control_addrs_from_env,
    get_control_addrs_from_world_info,
    get_dp_addrs_from_world_info,
    infer_control_addrs_from_gang_metadata,
)

SLEEP_STATUS_OK: Dict[str, Any] = {
    "sleep_mode_enabled": True,
    "effective": True,
    "supported_levels": [1],
    "supported_modes": ["wait", "abort"],
    "disabled_reason": "",
    "state": "SLEEPING",
    "sleep_epoch": "1",
    "kv_memory_state": "PAUSED",
    "device_kv_cache_valid": False,
    "active_request_count": "0",
    "active_cache_transfer_count": "0",
    "gpu_resource_state": "RELEASED",
    "last_error": "",
}


def build_test_client(grpc_post_request: AsyncMock) -> TestClient:
    grpc_client = MagicMock()
    grpc_client.post_request = grpc_post_request
    app = FastAPI()
    register_sleep_routes(app, grpc_client)
    return TestClient(app)


class FakeFfnDisaggregateConfig:
    def __init__(self):
        self.enable_ffn_disaggregate = False
        self.attention_tp_size = 1
        self.attention_dp_size = 1

    def to_string(self) -> str:
        return "FakeFfnDisaggregateConfig"


class FakeParallelismConfig:
    def __init__(self):
        self.tp_size = 1
        self.world_rank = 0
        self.world_size = 1
        self.local_world_size = 1
        self.ffn_disaggregate_config = FakeFfnDisaggregateConfig()


class FakeServerConfig:
    def __init__(self):
        self.start_port = 20000
        self.worker_info_port_num = 8


class FakeDistributeConfig:
    def __init__(self, gang_config_string="", distribute_config_file=""):
        self.gang_config_string = gang_config_string
        self.distribute_config_file = distribute_config_file


class FakeWorkerInfo:
    def __init__(
        self,
        ip: str,
        local_rank: int,
        world_rank: int,
        name: str,
        server_port: int,
        worker_info_port_num: int,
    ):
        self.ip = ip
        self.local_rank = local_rank
        self.world_rank = world_rank
        self.name = name
        self.server_port = server_port
        self.worker_info_port_num = worker_info_port_num

    @property
    def rpc_server_port(self) -> int:
        return self.server_port + self.local_rank * self.worker_info_port_num + 1


class FakeWorldInfo:
    def __init__(self, members, master, self_worker, num_nodes, initialized):
        self.members = members
        self.master = master
        self.self = self_worker
        self.num_nodes = num_nodes
        self.initialized = initialized


class SleepRoutesTest(unittest.TestCase):

    def test_sleep_success(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with client:
            response = client.post(
                "/sleep",
                json={"level": 1, "mode": "wait", "timeout_ms": 1000, "reason": "test"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        post_request.assert_awaited_once_with(
            "sleep", {"level": 1, "mode": "wait", "timeout_ms": 1000, "reason": "test"}
        )

    def test_sleep_empty_body(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep")
        self.assertEqual(response.status_code, 200)
        post_request.assert_awaited_once_with("sleep", {})

    def test_sleep_invalid_mode_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"mode": "whatever"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_sleep_invalid_level_type_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"level": "bad"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_sleep_level_zero_passes_to_backend_and_maps_unimplemented(self):
        post_request = AsyncMock(
            return_value={
                "error": "sleep level=0 state-preserving sleep is defined but not implemented",
                "grpc_status": "UNIMPLEMENTED",
            }
        )
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"level": 0})
        self.assertEqual(response.status_code, 501)
        self.assertIn("level=0", response.json()["error"])
        post_request.assert_awaited_once_with("sleep", {"level": 0})

    def test_sleep_invalid_tags_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"tags": "kv_cache"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_sleep_invalid_tag_element_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"tags": ["kv_cache", ""]})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_sleep_partial_tags_rejected_without_backend_call(self):
        post_request = AsyncMock()
        with build_test_client(post_request) as client:
            for tags in (["weights"], ["kv_cache"], ["unknown"]):
                with self.subTest(tags=tags):
                    response = client.post("/sleep", json={"tags": tags})
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("unsupported", response.json()["error"])
        post_request.assert_not_awaited()

    def test_sleep_null_tags_are_treated_as_empty_list(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"tags": None})
        self.assertEqual(response.status_code, 200)
        post_request.assert_awaited_once_with("sleep", {"tags": None})

    def test_sleep_phase_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"phase": "prepare"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_sleep_prepare_only_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={"prepare_only": True})
        self.assertEqual(response.status_code, 400)
        self.assertIn("prepare_only", response.json()["error"])
        post_request.assert_not_awaited()

    def test_sleep_conflict_maps_to_409(self):
        post_request = AsyncMock(
            return_value={
                "error": "sleep rejected in state WAKING_UP",
                "grpc_status": "FAILED_PRECONDITION",
            }
        )
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={})
        self.assertEqual(response.status_code, 409)
        self.assertIn("error", response.json())

    def test_sleep_disabled_maps_to_501(self):
        post_request = AsyncMock(
            return_value={
                "error": "sleep mode is disabled",
                "grpc_status": "UNIMPLEMENTED",
                "sleep_mode_enabled": False,
                "effective": False,
            }
        )
        client = build_test_client(post_request)
        with client:
            response = client.post("/sleep", json={})
        self.assertEqual(response.status_code, 501)
        self.assertFalse(response.json()["effective"])

    def test_wake_up_success(self):
        post_request = AsyncMock(return_value={"status": "ok"})
        client = build_test_client(post_request)
        with client:
            response = client.post("/wake_up")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        post_request.assert_awaited_once_with("wake_up", {})

    def test_wake_up_phase_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/wake_up", json={"phase": "prepare"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())
        post_request.assert_not_awaited()

    def test_wake_up_commit_only_rejected_without_backend_call(self):
        post_request = AsyncMock()
        client = build_test_client(post_request)
        with client:
            response = client.post("/wake_up", json={"commit_only": True})
        self.assertEqual(response.status_code, 400)
        self.assertIn("commit_only", response.json()["error"])
        post_request.assert_not_awaited()

    def test_wake_up_backend_error_maps_to_500(self):
        post_request = AsyncMock(return_value={"error": "backend unreachable"})
        client = build_test_client(post_request)
        with client:
            response = client.post("/wake_up")
        self.assertEqual(response.status_code, 500)

    def test_sleep_status_schema_passthrough(self):
        post_request = AsyncMock(return_value=dict(SLEEP_STATUS_OK))
        client = build_test_client(post_request)
        with client:
            response = client.get("/sleep_status")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        for key in SLEEP_STATUS_OK:
            self.assertIn(key, body)
        post_request.assert_awaited_once_with("sleep_status", {})

    def test_is_sleeping_schema_passthrough(self):
        post_request = AsyncMock(
            return_value={
                "is_sleeping": True,
                "sleep_mode_enabled": True,
                "effective": True,
                "supported_levels": [1],
                "supported_modes": ["wait", "abort"],
                "state": "SLEEPING",
                "disabled_reason": "",
            }
        )
        client = build_test_client(post_request)
        with client:
            response = client.get("/is_sleeping")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["is_sleeping"])
        post_request.assert_awaited_once_with("is_sleeping", {})

    def test_sleep_status_backend_error_maps_to_500(self):
        post_request = AsyncMock(return_value={"error": "no backend"})
        client = build_test_client(post_request)
        with client:
            response = client.get("/sleep_status")
        self.assertEqual(response.status_code, 500)


class SleepControlAddressTest(unittest.TestCase):

    def test_control_addresses_env_override_accepts_csv_and_dedupes(self):
        with patch.dict(
            "os.environ",
            {
                SLEEP_CONTROL_ADDRESSES_ENV: "10.0.0.1:20001,10.0.0.2:20009;10.0.0.1:20001"
            },
            clear=False,
        ):
            self.assertEqual(
                get_control_addrs_from_env(),
                ["10.0.0.1:20001", "10.0.0.2:20009"],
            )

    def test_control_addresses_env_override_accepts_json_list(self):
        with patch.dict(
            "os.environ",
            {SLEEP_CONTROL_ADDRESSES_ENV: '["10.0.0.1:20001", "10.0.0.2:20009"]'},
            clear=False,
        ):
            self.assertEqual(
                get_control_addrs_from_env(),
                ["10.0.0.1:20001", "10.0.0.2:20009"],
            )

    def test_control_addresses_include_all_ranks_but_dp_addresses_do_not(self):
        members = [
            FakeWorkerInfo(
                ip="127.0.0.1",
                local_rank=rank,
                world_rank=rank,
                name=f"rank_{rank}",
                server_port=20000,
                worker_info_port_num=8,
            )
            for rank in range(4)
        ]
        world_info = FakeWorldInfo(
            members=members,
            master=members[0],
            self_worker=members[0],
            num_nodes=1,
            initialized=True,
        )
        pc = FakeParallelismConfig()
        pc.tp_size = 2

        dp_addresses = get_dp_addrs_from_world_info(world_info, pc)
        control_addresses = get_control_addrs_from_world_info(world_info)

        self.assertEqual(dp_addresses, ["127.0.0.1:20001", "127.0.0.1:20017"])
        self.assertEqual(
            control_addresses,
            [
                "127.0.0.1:20001",
                "127.0.0.1:20009",
                "127.0.0.1:20017",
                "127.0.0.1:20025",
            ],
        )

    def test_ffn_disaggregate_control_addresses_still_include_all_ranks(self):
        members = [
            FakeWorkerInfo(
                ip="127.0.0.1",
                local_rank=rank,
                world_rank=rank,
                name=f"rank_{rank}",
                server_port=20000,
                worker_info_port_num=8,
            )
            for rank in range(4)
        ]
        world_info = FakeWorldInfo(
            members=members,
            master=members[0],
            self_worker=members[0],
            num_nodes=1,
            initialized=True,
        )
        pc = FakeParallelismConfig()
        pc.tp_size = 1
        pc.ffn_disaggregate_config.enable_ffn_disaggregate = True
        pc.ffn_disaggregate_config.attention_tp_size = 1
        pc.ffn_disaggregate_config.attention_dp_size = 2

        dp_addresses = get_dp_addrs_from_world_info(world_info, pc)
        control_addresses = get_control_addrs_from_world_info(world_info)

        self.assertEqual(dp_addresses, ["127.0.0.1:20001", "127.0.0.1:20009"])
        self.assertEqual(
            control_addresses,
            [
                "127.0.0.1:20001",
                "127.0.0.1:20009",
                "127.0.0.1:20017",
                "127.0.0.1:20025",
            ],
        )

    def test_infer_control_addresses_from_gang_metadata_is_opt_in(self):
        pc = FakeParallelismConfig()
        pc.world_size = 4
        pc.local_world_size = 2
        gang_config = (
            "name:foo_part0,ip:10.0.0.1,port:20000;"
            "name:foo_part1,ip:10.0.0.2,port:20000"
        )
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                infer_control_addrs_from_gang_metadata(
                    FakeServerConfig(), FakeDistributeConfig(gang_config), pc
                ),
                [],
            )

    def test_infer_control_addresses_from_gang_metadata(self):
        pc = FakeParallelismConfig()
        pc.world_size = 4
        pc.local_world_size = 2
        gang_config = (
            "name:foo_part1,ip:10.0.0.2,port:20000;"
            "name:foo_part0,ip:10.0.0.1,port:20000"
        )
        with patch.dict(
            "os.environ", {SLEEP_INFER_CONTROL_ADDRESSES_ENV: "1"}, clear=True
        ):
            self.assertEqual(
                infer_control_addrs_from_gang_metadata(
                    FakeServerConfig(), FakeDistributeConfig(gang_config), pc
                ),
                [
                    "10.0.0.1:20001",
                    "10.0.0.1:20009",
                    "10.0.0.2:20001",
                    "10.0.0.2:20009",
                ],
            )


if __name__ == "__main__":
    unittest.main()
