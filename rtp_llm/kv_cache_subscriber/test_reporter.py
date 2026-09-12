from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from rtp_llm.kv_cache_subscriber.models import CacheDiff
from rtp_llm.kv_cache_subscriber.reporter import HttpKvcmReporter
from rtp_llm.kv_cache_subscriber.test_utils import make_config


class _FakeManager:
    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.register_requests: list[dict[str, object]] = []
        self.event_requests: list[dict[str, object]] = []

    async def start(self) -> None:
        self.started = True

    async def register_instance(self, data: dict[str, object]) -> dict[str, object]:
        self.register_requests.append(data)
        return {"header": {"status": {"code": "OK"}}}

    async def report_event(self, data: dict[str, object]) -> dict[str, object]:
        self.event_requests.append(data)
        return {"header": {"status": {"code": "OK"}}}

    async def close(self) -> None:
        self.closed = True


class HttpKvcmReporterTest(unittest.IsolatedAsyncioTestCase):
    async def test_start_registers_rtp_location_and_model_shape(self) -> None:
        engine_config = {
            "model_name": "deepseek-v3",
            "dtype": "bfloat16",
            "use_mla": True,
            "tensor_parallel_size": 8,
            "data_parallel_size": 2,
            "pipeline_parallel_size": 1,
        }
        manager = _FakeManager()
        with patch.dict(
            os.environ,
            {"DS_LLM_ENGINE_CONFIG": json.dumps(engine_config)},
        ):
            reporter = HttpKvcmReporter(make_config(), manager)

        await reporter.start(block_size=16)

        self.assertTrue(manager.started)
        request = manager.register_requests[0]
        self.assertEqual(request["instance_id"], "instance-a")
        self.assertEqual(request["block_size"], 16)
        self.assertEqual(
            request["location_spec_infos"],
            [{"name": "rtp_llm_16", "size": 16}],
        )
        self.assertEqual(
            request["model_deployment"],
            {
                "model_name": "deepseek-v3",
                "dtype": "bfloat16",
                "use_mla": True,
                "tp_size": 8,
                "dp_size": 2,
                "lora_name": "",
                "pp_size": 1,
                "extra": "",
                "user_data": "",
            },
        )

    async def test_start_uses_explicit_rtp_metadata_without_engine_config(self) -> None:
        manager = _FakeManager()
        config = make_config(
            model_name="Qwen2-0.5B",
            model_dtype="bfloat16",
            tp_size=2,
            dp_size=1,
            pp_size=1,
        )
        with patch.dict(os.environ, {"DS_LLM_ENGINE_CONFIG": "{}"}):
            reporter = HttpKvcmReporter(config, manager)

        await reporter.start(block_size=64)

        self.assertEqual(
            manager.register_requests[0]["model_deployment"],
            {
                "model_name": "Qwen2-0.5B",
                "dtype": "bfloat16",
                "use_mla": False,
                "tp_size": 2,
                "dp_size": 1,
                "lora_name": "",
                "pp_size": 1,
                "extra": "",
                "user_data": "",
            },
        )

    async def test_diff_is_mapped_and_split_into_bounded_requests(self) -> None:
        manager = _FakeManager()
        config = make_config(kvcm_report_batch_size=2)
        reporter = HttpKvcmReporter(config, manager)
        await reporter.start(block_size=16)

        await reporter.report_diff(
            CacheDiff(added=(11, 12), removed=(13,)),
            block_size=16,
        )

        self.assertEqual(len(manager.event_requests), 2)
        events = [
            event
            for request in manager.event_requests
            for event in request["events"]
        ]
        self.assertEqual(
            [event["event_type"] for event in events],
            ["EVENT_BLOCK_ADD", "EVENT_BLOCK_ADD", "EVENT_BLOCK_DELETE"],
        )
        self.assertEqual(
            events[0]["block_add"],
            {
                "block_key": "11",
                "medium": "hbm",
                "specs": [
                    {
                        "name": "rtp_llm_16",
                        "uri": "rtp-llm://10.0.0.8:8088/hbm",
                    }
                ],
            },
        )
        self.assertEqual(
            events[2]["block_delete"],
            {"block_key": "13", "medium": "hbm"},
        )
        for request in manager.event_requests:
            self.assertEqual(request["instance_id"], "instance-a")
            self.assertEqual(request["host_ip_port"], "10.0.0.8:8088")
            self.assertEqual(request["storage_type"], "ST_EVENT_REPORT")

    async def test_lifecycle_events_have_expected_wire_shape(self) -> None:
        manager = _FakeManager()
        reporter = HttpKvcmReporter(make_config(), manager)
        await reporter.start(block_size=16)

        await reporter.register_node()
        await reporter.report_heartbeat()
        await reporter.report_host_down()

        events = [request["events"][0] for request in manager.event_requests]
        self.assertEqual(
            events,
            [
                {
                    "event_type": "EVENT_NODE_REGISTER",
                    "node_register": {"mediums": ["hbm"]},
                },
                {
                    "event_type": "EVENT_HEARTBEAT",
                    "heartbeat": {"system_status": {}},
                },
                {"event_type": "EVENT_HOST_DOWN", "host_down": {}},
            ],
        )

    async def test_empty_diff_does_not_send_report_request(self) -> None:
        manager = _FakeManager()
        reporter = HttpKvcmReporter(make_config(), manager)
        await reporter.start(block_size=16)

        await reporter.report_diff(CacheDiff(added=(), removed=()), block_size=16)

        self.assertEqual(manager.event_requests, [])

    async def test_report_before_start_is_rejected(self) -> None:
        reporter = HttpKvcmReporter(make_config(), _FakeManager())

        with self.assertRaisesRegex(RuntimeError, "has not been started"):
            await reporter.register_node()


if __name__ == "__main__":
    unittest.main()
