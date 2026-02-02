"""
Unit tests for worker_status.py models, focusing on ScheduleMeta type conversion.
Tests the validate_role method that converts string to RoleType enum.
"""

import unittest

from rtp_llm.config.generate_config import RoleType
from rtp_llm.server.worker_status import ScheduleMeta, ServerStatus


class TestScheduleMetaRoleConversion(unittest.TestCase):
    """Test ScheduleMeta model validation with string role input."""

    def setUp(self):
        """Set up test data for ScheduleMeta."""
        self.valid_schedule_meta = {
            "server_status": [
                {
                    "role": "PREFILL",
                    "server_ip": "127.0.0.1",
                    "http_port": 8000,
                    "grpc_port": 9000,
                },
                {
                    "role": "DECODE",
                    "server_ip": "127.0.0.1",
                    "http_port": 8001,
                    "grpc_port": 9001,
                },
            ],
            "cache_local": 0,
            "inter_request_id": 12345,
            "code": 200,
        }

    def test_schedule_meta_with_string_roles(self):
        """Test ScheduleMeta validation with string role names (from Master Server JSON)."""
        schedule_meta = ScheduleMeta.model_validate(self.valid_schedule_meta)

        # Verify the schedule meta was created successfully
        self.assertEqual(len(schedule_meta.server_status), 2)
        self.assertEqual(schedule_meta.cache_local, 0)
        self.assertEqual(schedule_meta.inter_request_id, 12345)
        self.assertEqual(schedule_meta.code, 200)

        # Verify roles were converted to RoleType enum
        self.assertIsInstance(schedule_meta.server_status[0].role, RoleType)
        self.assertEqual(schedule_meta.server_status[0].role, RoleType.PREFILL)
        self.assertEqual(schedule_meta.server_status[1].role, RoleType.DECODE)

    def test_schedule_meta_with_all_role_types(self):
        """Test ScheduleMeta validation with all role type strings."""
        role_types = ["PDFUSION", "PREFILL", "DECODE", "VIT", "FRONTEND"]

        for role_type_str in role_types:
            test_data = {
                "server_status": [
                    {
                        "role": role_type_str,
                        "server_ip": "127.0.0.1",
                        "http_port": 8000,
                        "grpc_port": 9000,
                    }
                ],
                "cache_local": 0,
                "inter_request_id": 12345,
                "code": 200,
            }

            schedule_meta = ScheduleMeta.model_validate(test_data)
            expected_role = getattr(RoleType, role_type_str)
            self.assertEqual(schedule_meta.server_status[0].role, expected_role)

    def test_server_status_with_debug_info(self):
        """Test ServerStatus with optional debug_info field."""
        test_data = {
            "server_status": [
                {
                    "role": "PREFILL",
                    "server_ip": "127.0.0.1",
                    "http_port": 8000,
                    "grpc_port": 9000,
                    "debug_info": {
                        "running_batch_size": 10,
                        "queue_size": 5,
                        "waiting_time_ms": 100,
                        "available_kv_cache_len": 1000,
                        "estimate_ttft_ms": 50,
                        "estimate_tpot_ms": 20,
                        "hit_cache_len": 500,
                    },
                }
            ],
            "cache_local": 0,
            "inter_request_id": 12345,
            "code": 200,
        }

        schedule_meta = ScheduleMeta.model_validate(test_data)
        self.assertIsNotNone(schedule_meta.server_status[0].debug_info)
        self.assertEqual(
            schedule_meta.server_status[0].debug_info.running_batch_size, 10
        )

    def test_schedule_meta_serialization(self):
        """Test ScheduleMeta serialization to dict and JSON."""
        schedule_meta = ScheduleMeta.model_validate(self.valid_schedule_meta)

        # Serialize to dict
        meta_dict = schedule_meta.model_dump()
        self.assertIn("server_status", meta_dict)
        self.assertEqual(len(meta_dict["server_status"]), 2)

        # Verify role is preserved in serialization
        self.assertEqual(meta_dict["server_status"][0]["role"], RoleType.PREFILL)

    def test_schedule_meta_with_empty_server_status(self):
        """Test ScheduleMeta with empty server_status list."""
        test_data = {
            "server_status": [],
            "cache_local": 0,
            "inter_request_id": 12345,
            "code": 200,
        }

        schedule_meta = ScheduleMeta.model_validate(test_data)
        self.assertEqual(len(schedule_meta.server_status), 0)


if __name__ == "__main__":
    unittest.main()
