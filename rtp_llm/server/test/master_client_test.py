import base64
import unittest

from rtp_llm.server.master_client import FlexlbResponse, MasterClient


class _FakeMasterConfig:
    master_max_connect_pool_size = 4
    master_session_timeout_s = 1
    master_default_timeout_ms = 3600000


class _FakeHostService:
    def get_master_addr(self):
        return "master:1234"

    def get_slave_addr(self):
        return None


class _FakeGenerateConfig:
    max_new_tokens = 17
    num_beams = 2
    force_disable_sp_run = True
    ttft_timeout_ms = 3000
    timeout_ms = -1
    traffic_reject_priority = 12


class _FakeInput:
    prompt_length = 5
    headers = {"x-request-id": "req-1"}

    def __init__(self):
        self.generate_config = _FakeGenerateConfig()


class _CaptureMasterClient(MasterClient):
    def __init__(self):
        super().__init__(
            host_service=_FakeHostService(),
            master_config=_FakeMasterConfig(),
        )
        self.calls = []

    async def _send_schedule_request(
        self, addr, payload, generate_timeout_ms, request_id, request_headers=None
    ):
        self.calls.append(
            {
                "addr": addr,
                "payload": payload,
                "generate_timeout_ms": generate_timeout_ms,
                "request_id": request_id,
                "request_headers": request_headers,
            }
        )
        return FlexlbResponse.ok_with_result(
            {
                "code": 200,
                "server_status": [
                    {
                        "role": "PREFILL",
                        "server_ip": "10.0.0.7",
                        "http_port": 8080,
                        "grpc_port": 9000,
                    }
                ],
                "enqueued_by_master": True,
            }
        )


class MasterClientBatchPayloadTest(unittest.IsolatedAsyncioTestCase):
    async def test_schedule_payload_contains_batch_fields_and_pb(self):
        client = _CaptureMasterClient()

        response = await client.get_backend_role_addrs(
            block_cache_keys=[1, 2, 3],
            input=_FakeInput(),
            request_id=99,
            input_pb_bytes=b"serialized-input",
        )

        self.assertTrue(response.is_ok)
        self.assertTrue(response.enqueued_by_master)
        self.assertEqual(response.role_addrs[0].ip, "10.0.0.7")

        call = client.calls[0]
        payload = call["payload"]
        self.assertEqual(call["addr"], "master:1234")
        self.assertEqual(call["generate_timeout_ms"], 3000)
        self.assertEqual(call["request_id"], 99)
        self.assertEqual(call["request_headers"], {"x-request-id": "req-1"})
        self.assertEqual(payload["block_cache_keys"], [1, 2, 3])
        self.assertEqual(payload["seq_len"], 5)
        self.assertEqual(payload["request_priority"], 12)
        self.assertEqual(payload["generate_timeout"], 3000)
        self.assertEqual(payload["request_id"], 99)
        self.assertEqual(payload["max_new_tokens"], 17)
        self.assertEqual(payload["num_beams"], 2)
        self.assertTrue(payload["force_disable_sp_run"])
        self.assertEqual(
            payload["generate_input_pb_b64"],
            base64.b64encode(b"serialized-input").decode("ascii"),
        )


if __name__ == "__main__":
    unittest.main()
