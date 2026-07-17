import unittest

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    ROLE_TYPE_PREFILL,
    FlexlbScheduleResponsePB,
    FlexlbServerStatusPB,
    GenerateInputPB,
)
from rtp_llm.server.master_client import MasterClient


class _FakeMasterConfig:
    master_default_timeout_ms = 3600000


class _FakeHostService:
    def get_master_addr(self):
        return "master:1234"

    def get_slave_addr(self):
        return "slave:1234"


class _FakeHostServiceWithSlave(_FakeHostService):
    def get_slave_addr(self):
        return "slave:1234"


class _FakeGenerateConfig:
    max_new_tokens = 17
    num_beams = 2
    force_disable_sp_run = True
    ttft_timeout_ms = 3000
    timeout_ms = -1


class _FakeInput:
    prompt_length = 5
    headers = {"x-api-key": "api-key"}

    def __init__(self):
        self.generate_config = _FakeGenerateConfig()


class _CaptureMasterClient(MasterClient):
    def __init__(self, response):
        super().__init__(
            host_service=_FakeHostService(),
            master_config=_FakeMasterConfig(),
        )
        self.response = response
        self.calls = []

    async def _send_schedule_request(self, addr, request_pb, timeout_s, request_id):
        self.calls.append(
            {
                "addr": addr,
                "request_pb": request_pb,
                "timeout_s": timeout_s,
                "request_id": request_id,
            }
        )
        return self.response


class MasterClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_schedule_request_contains_batch_fields_and_pb(self):
        schedule_response = FlexlbScheduleResponsePB(
            success=True,
            code=200,
            enqueued_by_master=True,
            server_status=[
                FlexlbServerStatusPB(
                    role=ROLE_TYPE_PREFILL,
                    server_ip="10.0.0.7",
                    http_port=8080,
                    grpc_port=9000,
                )
            ],
        )
        client = _CaptureMasterClient(schedule_response)
        input_pb = GenerateInputPB()

        response = await client.get_backend_role_addrs(
            block_cache_keys=[1, 2, 3],
            cache_key_block_size=1024,
            input=_FakeInput(),
            request_id=99,
            input_pb=input_pb,
        )

        self.assertTrue(response.is_ok)
        self.assertTrue(response.enqueued_by_master)
        self.assertEqual(response.role_addrs[0].ip, "10.0.0.7")

        self.assertEqual(len(client.calls), 1)
        call = client.calls[0]
        request_pb = call["request_pb"]
        self.assertEqual(call["addr"], "master:1234")
        self.assertEqual(call["timeout_s"], 3.0)
        self.assertEqual(call["request_id"], 99)
        self.assertEqual(list(request_pb.block_cache_keys), [1, 2, 3])
        self.assertEqual(request_pb.cache_key_block_size, 1024)
        self.assertEqual(request_pb.seq_len, 5)
        self.assertEqual(request_pb.generate_timeout, 3000)
        self.assertEqual(request_pb.request_id, 99)
        self.assertEqual(request_pb.max_new_tokens, 17)
        self.assertEqual(request_pb.num_beams, 2)
        self.assertTrue(request_pb.force_disable_sp_run)
        self.assertEqual(request_pb.api_key, "api-key")
        self.assertEqual(request_pb.generate_input, input_pb)

    async def test_transport_failure_does_not_retry_slave(self):
        client = _CaptureMasterClient(response=None)

        response = await client.get_backend_role_addrs(
            block_cache_keys=[],
            cache_key_block_size=1024,
            input=_FakeInput(),
            request_id=100,
        )

        self.assertTrue(response.connection_failed)
        self.assertFalse(response.is_ok)
        self.assertEqual([call["addr"] for call in client.calls], ["master:1234"])


if __name__ == "__main__":
    unittest.main()
