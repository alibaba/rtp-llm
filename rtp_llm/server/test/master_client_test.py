import unittest

from rtp_llm.cpp.model_rpc.proto.flexlb_schedule_service_pb2 import (
    FlexlbScheduleResponsePB,
    FlexlbServerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import GenerateInputPB
from rtp_llm.server.master_client import MasterClient


class _FakeMasterConfig:
    master_default_timeout_ms = 3600000


class _FakeHostService:
    """Minimal HostService mock with configurable master/slave addrs."""

    def __init__(self, master_addr="master:1234", slave_addr=None):
        self._master_addr = master_addr
        self._slave_addr = slave_addr

    def get_master_addr(self):
        return self._master_addr

    def get_slave_addr(self):
        return self._slave_addr


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
    """MasterClient that records calls and returns canned (response, is_conn_phase) tuples."""

    def __init__(self, host_service, responses):
        super().__init__(
            host_service=host_service,
            master_config=_FakeMasterConfig(),
        )
        # Each entry is a (proto_or_None, is_connection_phase_failure) tuple.
        self._responses = list(responses)
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
        if self._responses:
            return self._responses.pop(0)
        return (None, False)


def _make_schedule_response():
    """Build a simple success proto with one prefill server."""
    return FlexlbScheduleResponsePB(
        success=True,
        code=200,
        enqueued_by_master=True,
        server_status=[
            FlexlbServerStatusPB(
                role="PREFILL",
                server_ip="10.0.0.7",
                http_port=8080,
                grpc_port=9000,
            )
        ],
    )


class MasterClientTest(unittest.IsolatedAsyncioTestCase):
    # ------------------------------------------------------------------ #
    # Success path
    # ------------------------------------------------------------------ #
    async def test_schedule_request_contains_batch_fields_and_pb(self):
        schedule_response = _make_schedule_response()
        client = _CaptureMasterClient(
            host_service=_FakeHostService(),
            responses=[(schedule_response, False)],
        )
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

    # ------------------------------------------------------------------ #
    # No master address
    # ------------------------------------------------------------------ #
    async def test_no_master_addr_returns_connection_failed(self):
        client = _CaptureMasterClient(
            host_service=_FakeHostService(master_addr=None),
            responses=[],
        )

        response = await client.get_backend_role_addrs(
            block_cache_keys=[],
            cache_key_block_size=1024,
            input=_FakeInput(),
            request_id=1,
        )

        self.assertTrue(response.connection_failed)
        self.assertFalse(response.is_ok)
        self.assertEqual(len(client.calls), 0)

    # ------------------------------------------------------------------ #
    # Connection-phase failure + slave available -> retry slave
    # ------------------------------------------------------------------ #
    async def test_connection_failure_with_slave_retries_slave(self):
        schedule_response = _make_schedule_response()
        client = _CaptureMasterClient(
            host_service=_FakeHostService(slave_addr="slave:1234"),
            responses=[
                (None, True),  # master fails (connection phase)
                (schedule_response, False),  # slave succeeds
            ],
        )

        response = await client.get_backend_role_addrs(
            block_cache_keys=[],
            cache_key_block_size=1024,
            input=_FakeInput(),
            request_id=100,
        )

        self.assertTrue(response.is_ok)
        self.assertEqual(response.role_addrs[0].ip, "10.0.0.7")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["addr"], "master:1234")
        self.assertEqual(client.calls[1]["addr"], "slave:1234")

    # ------------------------------------------------------------------ #
    # Connection-phase failure + no slave -> directly connection_failed
    # ------------------------------------------------------------------ #
    async def test_connection_failure_no_slave_returns_connection_failed(self):
        client = _CaptureMasterClient(
            host_service=_FakeHostService(slave_addr=None),
            responses=[(None, True)],  # master fails (connection phase)
        )

        response = await client.get_backend_role_addrs(
            block_cache_keys=[],
            cache_key_block_size=1024,
            input=_FakeInput(),
            request_id=100,
        )

        self.assertTrue(response.connection_failed)
        self.assertFalse(response.is_ok)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["addr"], "master:1234")

    # ------------------------------------------------------------------ #
    # Connection-phase failure + slave also fails -> connection_failed
    # ------------------------------------------------------------------ #
    async def test_connection_failure_slave_also_fails(self):
        client = _CaptureMasterClient(
            host_service=_FakeHostService(slave_addr="slave:1234"),
            responses=[
                (None, True),  # master fails (connection phase)
                (None, True),  # slave also fails (connection phase)
            ],
        )

        response = await client.get_backend_role_addrs(
            block_cache_keys=[],
            cache_key_block_size=1024,
            input=_FakeInput(),
            request_id=100,
        )

        self.assertTrue(response.connection_failed)
        self.assertFalse(response.is_ok)
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["addr"], "master:1234")
        self.assertEqual(client.calls[1]["addr"], "slave:1234")

    # ------------------------------------------------------------------ #
    # Stream-level timeout (not connection-phase) -> no slave retry
    # ------------------------------------------------------------------ #
    async def test_stream_timeout_does_not_retry_slave(self):
        client = _CaptureMasterClient(
            host_service=_FakeHostService(slave_addr="slave:1234"),
            responses=[(None, False)],  # stream timeout (not connection phase)
        )

        response = await client.get_backend_role_addrs(
            block_cache_keys=[],
            cache_key_block_size=1024,
            input=_FakeInput(),
            request_id=100,
        )

        self.assertTrue(response.connection_failed)
        self.assertFalse(response.is_ok)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["addr"], "master:1234")


if __name__ == "__main__":
    unittest.main()
