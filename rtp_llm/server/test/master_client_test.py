import unittest

from rtp_llm.config.generate_config import RoleAddr, RoleType
from rtp_llm.server.master_client import FlexlbResponse, MasterClient


class _FakeGenerateConfig:
    pass


class _FakeInput:
    prompt_length = 17
    request_id = 123
    generate_config = _FakeGenerateConfig()


class _FakeHostService:
    class _MasterVip:
        domain = "master"

    master_vip = _MasterVip()

    def get_master_addr(self):
        return "master:1234"

    def get_slave_addr(self):
        return None


class MasterClientExcludedWorkersTest(unittest.IsolatedAsyncioTestCase):
    async def test_excluded_role_addrs_are_sent_as_host_wildcards(self):
        client = MasterClient(
            host_service=_FakeHostService(),
            master_config=None,
        )
        captured_payload = {}

        async def fake_send_schedule_request(
            _addr,
            payload,
            _generate_timeout_ms,
            _request_id,
            _request_headers,
        ):
            captured_payload.update(payload)
            return FlexlbResponse.connection_failed_response()

        client._send_schedule_request = fake_send_schedule_request
        excluded = [
            RoleAddr(
                role=RoleType.DECODE,
                ip="10.0.0.1",
                http_port=26650,
                grpc_port=26651,
            ),
            RoleAddr(
                role=RoleType.DECODE,
                ip="10.0.0.1",
                http_port=26660,
                grpc_port=26661,
            ),
            RoleAddr(
                role=RoleType.PREFILL,
                ip="10.0.0.2",
                http_port=25850,
                grpc_port=25851,
            ),
        ]

        await client.get_backend_role_addrs(
            block_cache_keys=[],
            cache_key_block_size=256,
            input=_FakeInput(),
            request_id=123,
            excluded_role_addrs=excluded,
        )

        self.assertEqual(
            captured_payload["excluded_workers"],
            [
                {"role": "DECODE", "server_ip": "10.0.0.1", "http_port": 0},
                {"role": "PREFILL", "server_ip": "10.0.0.2", "http_port": 0},
            ],
        )


if __name__ == "__main__":
    unittest.main()
