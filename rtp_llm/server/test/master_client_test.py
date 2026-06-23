import unittest
from types import SimpleNamespace

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.server.master_client import FlexlbResponse, MasterClient


class FakeHostService:
    master_vip = SimpleNamespace(domain="")

    def get_master_addr(self):
        return "127.0.0.1:7002"

    def get_slave_addr(self):
        return None


class CapturingMasterClient(MasterClient):
    def __init__(self):
        super().__init__(host_service=FakeHostService())
        self.payloads = []

    async def _send_schedule_request(
        self, addr, payload, generate_timeout_ms, request_id
    ):
        self.payloads.append(payload)
        return FlexlbResponse.ok_with_result(
            {"server_status": [], "cache_local": 0, "code": 200}
        )


class FakeGenerateInput:
    def __init__(self, chat_id):
        self.generate_config = GenerateConfig(chat_id=chat_id)
        self.prompt_length = 3


def make_generate_input(chat_id):
    return FakeGenerateInput(chat_id)


class MasterClientPayloadTest(unittest.IsolatedAsyncioTestCase):
    async def test_includes_non_empty_chat_id_in_schedule_payload(self):
        client = CapturingMasterClient()

        await client.get_backend_role_addrs(
            block_cache_keys=[11, 22],
            input=make_generate_input("chat-a"),
            request_id=12345,
        )

        self.assertEqual("chat-a", client.payloads[0]["chat_id"])

    async def test_omits_empty_chat_id_from_schedule_payload(self):
        client = CapturingMasterClient()

        await client.get_backend_role_addrs(
            block_cache_keys=[11, 22],
            input=make_generate_input(""),
            request_id=12345,
        )

        self.assertNotIn("chat_id", client.payloads[0])


if __name__ == "__main__":
    unittest.main()
