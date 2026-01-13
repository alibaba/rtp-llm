import time
import unittest

from rtp_llm.server.host_service import VipServerWrapper


from pytest import mark
@mark.A10
@mark.cuda
@mark.gpu
class TestVipServerClient(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_choose_host(self):
        domain = "127.0.0.1:26001,127.0.0.1:26041,127.0.0.1:26081,127.0.0.1:26121,127.0.0.2:26001,127.0.0.2:26041,127.0.0.2:26081,127.0.0.2:26121,127.0.0.3:26001,127.0.0.3:26041,127.0.0.3:26081,127.0.0.3:26121,127.0.0.4:26001,127.0.0.4:26041,127.0.0.4:26081,127.0.0.4:26121"
        domain_list = domain.split(",")
        client = VipServerWrapper(domain, True)
        print("Client created:", client, flush=True)
        time.sleep(3)  # Ensure the client is initialized
        for _ in range(100):
            host = client.get_host()
            host_str = f"{host.ip}:{host.port}"
            assert host_str in domain_list, f"Host {host_str} not in domain list"


if __name__ == "__main__":
    unittest.main()
