import importlib.util
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from uuid import uuid4

from rtp_llm.models_py.pluggable.control import verify_store_protocol


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "requires a real Torch TCPStore"
)
class StoreProtocolTest(unittest.TestCase):
    def setUp(self):
        from torch.distributed import TCPStore

        # A fresh real TCP server and two independent clients, with no GPU group.
        self.server = TCPStore("127.0.0.1", 0, None, True, timedelta(seconds=3))
        self.clients = [
            TCPStore("127.0.0.1", self.server.port, None, False, timedelta(seconds=3))
            for _ in range(2)
        ]
        self.namespace = "test-" + uuid4().hex

    def verify(self, rank, digest="a" * 64, timeout=1):
        return verify_store_protocol(
            self.clients[rank],
            namespace=self.namespace,
            rank=rank,
            ranks=(0, 1),
            digest=digest,
            timeout_s=timeout,
        )

    def test_agreement_and_duplicate_generation(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.verify, rank) for rank in range(2)]
            self.assertEqual([future.result(3) for future in futures], [True, True])
        with self.assertRaisesRegex(RuntimeError, "namespace already used"):
            self.verify(0)

    def test_mismatch_fails_on_both_ranks(self):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(self.verify, rank, str(rank) * 64) for rank in range(2)
            ]
            for future in futures:
                with self.assertRaisesRegex(RuntimeError, "protocol disagreement"):
                    future.result(3)

    def test_missing_peer_times_out(self):
        with self.assertRaises(RuntimeError):
            self.verify(0, timeout=0.1)


if __name__ == "__main__":
    unittest.main()
