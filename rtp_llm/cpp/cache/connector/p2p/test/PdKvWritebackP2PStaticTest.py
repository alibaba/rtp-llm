import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[6]


class PdKvWritebackP2PStaticTest(unittest.TestCase):
    def test_writeback_directional_worker_apis_are_wired(self):
        transfer_h = (
            REPO_ROOT / "rtp_llm/cpp/cache/writeback/PdKvWritebackTransfer.h"
        ).read_text()
        worker_h = (
            REPO_ROOT / "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.h"
        ).read_text()
        worker_cc = (
            REPO_ROOT / "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.cc"
        ).read_text()
        prefill_h = (
            REPO_ROOT / "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerPrefill.h"
        ).read_text()
        prefill_cc = (
            REPO_ROOT / "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerPrefill.cc"
        ).read_text()
        decode_h = (
            REPO_ROOT / "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerDecode.h"
        ).read_text()
        decode_cc = (
            REPO_ROOT / "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerDecode.cc"
        ).read_text()

        self.assertIn("struct PdKvWritebackTransferPlan", transfer_h)
        self.assertIn("decode_group_block_ids", transfer_h)
        self.assertIn("prefill_group_block_ids", transfer_h)

        self.assertIn("startDecodeToPrefillWriteback", worker_h)
        self.assertIn("sendDecodeToPrefillWriteback", worker_h)
        self.assertIn("receiveDecodeToPrefillWriteback", worker_h)
        self.assertIn("sendDecodeToPrefillWriteback(plan)", worker_cc)
        self.assertIn("receiveDecodeToPrefillWriteback(plan)", worker_cc)

        self.assertIn("sendDecodeToPrefillWriteback", prefill_h)
        self.assertIn("PdKvWritebackBlockSide::DecodeSource", prefill_cc)
        self.assertIn("plan.prefill_transfer_servers", prefill_cc)

        self.assertIn("receiveDecodeToPrefillWriteback", decode_h)
        self.assertIn("PdKvWritebackBlockSide::PrefillDestination", decode_cc)
        self.assertIn("plan.remote_tp_size", decode_cc)


if __name__ == "__main__":
    unittest.main()
