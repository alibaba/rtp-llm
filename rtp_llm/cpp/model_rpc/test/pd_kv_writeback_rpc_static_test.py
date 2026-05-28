import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


class PdKvWritebackRpcStaticTest(unittest.TestCase):
    def test_writeback_rpc_proto_and_routing_are_wired(self):
        proto = (
            REPO_ROOT / "rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto"
        ).read_text()
        prefill_h = (REPO_ROOT / "rtp_llm/cpp/model_rpc/PrefillRpcServer.h").read_text()
        prefill_cc = (
            REPO_ROOT / "rtp_llm/cpp/model_rpc/PrefillRpcServer.cc"
        ).read_text()
        decode_ctx_h = (
            REPO_ROOT / "rtp_llm/cpp/model_rpc/DecodeGenerateContext.h"
        ).read_text()
        decode_cc = (REPO_ROOT / "rtp_llm/cpp/model_rpc/DecodeRpcServer.cc").read_text()
        remote_impl_h = (
            REPO_ROOT / "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"
        ).read_text()

        self.assertIn("repeated string peer_grpc_addrs = 12;", proto)
        self.assertIn("message PdKvWritebackRequestPB", proto)
        self.assertIn("message PdKvWritebackResponsePB", proto)
        self.assertIn(
            "rpc PdKvWriteback(PdKvWritebackRequestPB) returns (PdKvWritebackResponsePB);",
            proto,
        )

        self.assertIn("std::vector<std::string> peer_grpc_addrs", decode_ctx_h)
        self.assertIn("allocate_request.peer_grpc_addrs()", decode_cc)

        self.assertIn("alloc_request.add_peer_grpc_addrs", prefill_cc)
        self.assertIn("resource_.grpc_workers", prefill_cc)
        self.assertIn("PdKvWriteback(", prefill_h)
        self.assertIn("PrefillRpcServer::PdKvWriteback", prefill_cc)

        self.assertIn("PdKvWriteback(", remote_impl_h)
        self.assertIn("prefill_server_->PdKvWriteback", remote_impl_h)


if __name__ == "__main__":
    unittest.main()
