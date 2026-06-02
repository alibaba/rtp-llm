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
        decode_h = (REPO_ROOT / "rtp_llm/cpp/model_rpc/DecodeRpcServer.h").read_text()
        decode_cc = (REPO_ROOT / "rtp_llm/cpp/model_rpc/DecodeRpcServer.cc").read_text()
        rpc_util_h = (
            REPO_ROOT / "rtp_llm/cpp/model_rpc/PdKvWritebackRpcUtil.h"
        ).read_text()
        rpc_util_cc = (
            REPO_ROOT / "rtp_llm/cpp/model_rpc/PdKvWritebackRpcUtil.cc"
        ).read_text()
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
        self.assertIn("pdKvWritebackLaunchRequestFromPB", rpc_util_h)
        self.assertIn("pdKvWritebackLaunchRequestFromPB", rpc_util_cc)
        self.assertIn("PdKvWritebackRpcUtil.h", prefill_cc)

        self.assertIn("PdKvWriteback(", remote_impl_h)
        self.assertIn("prefill_server_->PdKvWriteback", remote_impl_h)
        self.assertIn("DecodeRpcServer::PdKvWritebackSend", decode_cc)
        self.assertIn("sendOnDecode", decode_cc)
        self.assertIn("PdKvWritebackSend", decode_h)
        self.assertIn("decode_server_->PdKvWritebackSend", remote_impl_h)

    def test_decode_stream_writeback_uses_stable_non_empty_unique_key(self):
        query_converter_cc = (
            REPO_ROOT / "rtp_llm/cpp/model_rpc/QueryConverter.cc"
        ).read_text()
        decode_cc = (REPO_ROOT / "rtp_llm/cpp/model_rpc/DecodeRpcServer.cc").read_text()

        self.assertIn("generate_config->unique_key", query_converter_cc)
        self.assertIn("config_proto->unique_key()", query_converter_cc)
        self.assertIn("input->generate_config->unique_key.empty()", decode_cc)
        self.assertIn("input->generate_config->unique_key = decode_context.request_key", decode_cc)

    def test_writeback_p2p_transfer_plans_keep_manifest_request_key(self):
        manifest_cc = (
            REPO_ROOT / "rtp_llm/cpp/cache/writeback/PdKvWritebackManifest.cc"
        ).read_text()
        manager_cc = (
            REPO_ROOT / "rtp_llm/cpp/cache/writeback/PdKvWritebackManager.cc"
        ).read_text()

        self.assertIn("snapshot.request_key.empty()", manifest_cc)
        self.assertIn(
            'snapshot.request_key.empty() ? "pd_kv_writeback_" + std::to_string(snapshot.request_id) : snapshot.request_key',
            manifest_cc,
        )
        self.assertGreaterEqual(
            manager_cc.count("plan.request_key              = request.manifest.request_key"),
            2,
        )
        self.assertIn("request.manifest.request_key", manager_cc)


if __name__ == "__main__":
    unittest.main()
