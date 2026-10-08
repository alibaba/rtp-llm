#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include <memory>

#include <gtest/gtest.h>
#include "torch/all.h"

#define private public
#define protected public
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#undef private
#undef protected
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm::test {

namespace {

std::shared_ptr<GenerateStream> makeStream(const std::vector<int>& input_ids) {
    ModelConfig                    model_config;
    RuntimeConfig                  runtime_config;
    ResourceContext                resource_context;
    std::shared_ptr<GenerateInput> query = std::make_shared<GenerateInput>();

    model_config.max_seq_len                 = 4096;
    model_config.vocab_size                  = 32000;
    model_config.special_tokens.eos_token_id = 151643;

    query->input_ids       = torch::tensor(std::vector<int32_t>(input_ids.begin(), input_ids.end()), torch::kInt32);
    query->generate_config = std::make_shared<GenerateConfig>();

    return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
}

GenerateOutputsPB makeOutputsWithDecodeReuse(int total, int local, int remote, int memory, int disk = 0) {
    GenerateOutputsPB outputs_pb;
    outputs_pb.mutable_flatten_output()->add_finished(false);
    auto* aux_info = outputs_pb.mutable_flatten_output()->add_aux_info();
    aux_info->set_total_reuse_len(total);
    aux_info->set_local_reuse_len(local);
    aux_info->set_remote_reuse_len(remote);
    aux_info->set_memory_reuse_len(memory);
    aux_info->set_disk_reuse_len(disk);
    aux_info->set_step_output_len(1);
    return outputs_pb;
}

}  // namespace

TEST(DecodeRpcServerTest, ShouldUsePDSeparationIgnoresUniqueKeyPresence) {
    GenerateInputPB request;
    auto*           config = request.mutable_generate_config();
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);

    EXPECT_TRUE(checkPDSupport(request).supported);

    config->set_unique_key("user-cache-key");
    EXPECT_TRUE(checkPDSupport(request).supported);
}

TEST(DecodeRpcServerTest, DecodeEntranceHandoffUsesInternalKeyAndPreservesBusinessKey) {
    GenerateInputPB request;
    auto*           config = request.mutable_generate_config();
    config->set_unique_key("shared-business-key");

    auto first  = makeDecodeEntranceUniqueKey("127.0.0.1", 1, 100);
    auto second = makeDecodeEntranceUniqueKey("127.0.0.1", 2, 100);

    EXPECT_NE(first, second);
    EXPECT_NE(first, request.generate_config().unique_key());
    EXPECT_NE(second, request.generate_config().unique_key());

    auto first_handoff_request  = makeDecodeEntranceHandoffRequest(request, first);
    auto second_handoff_request = makeDecodeEntranceHandoffRequest(request, second);

    EXPECT_EQ(request.generate_config().unique_key(), "shared-business-key");
    EXPECT_EQ(first_handoff_request.generate_config().unique_key(), first);
    EXPECT_EQ(second_handoff_request.generate_config().unique_key(), second);
    EXPECT_NE(first_handoff_request.generate_config().unique_key(),
              second_handoff_request.generate_config().unique_key());
}

TEST(DecodeRpcServerTest, ParsePrefillDpAddrSupportsIpv4HostAndBracketIpv6) {
    std::string ip;
    uint32_t    port = 0;

    ASSERT_TRUE(DecodeRpcServer::parsePrefillDpAddr("127.0.0.1:9000", &ip, &port).ok());
    EXPECT_EQ(ip, "127.0.0.1");
    EXPECT_EQ(port, 9000);

    ASSERT_TRUE(DecodeRpcServer::parsePrefillDpAddr("prefill-0.service:9001", &ip, &port).ok());
    EXPECT_EQ(ip, "prefill-0.service");
    EXPECT_EQ(port, 9001);

    ASSERT_TRUE(DecodeRpcServer::parsePrefillDpAddr("[::1]:9002", &ip, &port).ok());
    EXPECT_EQ(ip, "[::1]");
    EXPECT_EQ(port, 9002);

    ASSERT_TRUE(DecodeRpcServer::parsePrefillDpAddr("fe80::1:9003", &ip, &port).ok());
    EXPECT_EQ(ip, "[fe80::1]");
    EXPECT_EQ(port, 9003);
}

TEST(DecodeRpcServerTest, ParsePrefillDpAddrRejectsMalformedAddressOrPort) {
    std::string ip;
    uint32_t    port = 0;

    EXPECT_FALSE(DecodeRpcServer::parsePrefillDpAddr("", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServer::parsePrefillDpAddr("127.0.0.1", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServer::parsePrefillDpAddr("fe80::1", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServer::parsePrefillDpAddr("[::1]9000", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServer::parsePrefillDpAddr("127.0.0.1:0", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServer::parsePrefillDpAddr("127.0.0.1:65536", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServer::parsePrefillDpAddr("127.0.0.1:not-a-port", &ip, &port).ok());
}

TEST(DecodeRpcServerTest, ShouldUsePDSeparationRejectsNonPdRequests) {
    GenerateInputPB request;
    auto*           config = request.mutable_generate_config();
    config->set_max_new_tokens(1);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);

    EXPECT_FALSE(checkPDSupport(request).supported);

    config->set_max_new_tokens(8);
    config->set_num_beams(2);
    EXPECT_FALSE(checkPDSupport(request).supported);
}

TEST(DecodeRpcServerTest, UpdateAuxInfoUsesPrefillReuseAsTopLevelAndPreservesDecodeReuse) {
    DecodeRpcServer server;
    auto            stream = makeStream({11, 12, 13});
    auto outputs_pb = makeOutputsWithDecodeReuse(/*total=*/7, /*local=*/3, /*remote=*/4, /*memory=*/1, /*disk=*/2);

    stream->setPrefillReuseLength(
        /*total=*/128, /*local=*/32, /*remote=*/96, /*memory=*/8, /*disk=*/4, /*independent_pools=*/false);

    server.updateAuxInfo(outputs_pb, stream);

    ASSERT_EQ(outputs_pb.flatten_output().aux_info_size(), 1);
    const auto& aux_info = outputs_pb.flatten_output().aux_info(0);
    EXPECT_TRUE(aux_info.pd_sep());

    EXPECT_EQ(aux_info.total_reuse_len(), 128);
    EXPECT_EQ(aux_info.local_reuse_len(), 32);
    EXPECT_EQ(aux_info.remote_reuse_len(), 96);
    EXPECT_EQ(aux_info.memory_reuse_len(), 8);
    EXPECT_EQ(aux_info.disk_reuse_len(), 4);

    EXPECT_EQ(aux_info.prefill_total_reuse_len(), 128);
    EXPECT_EQ(aux_info.prefill_local_reuse_len(), 32);
    EXPECT_EQ(aux_info.prefill_remote_reuse_len(), 96);
    EXPECT_EQ(aux_info.prefill_memory_reuse_len(), 8);
    EXPECT_EQ(aux_info.prefill_disk_reuse_len(), 4);

    EXPECT_EQ(aux_info.decode_total_reuse_len(), 7);
    EXPECT_EQ(aux_info.decode_local_reuse_len(), 3);
    EXPECT_EQ(aux_info.decode_remote_reuse_len(), 4);
    EXPECT_EQ(aux_info.decode_memory_reuse_len(), 1);
    EXPECT_EQ(aux_info.decode_disk_reuse_len(), 2);
}

TEST(DecodeRpcServerTest, IndependentPrefillPoolsExposeTheLongerReusePhase) {
    for (bool independent : {false, true}) {
        for (int decode_total : {64, 128, 256}) {
            DecodeRpcServer server;
            auto            stream = makeStream({11, 12, 13});
            stream->setPrefillReuseLength(128, 32, 96, 8, 4, independent);
            auto outputs = makeOutputsWithDecodeReuse(decode_total, 16, decode_total - 16, 2, 1);
            server.updateAuxInfo(outputs, stream);
            const auto& aux        = outputs.flatten_output().aux_info(0);
            const bool  use_decode = independent && decode_total > 128;
            EXPECT_EQ(aux.total_reuse_len(), use_decode ? decode_total : 128);
            EXPECT_EQ(aux.local_reuse_len(), use_decode ? 16 : 32);
            EXPECT_EQ(aux.remote_reuse_len(), use_decode ? decode_total - 16 : 96);
            EXPECT_EQ(aux.memory_reuse_len(), use_decode ? 2 : 8);
            EXPECT_EQ(aux.disk_reuse_len(), use_decode ? 1 : 4);
            EXPECT_EQ(aux.prefill_total_reuse_len(), 128);
            EXPECT_EQ(aux.decode_total_reuse_len(), decode_total);
        }
    }
}

}  // namespace rtp_llm::test

namespace rtp_llm {

TEST(DecodeRpcServerTest, CancelBeforeStreamBindingStopsTheFutureStream) {
    auto             meta = std::make_shared<RpcServerRuntimeMeta>();
    PDCancelRegistry registry(meta);
    GenerateInputPB  request;
    request.set_request_id(201);
    PDCancelRegistry::Handle handle;
    ASSERT_TRUE(registry.admit(request, "handoff201", "", currentTimeMs() + 1000, handle).ok());
    EXPECT_EQ(registry.cancel(201, {ErrorCode::PRIORITY_PREEMPTED, "preempted"}), CANCEL_STATUS_ACCEPTED);
    auto stream = test::makeStream({1, 2, 3});
    registry.attach(handle, stream);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::PRIORITY_PREEMPTED);
    registry.finishLocal(handle);
    // An error alone is not proof that the scheduler and resources are finished.
    EXPECT_FALSE(registry.complete(handle));
}

TEST(DecodeRpcServerTest, InternalCancelRetainsCleanupProofBeyondWorkerStatusDelta) {
    DecodeRpcServer server;
    server.meta_            = std::make_shared<RpcServerRuntimeMeta>();
    server.cancel_registry_ = std::make_unique<PDCancelRegistry>(server.meta_);
    GenerateInputPB input;
    input.set_request_id(202);
    PDCancelRegistry::Handle handle;
    ASSERT_TRUE(server.cancel_registry_->admit(input, "handoff202", "", currentTimeMs() + 1000, handle).ok());
    CancelRequestPB request;
    request.set_request_id(202);
    CancelResponsePB    response;
    grpc::ServerContext context;
    ASSERT_TRUE(server.Cancel(&context, &request, &response).ok());
    EXPECT_EQ(response.status(), CANCEL_STATUS_ACCEPTED);
    server.cancel_registry_->finishLocal(handle);
    ASSERT_TRUE(server.cancel_registry_->complete(handle));
    ASSERT_TRUE(server.Cancel(&context, &request, &response).ok());
    EXPECT_EQ(response.status(), CANCEL_STATUS_NOT_FOUND);
}

}  // namespace rtp_llm

namespace rtp_llm {
namespace {
class LocalPrefillCancelService: public RpcService::Service {
public:
    int          calls{0};
    int64_t      request_id{0};
    int64_t      cancel_error_code{0};
    grpc::Status Cancel(grpc::ServerContext*, const CancelRequestPB* request, CancelResponsePB* response) override {
        ++calls;
        request_id        = request->request_id();
        cancel_error_code = request->cancel_error_code();
        EXPECT_TRUE(request->prefill_address().empty());
        response->set_status(CANCEL_STATUS_TOMBSTONED);
        return grpc::Status::OK;
    }
};
}  // namespace

TEST(DecodeRpcServerTest, CancelBeforeDecodeAdmissionForwardsToSelectedPrefillAndFencesLateRequest) {
    for (auto code : {ErrorCode::PRIORITY_PREEMPTED, ErrorCode::CANCELLED, ErrorCode::GENERATE_TIMEOUT}) {

        LocalPrefillCancelService prefill;
        grpc::ServerBuilder       builder;
        int                       port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&prefill);
        auto rpc_server = builder.BuildAndStart();
        ASSERT_TRUE(rpc_server);
        DecodeRpcServer decode;
        decode.meta_            = std::make_shared<RpcServerRuntimeMeta>();
        decode.cancel_registry_ = std::make_unique<PDCancelRegistry>(decode.meta_);
        CancelRequestPB request;
        request.set_request_id(203);
        request.set_cancel_error_code(static_cast<int64_t>(code));
        request.set_prefill_address("127.0.0.1:" + std::to_string(port));
        CancelResponsePB    response;
        grpc::ServerContext context;
        ASSERT_TRUE(decode.Cancel(&context, &request, &response).ok());
        EXPECT_EQ(response.status(), CANCEL_STATUS_ACCEPTED);
        auto handle = decode.cancel_registry_->find(203);
        ASSERT_TRUE(handle);
        EXPECT_FALSE(handle->terminal.load());
        decode.cancelCleanupTick();
        EXPECT_EQ(prefill.calls, 1);
        EXPECT_EQ(prefill.request_id, 203);
        EXPECT_EQ(prefill.cancel_error_code, code);
        EXPECT_TRUE(handle->terminal.load());
        const auto finished = decode.meta_->getEngineScheduleInfo(0);
        ASSERT_EQ(finished.finished_task_info_list.size(), 1);
        EXPECT_EQ(finished.finished_task_info_list.front().error_code, code);
        GenerateInputPB late;
        late.set_request_id(203);
        PDCancelRegistry::Handle late_handle;
        EXPECT_EQ(
            decode.cancel_registry_->admit(late, "late", request.prefill_address(), currentTimeMs() + 1000, late_handle)
                .code(),
            code);
        rpc_server->Shutdown();
    }
}

TEST(DecodeRpcServerTest, UnreachablePrefillDoesNotPublishCancellationComplete) {
    DecodeRpcServer decode;
    decode.meta_            = std::make_shared<RpcServerRuntimeMeta>();
    decode.cancel_registry_ = std::make_unique<PDCancelRegistry>(decode.meta_);
    CancelRequestPB request;
    request.set_request_id(204);
    request.set_prefill_address("127.0.0.1:1");
    CancelResponsePB    response;
    grpc::ServerContext context;
    ASSERT_TRUE(decode.Cancel(&context, &request, &response).ok());
    decode.cancelCleanupTick();
    EXPECT_FALSE(decode.cancel_registry_->find(204)->terminal.load());
    EXPECT_TRUE(decode.meta_->getEngineScheduleInfo(0).finished_task_info_list.empty());
}
}  // namespace rtp_llm
