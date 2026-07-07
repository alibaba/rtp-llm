#include <memory>

#include <gtest/gtest.h>
#include "torch/all.h"

#define private public
#define protected public
#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
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

class RecordingWriter: public grpc::internal::WriterInterface<GenerateOutputsPB> {
public:
    explicit RecordingWriter(std::shared_ptr<GenerateStream> stream): stream_(std::move(stream)) {}

    bool Write(const GenerateOutputsPB& output, grpc::WriteOptions) override {
        outputs.push_back(output);
        if (outputs.size() == 1 && stream_) {
            stream_->generate_status_->status.store(StreamState::FINISHED, std::memory_order_release);
        }
        return true;
    }

    std::vector<GenerateOutputsPB> outputs;

private:
    std::shared_ptr<GenerateStream> stream_;
};

GenerateOutputsPB makeOutputsWithDecodeReuse(int total, int local, int remote, int memory) {
    GenerateOutputsPB outputs_pb;
    outputs_pb.mutable_flatten_output()->add_finished(false);
    auto* aux_info = outputs_pb.mutable_flatten_output()->add_aux_info();
    aux_info->set_total_reuse_len(total);
    aux_info->set_local_reuse_len(local);
    aux_info->set_remote_reuse_len(remote);
    aux_info->set_memory_reuse_len(memory);
    aux_info->set_step_output_len(1);
    return outputs_pb;
}

}  // namespace

TEST(DecodeRpcServerNew2Test, DecodeEntranceRequiresPrefillIgnoresUniqueKeyPresence) {
    GenerateInputPB request;
    auto*           config = request.mutable_generate_config();
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);

    EXPECT_TRUE(decodeEntranceRequiresPrefill(request));

    config->set_unique_key("user-cache-key");
    EXPECT_TRUE(decodeEntranceRequiresPrefill(request));
}

TEST(DecodeRpcServerNew2Test, DecodeEntranceHandoffUsesInternalKeyAndPreservesBusinessKey) {
    GenerateInputPB request;
    auto*           config = request.mutable_generate_config();
    config->set_unique_key("shared-business-key");

    auto first  = buildDecodeEntranceKeys(request, "127.0.0.1", 1, 100);
    auto second = buildDecodeEntranceKeys(request, "127.0.0.1", 2, 100);

    EXPECT_EQ(first.business_unique_key, "shared-business-key");
    EXPECT_EQ(second.business_unique_key, "shared-business-key");
    EXPECT_NE(first.handoff_unique_key, second.handoff_unique_key);
    EXPECT_NE(first.handoff_unique_key, first.business_unique_key);
    EXPECT_NE(second.handoff_unique_key, second.business_unique_key);

    auto first_handoff_request  = makeDecodeEntranceHandoffRequest(request, first.handoff_unique_key);
    auto second_handoff_request = makeDecodeEntranceHandoffRequest(request, second.handoff_unique_key);

    EXPECT_EQ(request.generate_config().unique_key(), "shared-business-key");
    EXPECT_EQ(first_handoff_request.generate_config().unique_key(), first.handoff_unique_key);
    EXPECT_EQ(second_handoff_request.generate_config().unique_key(), second.handoff_unique_key);
    EXPECT_NE(first_handoff_request.generate_config().unique_key(), second_handoff_request.generate_config().unique_key());
}

TEST(DecodeRpcServerNew2Test, SelectDecodeEntranceDpIndexUsesHandoffSequence) {
    EXPECT_EQ(selectDecodeEntranceDpIndex(3, 0), 0);
    EXPECT_EQ(selectDecodeEntranceDpIndex(3, 1), 1);
    EXPECT_EQ(selectDecodeEntranceDpIndex(3, 2), 2);
    EXPECT_EQ(selectDecodeEntranceDpIndex(3, 3), 0);
    EXPECT_EQ(selectDecodeEntranceDpIndex(0, 10), 0);
}

TEST(DecodeRpcServerNew2Test, ParsePrefillDpAddrSupportsIpv4HostAndBracketIpv6) {
    std::string ip;
    uint32_t    port = 0;

    ASSERT_TRUE(DecodeRpcServerNew2::parsePrefillDpAddr("127.0.0.1:9000", &ip, &port).ok());
    EXPECT_EQ(ip, "127.0.0.1");
    EXPECT_EQ(port, 9000);

    ASSERT_TRUE(DecodeRpcServerNew2::parsePrefillDpAddr("prefill-0.service:9001", &ip, &port).ok());
    EXPECT_EQ(ip, "prefill-0.service");
    EXPECT_EQ(port, 9001);

    ASSERT_TRUE(DecodeRpcServerNew2::parsePrefillDpAddr("[::1]:9002", &ip, &port).ok());
    EXPECT_EQ(ip, "[::1]");
    EXPECT_EQ(port, 9002);

    ASSERT_TRUE(DecodeRpcServerNew2::parsePrefillDpAddr("fe80::1:9003", &ip, &port).ok());
    EXPECT_EQ(ip, "[fe80::1]");
    EXPECT_EQ(port, 9003);
}

TEST(DecodeRpcServerNew2Test, ParsePrefillDpAddrRejectsMalformedAddressOrPort) {
    std::string ip;
    uint32_t    port = 0;

    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("127.0.0.1", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("fe80::1", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("[::1]9000", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("127.0.0.1:0", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("127.0.0.1:65536", &ip, &port).ok());
    EXPECT_FALSE(DecodeRpcServerNew2::parsePrefillDpAddr("127.0.0.1:not-a-port", &ip, &port).ok());
}

TEST(DecodeRpcServerNew2Test, DecodeEntranceRequiresPrefillRejectsNonPdRequests) {
    GenerateInputPB request;
    auto*           config = request.mutable_generate_config();
    config->set_max_new_tokens(1);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(true);

    EXPECT_FALSE(decodeEntranceRequiresPrefill(request));

    config->set_max_new_tokens(8);
    config->set_num_beams(2);
    EXPECT_FALSE(decodeEntranceRequiresPrefill(request));
}

TEST(DecodeRpcServerNew2Test, UpdateAuxInfoUsesPrefillReuseAsTopLevelAndPreservesDecodeReuse) {
    auto stream     = makeStream({11, 12, 13});
    auto outputs_pb = makeOutputsWithDecodeReuse(/*total=*/7, /*local=*/3, /*remote=*/4, /*memory=*/1);

    stream->setPrefillReuseLength(/*total=*/128, /*local=*/32, /*remote=*/96, /*memory=*/8);

    updateDecodeAuxInfo(outputs_pb, stream, nullptr);

    ASSERT_EQ(outputs_pb.flatten_output().aux_info_size(), 1);
    const auto& aux_info = outputs_pb.flatten_output().aux_info(0);
    EXPECT_TRUE(aux_info.pd_sep());

    EXPECT_EQ(aux_info.total_reuse_len(), 128);
    EXPECT_EQ(aux_info.local_reuse_len(), 32);
    EXPECT_EQ(aux_info.remote_reuse_len(), 96);
    EXPECT_EQ(aux_info.memory_reuse_len(), 8);

    EXPECT_EQ(aux_info.prefill_total_reuse_len(), 128);
    EXPECT_EQ(aux_info.prefill_local_reuse_len(), 32);
    EXPECT_EQ(aux_info.prefill_remote_reuse_len(), 96);
    EXPECT_EQ(aux_info.prefill_memory_reuse_len(), 8);

    EXPECT_EQ(aux_info.decode_total_reuse_len(), 7);
    EXPECT_EQ(aux_info.decode_local_reuse_len(), 3);
    EXPECT_EQ(aux_info.decode_remote_reuse_len(), 4);
    EXPECT_EQ(aux_info.decode_memory_reuse_len(), 1);
}

TEST(DecodeRpcServerNew2Test, UpdateAuxInfoFallsBackToPrefillContextSnapshot) {
    auto stream      = makeStream({21, 22, 23});
    auto outputs_pb  = makeOutputsWithDecodeReuse(/*total=*/0, /*local=*/0, /*remote=*/0, /*memory=*/0);
    auto prefill_ctx = std::make_shared<PrefillServerCallerContext>("127.0.0.1", "reuse-key");

    PrefillServerCallerContext::ReuseLensSnapshot snapshot;
    snapshot.total  = 64;
    snapshot.local  = 16;
    snapshot.remote = 48;
    snapshot.memory = 2;
    prefill_ctx->setPrefillReuseLensSnapshotForTest(snapshot);

    updateDecodeAuxInfo(outputs_pb, stream, prefill_ctx);

    ASSERT_EQ(outputs_pb.flatten_output().aux_info_size(), 1);
    const auto& aux_info = outputs_pb.flatten_output().aux_info(0);
    EXPECT_EQ(aux_info.total_reuse_len(), 64);
    EXPECT_EQ(aux_info.local_reuse_len(), 16);
    EXPECT_EQ(aux_info.remote_reuse_len(), 48);
    EXPECT_EQ(aux_info.memory_reuse_len(), 2);

    EXPECT_EQ(aux_info.prefill_total_reuse_len(), 64);
    EXPECT_EQ(aux_info.prefill_local_reuse_len(), 16);
    EXPECT_EQ(aux_info.prefill_remote_reuse_len(), 48);
    EXPECT_EQ(aux_info.prefill_memory_reuse_len(), 2);

    EXPECT_EQ(aux_info.decode_total_reuse_len(), 0);
    EXPECT_EQ(aux_info.decode_local_reuse_len(), 0);
    EXPECT_EQ(aux_info.decode_remote_reuse_len(), 0);
    EXPECT_EQ(aux_info.decode_memory_reuse_len(), 0);
}

// Test that prefill_finished detection correctly identifies when the first token
// triggers termination (e.g., max_new_tokens=1 or first token is EOS).
// This verifies the logic in pollStreamOutputWithPrefill that tracks prefill_finished
// to ensure the finished=true flag is not lost when the first token is a termination token.
TEST(DecodeRpcServerNew2Test, PrefillFirstResponseFinishedDetection) {
    // Simulate a prefill response where finished=true (first token is EOS / max_new_tokens=1)
    GenerateOutputsPB prefill_output_finished;
    prefill_output_finished.mutable_flatten_output()->add_finished(true);
    auto* aux_info = prefill_output_finished.mutable_flatten_output()->add_aux_info();
    aux_info->set_step_output_len(1);

    // Verify the finished flag is set correctly
    ASSERT_EQ(prefill_output_finished.flatten_output().finished_size(), 1);
    EXPECT_TRUE(prefill_output_finished.flatten_output().finished(0));

    // Simulate a normal prefill response where finished=false
    GenerateOutputsPB prefill_output_not_finished;
    prefill_output_not_finished.mutable_flatten_output()->add_finished(false);
    prefill_output_not_finished.mutable_flatten_output()->add_aux_info();

    ASSERT_EQ(prefill_output_not_finished.flatten_output().finished_size(), 1);
    EXPECT_FALSE(prefill_output_not_finished.flatten_output().finished(0));
}

TEST(DecodeRpcServerNew2Test, ConsumePrefillFirstResponseWritesFirstChunkAndClearsFinishedFlag) {
    auto stream      = makeStream({31, 32, 33});
    auto prefill_ctx = std::make_shared<PrefillServerCallerContext>("127.0.0.1:9000", "first-response");

    prefill_ctx->first_response_received_ = true;
    prefill_ctx->response_received_ = true;
    prefill_ctx->first_response_.mutable_flatten_output()->add_finished(true);
    prefill_ctx->first_response_.mutable_flatten_output()->add_aux_info()->set_step_output_len(1);

    bool              prefill_finished      = false;
    int               prefill_finished_size = 0;
    bool              skip_next_decode      = false;
    GenerateOutputsPB client_output;

    EXPECT_TRUE(DecodeRpcServerNew2::consumePrefillFirstResponse(prefill_ctx,
                                                                 stream,
                                                                 /*client_first_chunk_sent=*/false,
                                                                 &prefill_finished,
                                                                 &prefill_finished_size,
                                                                 &skip_next_decode,
                                                                 &client_output));
    EXPECT_TRUE(prefill_finished);
    EXPECT_EQ(prefill_finished_size, 1);
    EXPECT_TRUE(skip_next_decode);
    ASSERT_EQ(client_output.flatten_output().finished_size(), 1);
    EXPECT_FALSE(client_output.flatten_output().finished(0));
    EXPECT_TRUE(prefill_ctx->first_response_consumed_);
}

TEST(DecodeRpcServerNew2Test, ConsumePrefillFirstResponseAfterDecodeFirstChunkStillRecordsTermination) {
    auto stream      = makeStream({41, 42, 43});
    auto prefill_ctx = std::make_shared<PrefillServerCallerContext>("127.0.0.1:9000", "late-first-response");

    prefill_ctx->first_response_received_ = true;
    prefill_ctx->response_received_ = true;
    prefill_ctx->first_response_.mutable_flatten_output()->add_finished(true);
    prefill_ctx->first_response_.mutable_flatten_output()->add_aux_info()->set_step_output_len(1);

    bool              prefill_finished      = false;
    int               prefill_finished_size = 0;
    bool              skip_next_decode      = true;
    GenerateOutputsPB client_output;

    EXPECT_FALSE(DecodeRpcServerNew2::consumePrefillFirstResponse(prefill_ctx,
                                                                  stream,
                                                                  /*client_first_chunk_sent=*/true,
                                                                  &prefill_finished,
                                                                  &prefill_finished_size,
                                                                  &skip_next_decode,
                                                                  &client_output));
    EXPECT_TRUE(prefill_finished);
    EXPECT_EQ(prefill_finished_size, 1);
    EXPECT_FALSE(skip_next_decode);
    EXPECT_TRUE(prefill_ctx->first_response_consumed_);
    EXPECT_EQ(client_output.flatten_output().finished_size(), 0);
}

TEST(DecodeRpcServerNew2Test, PollStreamOutputWithPrefillSendsTerminalFrameWithAuxInfo) {
    auto stream      = makeStream({71, 72, 73});
    auto prefill_ctx = std::make_shared<PrefillServerCallerContext>("127.0.0.1:9000", "first-token-eos");

    stream->generate_status_->status.store(StreamState::RUNNING, std::memory_order_release);
    stream->setPrefillReuseLength(/*total=*/8, /*local=*/2, /*remote=*/6, /*memory=*/1);

    prefill_ctx->first_response_received_ = true;
    prefill_ctx->response_received_ = true;
    prefill_ctx->first_response_.mutable_flatten_output()->add_finished(true);
    auto* prefill_aux = prefill_ctx->first_response_.mutable_flatten_output()->add_aux_info();
    prefill_aux->set_step_output_len(1);

    DecodeRpcServerNew2 server;
    grpc::ServerContext context;
    RecordingWriter writer(stream);

    auto status = server.pollStreamOutputWithPrefill(&context, "first-token-eos", &writer, stream, prefill_ctx);

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(writer.outputs.size(), 2);

    const auto& first = writer.outputs[0].flatten_output();
    ASSERT_EQ(first.finished_size(), 1);
    EXPECT_FALSE(first.finished(0));

    const auto& terminal = writer.outputs[1].flatten_output();
    ASSERT_EQ(terminal.finished_size(), 1);
    EXPECT_TRUE(terminal.finished(0));
    ASSERT_EQ(terminal.aux_info_size(), 1);
    EXPECT_TRUE(terminal.aux_info(0).pd_sep());
    EXPECT_EQ(terminal.aux_info(0).total_reuse_len(), 8);
    EXPECT_EQ(terminal.aux_info(0).local_reuse_len(), 2);
    EXPECT_EQ(terminal.aux_info(0).remote_reuse_len(), 6);
    EXPECT_EQ(terminal.aux_info(0).memory_reuse_len(), 1);
}

TEST(DecodeRpcServerNew2Test, RefreshIdleStreamStateReportsTimeout) {
    auto stream = makeStream({51, 52, 53});
    stream->generateConfig()->timeout_ms = 1;

    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    EXPECT_TRUE(DecodeRpcServerNew2::refreshIdleStreamState(stream));
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::GENERATE_TIMEOUT);
}

TEST(DecodeRpcServerNew2Test, ConsumePrefillFirstResponseSkipsErrorChunk) {
    auto stream      = makeStream({61, 62, 63});
    auto prefill_ctx = std::make_shared<PrefillServerCallerContext>("127.0.0.1:9000", "error-first-response");

    prefill_ctx->first_response_received_ = true;
    prefill_ctx->first_response_.mutable_flatten_output()->add_finished(false);
    prefill_ctx->first_response_.mutable_flatten_output()->add_aux_info()->set_step_output_len(1);
    prefill_ctx->error_info_ = ErrorInfo(ErrorCode::UNKNOWN_ERROR, "prefill chunk failed");

    bool              prefill_finished      = false;
    int               prefill_finished_size = 0;
    bool              skip_next_decode      = false;
    GenerateOutputsPB client_output;

    EXPECT_TRUE(prefill_ctx->failed());
    EXPECT_FALSE(DecodeRpcServerNew2::consumePrefillFirstResponse(prefill_ctx,
                                                                  stream,
                                                                  /*client_first_chunk_sent=*/false,
                                                                  &prefill_finished,
                                                                  &prefill_finished_size,
                                                                  &skip_next_decode,
                                                                  &client_output));
    EXPECT_FALSE(prefill_ctx->first_response_consumed_);
    EXPECT_EQ(client_output.flatten_output().finished_size(), 0);
}

}  // namespace rtp_llm::test
