#include <limits>
#include "autil/Scope.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <thread>
#include <set>
#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PBroadcastClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/PipelineLayoutCodec.h"
#include "rtp_llm/cpp/model_rpc/BatchStreamOutputCollector.h"
#include <memory>

#include <gtest/gtest.h>
#include "torch/all.h"

#define private public
#define protected public
#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"
#undef private
#undef protected
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/model_rpc/RemoteRpcServiceImpl.h"

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

TEST(DecodeRpcServerTest, TimeoutLearnedAfterConstructionKeepsAbsoluteDeadline) {
    kmonitor::MetricsReporterPtr metrics_reporter;
    GenerateContext              context(/*request_id=*/1, /*timeout_ms=*/0, nullptr, metrics_reporter, nullptr);
    context.markRpcHandlingCompleted();

    EXPECT_FALSE(context.request_deadline.has_value());
    EXPECT_FALSE(context.requestDeadlineExceeded());

    context.setRequestTimeoutMs(1000);
    ASSERT_TRUE(context.request_deadline.has_value());
    const auto absolute_deadline = *context.request_deadline;
    EXPECT_GT(absolute_deadline, std::chrono::system_clock::now());
    EXPECT_EQ(context.request_timeout_ms, 1000);

    context.setRequestTimeoutMs(1000);
    EXPECT_EQ(context.request_deadline, absolute_deadline) << "reapplying a timeout must not extend its budget";

    context.setRequestTimeoutMs(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    EXPECT_TRUE(context.requestDeadlineExceeded());

    context.setRequestTimeoutMs(0);
    EXPECT_FALSE(context.request_deadline.has_value());
    EXPECT_FALSE(context.requestDeadlineExceeded());

    context.setRequestTimeoutMs(-1);
    EXPECT_FALSE(context.request_deadline.has_value());
    EXPECT_FALSE(context.requestDeadlineExceeded());
}

TEST(DecodeRpcServerTest, InvalidPrefillOnlyFailsBeforeEngineAccess) {
    LocalRpcServer  server;
    GenerateInputPB request;
    request.set_request_id(1);
    request.add_token_ids(0);
    request.mutable_generate_config()->set_max_new_tokens(1);
    request.mutable_generate_config()->set_prefill_only(true);
    const auto status = server.GenerateStreamCall(nullptr, &request, nullptr);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
    EXPECT_NE(details.error_message().find("max_new_tokens"), std::string::npos);
}

TEST(ModelRpcProtoTest, P2PControlFieldsPreserveReservedAndRoutingNumbers) {
    // The old RemoteGenerate/BroadcastLoad protocol is retired. Its replacement
    // retains explicit reservations instead of reusing removed control fields.
    const auto* start = P2PConnectorStartLoadRequestPB::descriptor();
    ASSERT_NE(start, nullptr);
    EXPECT_TRUE(start->IsReservedNumber(5));
    EXPECT_TRUE(start->IsReservedName("request_timeout_ms"));
    EXPECT_EQ(start->FindFieldByName("workers")->number(), 2);
    EXPECT_EQ(start->FindFieldByName("plan_digest")->number(), 6);
    EXPECT_EQ(start->FindFieldByName("active_route_ids")->number(), 7);
    EXPECT_EQ(start->FindFieldByName("decode_tp_size")->number(), 9);
    EXPECT_EQ(start->FindFieldByName("decode_pipeline_layout")->number(), 10);
    const auto* remote = RemoteOperationRequestPB::descriptor();
    ASSERT_NE(remote, nullptr);
    EXPECT_TRUE(remote->IsReservedNumber(3));
    EXPECT_EQ(remote->FindFieldByName("group_ids"), nullptr);
    EXPECT_EQ(remote->FindFieldByName("block_ids")->number(), 4);
    EXPECT_EQ(remote->FindFieldByName("uris")->number(), 5);
    EXPECT_EQ(remote->FindFieldByName("group_tags")->number(), 6);
}

TEST(ModelRpcProtoTest, RemoteCacheMetadataPreservesGdrStatusWireFormat) {
    RemoteOperationResponsePB response;
    // Existing GDR workers encode TIMEOUT as field 3 with enum value 3.
    ASSERT_TRUE(response.ParseFromString(std::string("\x18\x03", 2)));
    EXPECT_EQ(response.transfer_status(), REMOTE_TRANSFER_STATUS_TIMEOUT);

    response.add_hosts()->set_host_ip_port("host:1");
    response.add_backend_locations()->add_locations()->add_specs()->set_uri("cache_uri");
    std::string serialized;
    ASSERT_TRUE(response.SerializeToString(&serialized));

    RemoteOperationResponsePB parsed;
    ASSERT_TRUE(parsed.ParseFromString(serialized));
    EXPECT_EQ(parsed.transfer_status(), REMOTE_TRANSFER_STATUS_TIMEOUT);
    ASSERT_EQ(parsed.hosts_size(), 1);
    EXPECT_EQ(parsed.hosts(0).host_ip_port(), "host:1");
    ASSERT_EQ(parsed.backend_locations_size(), 1);
    ASSERT_EQ(parsed.backend_locations(0).locations_size(), 1);
    ASSERT_EQ(parsed.backend_locations(0).locations(0).specs_size(), 1);
    EXPECT_EQ(parsed.backend_locations(0).locations(0).specs(0).uri(), "cache_uri");
}

TEST(DecodeRpcServerTest, P2PHandoffRequiresEveryRankIncludingOddTpWidths) {
    // P2P publishes an atomic transfer result, rather than the old RPC's minimum
    // partial-key prefix. Every rank (including an empty-route PP rank) must ack.
    using Broadcast = P2PBroadcastClient::TpBroadcastResult;
    for (size_t count : {3u, 4u, 5u}) {
        std::vector<std::shared_ptr<Broadcast::WorkerRpcContext>> workers;
        for (size_t rank = 0; rank < count; ++rank) {
            auto worker    = std::make_shared<Broadcast::WorkerRpcContext>();
            worker->status = grpc::Status::OK;
            worker->response.mutable_p2p_response()->set_error_code(transErrorCodeToRPC(ErrorCode::NONE_ERROR));
            workers.push_back(worker);
        }
        auto                       broadcast = std::make_shared<Broadcast>(workers);
        P2PBroadcastClient::Result result("handoff", broadcast);
        for (size_t rank = 0; rank + 1 < count; ++rank) {
            broadcast->finishRank(rank, true);
            EXPECT_FALSE(result.done());
            EXPECT_FALSE(result.success());
        }
        broadcast->finishRank(count - 1, true);
        EXPECT_TRUE(result.done());
        EXPECT_TRUE(result.success());
    }
}

TEST(DecodeRpcServerTest, P2PHandoffDoesNotPublishPartialRankSuccess) {
    using Broadcast = P2PBroadcastClient::TpBroadcastResult;
    std::vector<std::shared_ptr<Broadcast::WorkerRpcContext>> workers;
    for (int rank = 0; rank < 2; ++rank) {
        auto worker    = std::make_shared<Broadcast::WorkerRpcContext>();
        worker->status = rank == 0 ? grpc::Status::OK : grpc::Status(grpc::StatusCode::UNAVAILABLE, "rank failed");
        workers.push_back(worker);
    }
    auto                       broadcast = std::make_shared<Broadcast>(workers);
    P2PBroadcastClient::Result result("handoff", broadcast);
    broadcast->finishRank(0, true);
    EXPECT_FALSE(result.done());
    EXPECT_FALSE(result.success());
    broadcast->finishRank(1, true);
    EXPECT_TRUE(result.done());
    EXPECT_FALSE(result.success());
    EXPECT_TRUE(result.firstError().error.hasError());
}

TEST(DecodeRpcServerTest, PipelineHandoffCarriesStageLocalGeometry) {
    P2PPipelineLayoutPB layout;
    layout.set_worker_count(4);
    for (int stage_id = 0; stage_id < 2; ++stage_id) {
        auto* stage = layout.add_stages();
        stage->set_main_layer_begin(stage_id * 2);
        stage->set_main_layer_count(2);
        stage->add_worker_indices(stage_id * 2);
        stage->add_worker_indices(stage_id * 2 + 1);
        for (int layer = 0; layer < 2; ++layer)
            stage->add_layers()->add_tags("full");
        auto* shards = stage->mutable_main_shards();
        shards->set_tp_size(2);
        shards->set_prefill_cp_size(1);
        shards->set_cache_key_tokens_per_block(4);
        auto* group = shards->add_groups();
        group->set_tag("full");
        group->set_group_type(static_cast<int>(CacheGroupType::FULL));
        group->set_spec_type(KVCacheSpecType::MultiHeadAttention);
        group->set_seq_size_per_block(4);
        group->set_kv_block_stride_bytes(128);
        group->set_k_block_payload_bytes(128);
        group->set_global_kv_heads(2);
    }
    P2PConnectorStartLoadRequestPB request;
    request.set_decode_tp_size(2);
    request.set_decode_pipeline_layout(layout.SerializeAsString());
    const auto decoded = decodeP2PPipelineLayout(request.decode_pipeline_layout(), RoleType::DECODE);
    ASSERT_TRUE(decoded.has_value());
    EXPECT_EQ(decoded->worker_count, 4u);
    ASSERT_EQ(decoded->stages.size(), 2u);
    EXPECT_EQ(decoded->stages[0].main_layer_begin, 0);
    EXPECT_EQ(decoded->stages[1].main_layer_begin, 2);
    EXPECT_EQ(decoded->stages[0].worker_indices, (std::vector<int>{0, 1}));
    EXPECT_EQ(decoded->stages[1].worker_indices, (std::vector<int>{2, 3}));
    const auto plan = planP2PPipelineTransfer(layout.SerializeAsString(), request.decode_pipeline_layout(), "request");
    ASSERT_TRUE(plan.ok()) << plan.error.ToString();
    std::set<std::pair<int, int>> layers_by_rank;
    for (const auto& route : plan.plan.routes) {
        for (const auto& layer : route.layers) {
            EXPECT_EQ(route.src_rank / 2, route.dst_rank / 2);
            EXPECT_EQ(layer.src_local_id, layer.global_id % 2);
            EXPECT_EQ(layer.dst_local_id, layer.global_id % 2);
            layers_by_rank.emplace(layer.global_id, route.dst_rank);
        }
    }
    EXPECT_EQ(layers_by_rank.size(), 8u);
}

TEST(DecodeRpcServerTest, EmptyRemoteLoadTopologyFailsBeforePlanning) {
    EXPECT_FALSE(planP2PPipelineTransfer("", "", "request").ok());
    P2PPipelineLayoutPB empty;
    empty.set_worker_count(0);
    EXPECT_FALSE(planP2PPipelineTransfer(empty.SerializeAsString(), empty.SerializeAsString(), "request").ok());
}

TEST(DecodeRpcServerTest, NullGenerateRequestReturnsInvalidParamsBeforeEngineAccess) {
    DecodeRpcServer server;
    const auto      status = server.GenerateStreamCall(nullptr, nullptr, nullptr);
    EXPECT_FALSE(status.ok());
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), static_cast<int>(ErrorCode::INVALID_PARAMS));
}

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

TEST(DecodeRpcServerTest, BatchResultKeepsFirstTokenAndPrefillOnlyOptionalOutputs) {
    BatchStreamOutputCollector collector(/*logits_index=*/1);
    const auto                 loss          = torch::tensor({0.25f});
    const auto                 hidden        = torch::ones({1, 3});
    const auto                 logits        = torch::tensor({1.0f, 2.0f, 3.0f});
    const auto                 prompt_scores = torch::tensor({-0.5f, -0.25f});
    GenerateOutputs            first;
    first.request_id                       = 42;
    auto& prefill                          = first.generate_outputs.emplace_back();
    prefill.output_ids                     = torch::tensor({{100}}, torch::kInt32);
    prefill.finished                       = false;
    prefill.aux_info.output_len            = 1;
    prefill.loss                           = loss;
    prefill.hidden_states                  = hidden;
    prefill.logits                         = logits;
    prefill.prompt_logits                  = PromptLogitsOutput{};
    prefill.prompt_logits->target_logprobs = prompt_scores;
    ASSERT_TRUE(collector.add(std::move(first)).ok());

    GenerateOutputs next;
    next.request_id            = 42;
    auto& decode               = next.generate_outputs.emplace_back();
    decode.output_ids          = torch::tensor({{101, 102}}, torch::kInt32);
    decode.finished            = true;
    decode.aux_info.output_len = 3;
    decode.logits              = torch::zeros({3});
    ASSERT_TRUE(collector.add(std::move(next)).ok());
    const auto result = collector.finish();
    ASSERT_EQ(result.generate_outputs.size(), 1);
    const auto& output = result.generate_outputs[0];
    EXPECT_EQ(result.request_id, 42);
    EXPECT_TRUE(output.finished);
    EXPECT_TRUE(torch::equal(output.output_ids, torch::tensor({{100, 101, 102}}, torch::kInt32)));
    ASSERT_TRUE(output.loss.has_value());
    ASSERT_TRUE(output.hidden_states.has_value());
    ASSERT_TRUE(output.logits.has_value());
    ASSERT_TRUE(output.prompt_logits.has_value());
    EXPECT_TRUE(torch::equal(*output.loss, loss));
    EXPECT_TRUE(torch::equal(*output.hidden_states, hidden));
    EXPECT_TRUE(torch::equal(*output.logits, logits));
    EXPECT_TRUE(torch::equal(output.prompt_logits->target_logprobs, prompt_scores));
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

TEST(DecodeRpcServerTest, PDSupportUsesTheSameTokenRulesAsQueryConversion) {
    GenerateInputPB request;
    auto*           config = request.mutable_generate_config();
    config->set_can_use_pd_separation(true);
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    ASSERT_TRUE(checkPDSupport(request).supported);

    config->set_return_prompt_logits(true);
    EXPECT_FALSE(checkPDSupport(request).supported);
    config->set_return_prompt_logits(false);
    config->set_prefill_only(true);
    EXPECT_FALSE(checkPDSupport(request).supported);
    EXPECT_STREQ(checkPDSupport(request).reason, "invalid token generation configuration");
    config->set_prefill_only(false);
    config->set_max_new_tokens(-1);
    EXPECT_FALSE(checkPDSupport(request).supported);
    config->set_max_new_tokens(0);
    config->set_prefill_only(true);
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

TEST(DecodeRpcServerTest, GenericReuseAlwaysDescribesPrefillAcrossPoolLayouts) {
    for (bool independent : {false, true}) {
        for (int prefill_total : {0, 128}) {
            for (int decode_total : {64, 128, 256, 2048}) {
                DecodeRpcServer server;
                auto            stream = makeStream({11, 12, 13});
                const int       local  = prefill_total == 0 ? 0 : 32;
                const int       remote = prefill_total - local;
                const int       memory = prefill_total == 0 ? 0 : 8;
                const int       disk   = prefill_total == 0 ? 0 : 4;
                stream->setPrefillReuseLength(prefill_total, local, remote, memory, disk, independent);
                auto outputs = makeOutputsWithDecodeReuse(decode_total, 16, decode_total - 16, 2, 1);
                server.updateAuxInfo(outputs, stream);
                const auto& aux = outputs.flatten_output().aux_info(0);
                EXPECT_EQ(aux.total_reuse_len(), prefill_total);
                EXPECT_EQ(aux.local_reuse_len(), local);
                EXPECT_EQ(aux.remote_reuse_len(), remote);
                EXPECT_EQ(aux.memory_reuse_len(), memory);
                EXPECT_EQ(aux.disk_reuse_len(), disk);
                EXPECT_EQ(aux.prefill_total_reuse_len(), prefill_total);
                EXPECT_EQ(aux.decode_total_reuse_len(), decode_total);
                EXPECT_EQ(aux.decode_local_reuse_len(), 16);
                EXPECT_EQ(aux.decode_remote_reuse_len(), decode_total - 16);
                EXPECT_EQ(aux.decode_memory_reuse_len(), 2);
                EXPECT_EQ(aux.decode_disk_reuse_len(), 1);
            }
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
    EXPECT_EQ(registry.cancel(201, {ErrorCode::PRIORITY_PREEMPTED, "preempted"}).value(), CANCEL_STATUS_ACCEPTED);
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
    ASSERT_TRUE(server.cancel_registry_->complete(handle) || handle->terminal.load());
    ASSERT_TRUE(server.Cancel(&context, &request, &response).ok());
    EXPECT_EQ(response.status(), CANCEL_STATUS_NOT_FOUND);
}

}  // namespace rtp_llm

namespace rtp_llm {
namespace {
class FallbackPrefillEngine: public EngineBase {
public:
    FallbackPrefillEngine(): EngineBase(EngineInitParams()) {}

    GenerateStreamPtr makeStream(const std::shared_ptr<GenerateInput>& input) override {
        ModelConfig model_config;
        model_config.max_seq_len = 4096;
        model_config.vocab_size  = 32000;
        ResourceContext resource_context;
        resource_context.role_type = RoleType::PREFILL;
        return std::make_shared<NormalGenerateStream>(input, model_config, RuntimeConfig{}, resource_context, nullptr);
    }

    GenerateStreamPtr enqueue(const std::shared_ptr<GenerateInput>& input) override {
        auto stream = makeStream(input);
        enqueue(stream);
        return stream;
    }

    void enqueue(GenerateStreamPtr& stream) override {
        enqueued.set_value(stream);
    }

    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override {
        std::vector<GenerateStreamPtr> streams;
        for (const auto& input : inputs)
            streams.push_back(makeStream(input));
        batch_enqueued.set_value(streams);
        return {std::vector<bool>(streams.size(), true), streams};
    }

    absl::Status stop() override {
        return absl::OkStatus();
    }

    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in fallback cancellation test");
    }

    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }

    std::promise<GenerateStreamPtr>              enqueued;
    std::promise<std::vector<GenerateStreamPtr>> batch_enqueued;
};

class LocalPrefillCancelService: public RpcService::Service {
public:
    std::atomic<int>      calls{0};
    std::atomic<int>      status_calls{0};
    bool                  fail_cancel{false};
    bool                  wait_for_status{false};
    std::function<void()> on_cancel;
    std::atomic<int64_t>  request_id{0};
    std::atomic<int64_t>  cancel_error_code{0};
    grpc::Status Cancel(grpc::ServerContext*, const CancelRequestPB* request, CancelResponsePB* response) override {
        if (on_cancel) {
            on_cancel();
        }
        if (fail_cancel) {
            ++calls;
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "test Prefill unavailable");
        }
        request_id        = request->request_id();
        cancel_error_code = request->cancel_error_code();
        EXPECT_TRUE(request->prefill_address().empty());
        response->set_status(wait_for_status ? CANCEL_STATUS_ACCEPTED : CANCEL_STATUS_TOMBSTONED);
        ++calls;
        return grpc::Status::OK;
    }
    grpc::Status GetWorkerStatus(grpc::ServerContext*, const StatusVersionPB*, WorkerStatusPB* response) override {
        response->add_finished_task_list()->set_request_id(request_id.load());
        ++status_calls;
        return grpc::Status::OK;
    }
};

template<typename Predicate>
bool waitForCancel(Predicate predicate) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!predicate() && std::chrono::steady_clock::now() < until) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return predicate();
}
}  // namespace

TEST(DecodeRpcServerTest, IndependentCancelStopsForwardedFallbackStream) {
    auto prefill                                       = std::make_shared<PrefillRpcServer>();
    auto engine                                        = std::make_shared<FallbackPrefillEngine>();
    prefill->maga_init_params_.pd_sep_config.role_type = RoleType::PREFILL;
    prefill->engine_                                   = engine;
    prefill->meta_                                     = std::make_shared<RpcServerRuntimeMeta>();
    prefill->cancel_registry_                          = std::make_shared<PDCancelRegistry>(prefill->meta_);
    RemoteRpcServiceImpl prefill_service;
    prefill_service.local_server_   = prefill;
    prefill_service.prefill_server_ = prefill;
    grpc::ServerBuilder prefill_builder;
    int                 prefill_port = 0;
    prefill_builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &prefill_port);
    prefill_builder.RegisterService(&prefill_service);
    auto prefill_rpc = prefill_builder.BuildAndStart();
    ASSERT_NE(prefill_rpc, nullptr);

    auto decode                                       = std::make_shared<DecodeRpcServer>();
    decode->maga_init_params_.pd_sep_config.role_type = RoleType::DECODE;
    decode->meta_                                     = std::make_shared<RpcServerRuntimeMeta>();
    decode->prefill_server_caller_                    = std::make_shared<PrefillServerCaller>();
    decode->initCancelRegistry();
    RemoteRpcServiceImpl decode_service;
    decode_service.local_server_  = decode;
    decode_service.decode_server_ = decode;
    grpc::ServerBuilder decode_builder;
    int                 decode_port = 0;
    decode_builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &decode_port);
    decode_builder.RegisterService(&decode_service);
    auto decode_rpc = decode_builder.BuildAndStart();
    ASSERT_NE(decode_rpc, nullptr);
    auto stub = RpcService::NewStub(
        grpc::CreateChannel("127.0.0.1:" + std::to_string(decode_port), grpc::InsecureChannelCredentials()));

    GenerateInputPB input;
    input.set_request_id(203);
    input.add_token_ids(1);
    auto* config = input.mutable_generate_config();
    config->set_timeout_ms(10000);
    config->set_max_new_tokens(1);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_can_use_pd_separation(false);
    auto* address = config->add_role_addrs();
    address->set_role(RoleAddrPB::PREFILL);
    address->set_ip("127.0.0.1");
    address->set_grpc_port(prefill_port);
    auto enqueued = engine->enqueued.get_future();
    auto generate = std::async(std::launch::async, [&] {
        grpc::ClientContext context;
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
        auto              reader = stub->GenerateStreamCall(&context, input);
        GenerateOutputsPB output;
        while (reader->Read(&output)) {}
        return reader->Finish();
    });
    ASSERT_EQ(enqueued.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    auto stream = enqueued.get();
    EXPECT_FALSE(stream->hasError());
    EXPECT_EQ(generate.wait_for(std::chrono::seconds(0)), std::future_status::timeout);

    // No Prefill address on Cancel: Decode must use the route of the active request.
    grpc::ClientContext cancel_context;
    cancel_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(3));
    CancelRequestPB cancel;
    cancel.set_request_id(input.request_id());
    CancelResponsePB response;
    ASSERT_TRUE(stub->Cancel(&cancel_context, cancel, &response).ok());
    EXPECT_EQ(response.status(), CANCEL_STATUS_ACCEPTED);
    ASSERT_EQ(generate.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_EQ(errorInfoFromGrpcStatus(generate.get()).code(), ErrorCode::PRIORITY_PREEMPTED);
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::PRIORITY_PREEMPTED);
    EXPECT_TRUE(stream->finishWithoutGenerate());
    const auto handle = decode->cancel_registry_->find(input.request_id());
    ASSERT_NE(handle, nullptr);
    EXPECT_TRUE(waitForCancel(
        [&] { return handle->terminal.load() && !decode->cancel_registry_->needsDownstreamCancel(handle); }));
}

TEST(DecodeRpcServerTest, UnimplementedBatchFallbackKeepsIndependentCancelLive) {
    auto prefill                                          = std::make_shared<PrefillRpcServer>();
    auto prefill_engine                                   = std::make_shared<FallbackPrefillEngine>();
    prefill->maga_init_params_.pd_sep_config.role_type    = RoleType::PREFILL;
    prefill->maga_init_params_.parallelism_config.dp_size = 2;
    prefill->engine_                                      = prefill_engine;
    prefill->meta_                                        = std::make_shared<RpcServerRuntimeMeta>();
    prefill->cancel_registry_                             = std::make_shared<PDCancelRegistry>(prefill->meta_);
    RemoteRpcServiceImpl prefill_service;
    prefill_service.local_server_   = prefill;
    prefill_service.prefill_server_ = prefill;
    grpc::ServerBuilder prefill_builder;
    int                 prefill_port = 0;
    prefill_builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &prefill_port);
    prefill_builder.RegisterService(&prefill_service);
    auto prefill_rpc = prefill_builder.BuildAndStart();
    ASSERT_NE(prefill_rpc, nullptr);

    auto decode                                       = std::make_shared<DecodeRpcServer>();
    decode->maga_init_params_.pd_sep_config.role_type = RoleType::DECODE;
    decode->engine_                                   = std::make_shared<FallbackPrefillEngine>();
    decode->meta_                                     = std::make_shared<RpcServerRuntimeMeta>();
    decode->prefill_server_caller_                    = std::make_shared<PrefillServerCaller>();
    decode->initCancelRegistry();
    RemoteRpcServiceImpl decode_service;
    decode_service.local_server_  = decode;
    decode_service.decode_server_ = decode;
    grpc::ServerBuilder decode_builder;
    int                 decode_port = 0;
    decode_builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &decode_port);
    decode_builder.RegisterService(&decode_service);
    auto decode_rpc = decode_builder.BuildAndStart();
    ASSERT_NE(decode_rpc, nullptr);
    auto stub = RpcService::NewStub(
        grpc::CreateChannel("127.0.0.1:" + std::to_string(decode_port), grpc::InsecureChannelCredentials()));
    BatchGenerateInputPB batch;
    for (int64_t id : {301, 302}) {
        auto* item = batch.add_inputs();
        item->set_request_id(id);
        item->add_token_ids(1);
        auto* config = item->mutable_generate_config();
        config->set_timeout_ms(10000);
        config->set_max_new_tokens(2);
        config->set_num_beams(1);
        config->set_num_return_sequences(1);
        config->set_can_use_pd_separation(true);
        auto* address = config->add_role_addrs();
        address->set_role(RoleAddrPB::PREFILL);
        address->set_ip("127.0.0.1");
        address->set_grpc_port(prefill_port);
    }
    auto                enqueued = prefill_engine->batch_enqueued.get_future();
    grpc::ClientContext context;
    context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
    BatchGenerateOutputsPB response;
    auto call = std::async(std::launch::async, [&] { return stub->BatchGenerateCall(&context, batch, &response); });
    // Always release the blocked RPC if an assertion fails.
    autil::ScopeGuard cancel_call([&] { context.TryCancel(); });
    ASSERT_EQ(enqueued.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    auto streams = enqueued.get();
    ASSERT_EQ(streams.size(), 2u);
    ASSERT_TRUE(waitForCancel([&] { return prefill->cancel_registry_->find(302) != nullptr; }));
    const auto handle = decode->cancel_registry_->find(301);
    ASSERT_NE(handle, nullptr);
    EXPECT_FALSE(handle->local_done.load());
    EXPECT_FALSE(handle->terminal.load());
    EXPECT_TRUE(decode->meta_->getEngineScheduleInfo(-1).finished_task_info_list.empty());
    EXPECT_EQ(streams[0]->generateInput()->group_id, -1);

    grpc::ClientContext cancel_context;
    cancel_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(3));
    CancelRequestPB cancel;
    cancel.set_request_id(301);  // No route hint: use the live Decode control record.
    CancelResponsePB cancel_response;
    ASSERT_TRUE(stub->Cancel(&cancel_context, cancel, &cancel_response).ok());
    EXPECT_EQ(cancel_response.status(), CANCEL_STATUS_ACCEPTED);
    ASSERT_TRUE(waitForCancel([&] { return streams[0]->hasError(); }));
    EXPECT_EQ(streams[0]->statusInfo().code(), ErrorCode::PRIORITY_PREEMPTED);
    EXPECT_FALSE(streams[1]->hasError());
    streams[0]->finishWithoutGenerate();
    EXPECT_EQ(call.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
    // The unaffected sibling finishes normally, not by timeout or client cancellation.
    streams[1]->generate_status_->status.store(StreamState::RUNNING);
    streams[1]->step();
    streams[1]->update(StreamUpdateInfo{.new_tokens     = torch::tensor({{7, 8}}, torch::kInt32),
                                        .num_new_tokens = 2,
                                        .cum_log_probs  = torch::tensor({-0.5f})});
    ASSERT_EQ(call.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_TRUE(call.get().ok());
    ASSERT_EQ(response.results_size(), 2);
    EXPECT_EQ(response.results(0).error_info().error_code(), ErrorCodePB::PRIORITY_PREEMPTED);
    EXPECT_FALSE(response.results(1).has_error_info());
    EXPECT_TRUE(response.results(1).has_final_output());
    streams[0]->finishWithoutGenerate();
    EXPECT_TRUE(waitForCancel(
        [&] { return handle->terminal.load() && !decode->cancel_registry_->needsDownstreamCancel(handle); }));
}

TEST(DecodeRpcServerTest, CancelBeforeDecodeAdmissionForwardsToSelectedPrefillAndFencesLateRequest) {
    for (auto code : {ErrorCode::PRIORITY_PREEMPTED, ErrorCode::CANCELLED, ErrorCode::GENERATE_TIMEOUT}) {

        LocalPrefillCancelService prefill;
        prefill.wait_for_status = code == ErrorCode::PRIORITY_PREEMPTED;
        grpc::ServerBuilder builder;
        int                 port = 0;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
        builder.RegisterService(&prefill);
        auto rpc_server = builder.BuildAndStart();
        ASSERT_TRUE(rpc_server);
        DecodeRpcServer decode;
        decode.meta_ = std::make_shared<RpcServerRuntimeMeta>();
        decode.initCancelRegistry();
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
        ASSERT_TRUE(waitForCancel(
            [&] { return handle->terminal.load() && !decode.cancel_registry_->needsDownstreamCancel(handle); }));
        EXPECT_EQ(prefill.calls.load(), 1);
        EXPECT_EQ(prefill.status_calls.load(), prefill.wait_for_status ? 1 : 0);
        EXPECT_EQ(prefill.request_id.load(), 203);
        EXPECT_EQ(prefill.cancel_error_code.load(), static_cast<int64_t>(code));
        const auto finished = decode.meta_->getEngineScheduleInfo(0);
        if (code == ErrorCode::PRIORITY_PREEMPTED) {
            ASSERT_EQ(finished.finished_task_info_list.size(), 1u);
            EXPECT_EQ(finished.finished_task_info_list.front().error_code, static_cast<int64_t>(code));
        } else {
            EXPECT_TRUE(finished.finished_task_info_list.empty());
        }
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

TEST(DecodeRpcServerTest, FailedPrefillCancelBacksOffAfterLocalCompletion) {
    LocalPrefillCancelService prefill;
    prefill.fail_cancel = true;
    grpc::ServerBuilder builder;
    int                 port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&prefill);
    auto rpc_server = builder.BuildAndStart();
    ASSERT_TRUE(rpc_server);
    DecodeRpcServer decode;
    decode.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    decode.initCancelRegistry();
    CancelRequestPB request;
    request.set_request_id(205);
    request.set_prefill_address("127.0.0.1:" + std::to_string(port));
    CancelResponsePB    response;
    grpc::ServerContext context;
    ASSERT_TRUE(decode.Cancel(&context, &request, &response).ok());
    auto handle = decode.cancel_registry_->find(205);
    ASSERT_TRUE(handle);
    ASSERT_TRUE(waitForCancel([&] { return handle->terminal.load(); }));
    EXPECT_TRUE(decode.cancel_registry_->needsDownstreamCancel(handle));
    ASSERT_TRUE(waitForCancel([&] { return prefill.calls.load() >= 2; }));
    EXPECT_GE(handle->cancel_retry_delay_ms.load(), 100);
    ASSERT_EQ(decode.meta_->getEngineScheduleInfo(0).finished_task_info_list.size(), 1u);
    rpc_server->Shutdown();
}

TEST(DecodeRpcServerTest, SlowPrefillCancelDoesNotBlockOtherLocalTerminals) {
    LocalPrefillCancelService prefill;
    std::promise<void>        release;
    const auto                gate = release.get_future().share();
    std::atomic<bool>         prefill_entered{false};
    prefill.on_cancel = [&] {
        prefill_entered.store(true);
        gate.wait_for(std::chrono::seconds(2));
    };
    grpc::ServerBuilder builder;
    int                 port = 0;
    builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &port);
    builder.RegisterService(&prefill);
    auto rpc_server = builder.BuildAndStart();
    ASSERT_TRUE(rpc_server);
    DecodeRpcServer decode;
    decode.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    decode.initCancelRegistry();
    const auto cancel = [&](int64_t id, const std::string& peer) {
        CancelRequestPB request;
        request.set_request_id(id);
        request.set_prefill_address(peer);
        CancelResponsePB    response;
        grpc::ServerContext context;
        return decode.Cancel(&context, &request, &response);
    };
    ASSERT_TRUE(cancel(210, "127.0.0.1:" + std::to_string(port)).ok());
    ASSERT_TRUE(waitForCancel([&] { return prefill_entered.load(); }));

    GenerateInputPB input;
    input.set_request_id(211);
    PDCancelRegistry::Handle local;
    ASSERT_TRUE(decode.cancel_registry_->admit(input, "local", "", currentTimeMs() + 10000, local).ok());
    // Cancel while the local request is active, then publish its cleanup.
    EXPECT_TRUE(cancel(211, "").ok());
    decode.cancel_registry_->finishLocal(local);
    EXPECT_TRUE(waitForCancel([&] { return local->terminal.load(); }));
    EXPECT_EQ(prefill.calls.load(), 0);
    release.set_value();
    rpc_server->Shutdown();
}

TEST(DecodeRpcServerTest, LocalReadMustStopBeforeTerminalEvenDuringRemoteCancelBackoff) {
    DecodeRpcServer decode;
    decode.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    decode.initCancelRegistry();
    GenerateInputPB input;
    input.set_request_id(206);
    PDCancelRegistry::Handle handle;
    ASSERT_TRUE(
        decode.cancel_registry_->admit(input, "local-read206", "127.0.0.1:1", currentTimeMs() + 1000, handle).ok());
    decode.cancel_registry_->beginRead(handle->unique_key);
    decode.cancel_registry_->finishLocal(handle);
    ASSERT_EQ(decode.cancel_registry_->cancel(206, {ErrorCode::PRIORITY_PREEMPTED, "preempted"}).value(),
              CANCEL_STATUS_ACCEPTED);
    EXPECT_FALSE(handle->terminal.load());
    EXPECT_TRUE(decode.meta_->getEngineScheduleInfo(0).finished_task_info_list.empty());
    decode.cancel_registry_->endRead(handle->unique_key);
    ASSERT_TRUE(waitForCancel([&] { return handle->terminal.load(); }));
    EXPECT_TRUE(decode.cancel_registry_->needsDownstreamCancel(handle));
    ASSERT_EQ(decode.meta_->getEngineScheduleInfo(0).finished_task_info_list.size(), 1u);
}

TEST(DecodeRpcServerTest, UnreachablePrefillAllowsLocalTerminalButRetainsRemoteRetry) {
    DecodeRpcServer decode;
    decode.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    decode.initCancelRegistry();
    CancelRequestPB request;
    request.set_request_id(204);
    request.set_prefill_address("127.0.0.1:1");
    CancelResponsePB    response;
    grpc::ServerContext context;
    ASSERT_TRUE(decode.Cancel(&context, &request, &response).ok());
    auto handle = decode.cancel_registry_->find(204);
    ASSERT_TRUE(handle);
    ASSERT_TRUE(waitForCancel([&] { return handle->terminal.load(); }));
    EXPECT_TRUE(decode.cancel_registry_->needsDownstreamCancel(handle));
    EXPECT_TRUE(decode.cancel_registry_->isCanceled(handle->unique_key));
    ASSERT_EQ(decode.meta_->getEngineScheduleInfo(0).finished_task_info_list.size(), 1u);
}
TEST(DecodeRpcServerTest, CancelCapacityFailureDoesNotAcknowledgeAnUnstoredFence) {
    DecodeRpcServer server;
    server.meta_                               = std::make_shared<RpcServerRuntimeMeta>();
    server.cancel_registry_                    = std::make_unique<PDCancelRegistry>(server.meta_);
    server.cancel_registry_->control_capacity_ = 1;
    ASSERT_TRUE(server.cancel_registry_->cancel(130, {ErrorCode::CANCELLED, "existing fence"}).ok());
    CancelRequestPB request;
    request.set_request_id(131);
    CancelResponsePB    response;
    grpc::ServerContext context;
    const auto          status = server.Cancel(&context, &request, &response);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_NE(response.status(), CANCEL_STATUS_TOMBSTONED);
    EXPECT_NE(response.status(), CANCEL_STATUS_ACCEPTED);
    EXPECT_EQ(server.cancel_registry_->absent_fences_.count(131), 0u);
}

TEST(DecodeRpcServerTest, EarlyCancelWithPrefillAddressUsesControlCapacity) {
    DecodeRpcServer server;
    server.meta_                               = std::make_shared<RpcServerRuntimeMeta>();
    server.cancel_registry_                    = std::make_unique<PDCancelRegistry>(server.meta_);
    server.cancel_registry_->record_capacity_  = 1;
    server.cancel_registry_->control_capacity_ = 1;

    grpc::ServerContext context;
    CancelRequestPB     early_cancel;
    early_cancel.set_request_id(140);
    early_cancel.set_prefill_address("prefill:123");
    CancelResponsePB response;
    ASSERT_TRUE(server.Cancel(&context, &early_cancel, &response).ok());
    EXPECT_EQ(response.status(), CANCEL_STATUS_ACCEPTED);

    GenerateInputPB input;
    input.set_request_id(141);
    PDCancelRegistry::Handle handle;
    ASSERT_TRUE(server.cancel_registry_->admit(input, "active141", "", currentTimeMs() + 10000, handle).ok());

    early_cancel.set_request_id(142);
    response.Clear();
    EXPECT_EQ(server.Cancel(&context, &early_cancel, &response).error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_FALSE(server.cancel_registry_->find(142));
    EXPECT_EQ(server.cancel_registry_->control_record_count_, 1u);

    const auto early = server.cancel_registry_->find(140);
    ASSERT_TRUE(early);
    ASSERT_TRUE(server.cancel_registry_->complete(early) || early->terminal.load());
    {
        std::lock_guard<std::mutex> lock(server.cancel_registry_->mutex_);
        server.cancel_registry_->trimLocked(early->retain_until_ms);
    }
    EXPECT_EQ(server.cancel_registry_->control_record_count_, 0u);
    EXPECT_TRUE(server.Cancel(&context, &early_cancel, &response).ok());
    EXPECT_EQ(response.status(), CANCEL_STATUS_ACCEPTED);
}

TEST(BatchStreamOutputCollectorTest, MetricOnlyMarkersDoNotReplaceTokensOrLoseSpeculativeSamples) {
    BatchStreamOutputCollector collector;
    GenerateOutputs            metric;
    metric.frontend_metric_only = true;
    metric.frontend_sp_tpot_samples.emplace_back(1, 2.0);
    ASSERT_TRUE(collector.add(std::move(metric)).ok());
    EXPECT_TRUE(collector.empty());
    GenerateOutputs terminal;
    terminal.request_id = 7;
    terminal.generate_outputs.emplace_back();
    terminal.generate_outputs.back().finished   = true;
    terminal.generate_outputs.back().output_ids = torch::tensor({{3}}, torch::kInt32);
    terminal.frontend_sp_tpot_samples.emplace_back(2, 4.0);
    ASSERT_TRUE(collector.add(std::move(terminal)).ok());
    auto output = collector.finish();
    ASSERT_EQ(output.generate_outputs.size(), 1u);
    EXPECT_TRUE(output.generate_outputs.front().finished);
    EXPECT_EQ(output.generate_outputs.front().output_ids.item<int32_t>(), 3);
    ASSERT_EQ(output.frontend_sp_tpot_samples.size(), 2u);
    EXPECT_EQ(output.frontend_sp_tpot_samples.front().first, 1);
    EXPECT_EQ(output.frontend_sp_tpot_samples.back().first, 2);
}

}  // namespace rtp_llm
