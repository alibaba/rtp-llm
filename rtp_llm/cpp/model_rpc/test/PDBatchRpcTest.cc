#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>
#include <thread>
#include "gtest/gtest.h"
#include "rtp_llm/cpp/model_rpc/BatchStreamOutputCollector.h"
#include "rtp_llm/cpp/model_rpc/DecodeRpcServerNew2.h"
#include "rtp_llm/cpp/model_rpc/PrefillRpcServerNew2.h"
#include "rtp_llm/cpp/model_rpc/PDRequestUtils.h"
#include "rtp_llm/cpp/model_rpc/QueryConverter.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorPrefill.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/testing/TestBase.h"

namespace rtp_llm {
namespace {

GenerateInputPB batchItem(int64_t id) {
    GenerateInputPB item;
    item.set_request_id(id);
    item.add_token_ids(1);
    item.add_token_ids(2);
    auto* config = item.mutable_generate_config();
    config->set_can_use_pd_separation(true);
    config->set_max_new_tokens(8);
    config->set_num_beams(1);
    config->set_num_return_sequences(1);
    config->set_timeout_ms(5000);
    config->set_unique_key("business-key");
    return item;
}

GenerateOutputs chunk(int64_t id, std::vector<int32_t> ids, bool finished) {
    GenerateOutputs output;
    output.request_id = id;
    GenerateOutput item;
    item.finished                 = finished;
    item.output_ids               = torch::tensor(ids, torch::kInt32).reshape({1, static_cast<int64_t>(ids.size())});
    item.aux_info.step_output_len = ids.size();
    output.generate_outputs.push_back(std::move(item));
    return output;
}

class BatchTestStream: public NormalGenerateStream {
public:
    BatchTestStream(const std::shared_ptr<GenerateInput>& input):
        NormalGenerateStream(input, modelConfig(), RuntimeConfig{}, ResourceContext{}, nullptr) {}
    StreamState getStatus() const override {
        return terminal ? StreamState::FINISHED : StreamState::WAITING;
    }
    bool terminal = false;

private:
    static ModelConfig modelConfig() {
        ModelConfig config;
        config.max_seq_len = 4096;
        config.vocab_size  = 32000;
        return config;
    }
};

class BatchTestEngine: public EngineBase {
public:
    BatchTestEngine(): EngineBase(EngineInitParams{}) {}
    std::shared_ptr<GenerateStream> makeStream(const std::shared_ptr<GenerateInput>& input) override {
        auto stream = std::make_shared<BatchTestStream>(input);
        made.push_back(stream);
        return stream;
    }
    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        ADD_FAILURE() << "single enqueue used for batch";
        return nullptr;
    }
    void enqueue(std::shared_ptr<GenerateStream>&) override {
        ADD_FAILURE() << "single enqueue used for batch";
    }
    std::vector<GenerateStreamPtr> batchEnqueue(const std::vector<std::shared_ptr<GenerateInput>>& inputs) override {
        std::vector<GenerateStreamPtr> streams;
        for (const auto& input : inputs)
            streams.push_back(makeStream(input));
        return batchEnqueue(streams);
    }
    std::vector<GenerateStreamPtr> batchEnqueue(const std::vector<GenerateStreamPtr>& streams) override {
        for (const auto& stream : streams) {
            auto       typed = std::static_pointer_cast<BatchTestStream>(stream);
            const auto id    = stream->generateInput()->request_id;
            if (id == fail_id) {
                stream->reportError(ErrorCode::MALLOC_FAILED, "injected allocation failure");
            } else if (!hold) {
                typed->enqueueGenerateOutput(chunk(id, {static_cast<int32_t>(id)}, false));
                typed->enqueueGenerateOutput(chunk(id, {10, 20}, true));
                typed->terminal = true;
            }
        }
        ++batch_calls;
        return streams;
    }
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("test");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return {};
    }
    std::atomic<int>                              batch_calls{0};
    std::vector<std::shared_ptr<BatchTestStream>> made;
    bool                                          hold    = false;
    int64_t                                       fail_id = -1;
};

class BatchRpcService: public RpcService::Service {
public:
    LocalRpcServer*          target = nullptr;
    int                      port   = 0;
    std::atomic<int>         peer_calls{0};
    std::atomic<int>         batch_calls{0};
    std::atomic<int>         completed_calls{0};
    BatchGenerateInputPB     received;
    std::vector<std::string> dp_addrs;
    bool                     change_topology = false;
    bool                     fail_peer       = false;
    bool                     wrong_count     = false;
    bool                     response_error  = false;
    grpc::Status
    GetPeerInfo(grpc::ServerContext*, const GetPeerInfoRequestPB*, GetPeerInfoResponsePB* response) override {
        const auto call = ++peer_calls;
        if (fail_peer)
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "peer unavailable");
        response->set_tp_size(change_topology && call > 1 ? 4 : 2);
        response->set_cp_size(1);
        if (dp_addrs.empty())
            response->add_dp_grpc_addrs("127.0.0.1:" + std::to_string(port));
        else
            for (const auto& address : dp_addrs)
                response->add_dp_grpc_addrs(address);
        return grpc::Status::OK;
    }
    grpc::Status BatchGenerateCall(grpc::ServerContext*        context,
                                   const BatchGenerateInputPB* request,
                                   BatchGenerateOutputsPB*     response) override {
        received.CopyFrom(*request);
        ++batch_calls;
        auto status = target->BatchGenerateCall(context, request, response);
        if (wrong_count) {
            if (response_error && response->results_size() > 1)
                response->mutable_results()->RemoveLast();
            else
                response->clear_results();
        }
        if (response_error && response->results_size() > 0) {
            auto* error = response->mutable_results(0)->mutable_error_info();
            error->set_error_code(ErrorCodePB::MM_PROCESS_ERROR);
            error->set_error_message("injected result error");
        }
        ++completed_calls;
        return status;
    }
};

class BatchRpcHarness {
public:
    explicit BatchRpcHarness(LocalRpcServer* target) {
        service.target = target;
        grpc::ServerBuilder builder;
        builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &service.port);
        builder.RegisterService(&service);
        server = builder.BuildAndStart();
    }
    ~BatchRpcHarness() {
        if (server) {
            server->Shutdown(std::chrono::system_clock::now());
            server->Wait();
        }
    }
    std::string address() const {
        return "127.0.0.1:" + std::to_string(service.port);
    }
    BatchRpcService               service;
    std::unique_ptr<grpc::Server> server;
};

bool waitUntil(const std::function<bool()>& condition) {
    const auto deadline = currentTimeMs() + 3000;
    while (!condition() && currentTimeMs() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return condition();
}

class PDBatchRpcTest: public DeviceTestBase {
public:
    void SetUp() override {
        DeviceTestBase::SetUp();
        decode_engine                 = std::make_shared<BatchTestEngine>();
        prefill_engine                = std::make_shared<BatchTestEngine>();
        decode.engine_                = decode_engine;
        prefill.engine_               = prefill_engine;
        // Stub model execution, but use the real per-request timeout store.
        prefill_engine->resource_context_.cache_manager = std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(1, 2, 1, DataType::TYPE_FP16), true);
        auto connector = std::make_shared<P2PConnector>(P2PConnectorConfig{}, nullptr, nullptr);
        connector->prefill_ = std::make_unique<P2PConnectorPrefill>(P2PConnectorConfig{}, nullptr, nullptr);
        connector->prefill_->stream_store_ = std::make_shared<P2PConnectorResourceStore>(nullptr, 10);
        prefill_engine->resource_context_.cache_manager->p2p_connector_ = connector;
        decode.meta_                  = std::make_shared<RpcServerRuntimeMeta>();
        prefill.meta_                 = std::make_shared<RpcServerRuntimeMeta>();
        decode.prefill_server_caller_ = std::make_shared<PrefillServerCaller>("batch-test");
        prefill_rpc                   = std::make_unique<BatchRpcHarness>(&prefill);
        ASSERT_NE(prefill_rpc->server, nullptr);
    }
    BatchGenerateInputPB request() {
        BatchGenerateInputPB batch;
        for (int i : {1, 2})
            *batch.add_inputs() = batchItem(i);
        auto* addr = batch.mutable_inputs(0)->mutable_generate_config()->add_role_addrs();
        addr->set_role(RoleAddrPB::PREFILL);
        addr->set_ip("127.0.0.1");
        addr->set_grpc_port(prefill_rpc->service.port);
        return batch;
    }
    BatchGenerateInputPB handoff() {
        auto batch = request();
        for (auto& item : *batch.mutable_inputs()) {
            item.mutable_generate_config()->set_unique_key("handoff-" + std::to_string(item.request_id()));
        }
        return batch;
    }
    DecodeRpcServerNew2              decode;
    PrefillRpcServerNew2             prefill;
    std::shared_ptr<BatchTestEngine> decode_engine;
    std::shared_ptr<BatchTestEngine> prefill_engine;
    std::unique_ptr<BatchRpcHarness> prefill_rpc;
};

TEST_F(PDBatchRpcTest, OneRpcAndOneEnqueuePerSidePreserveIdentityKeysAndOutputOrder) {
    auto batch = request();
    batch.mutable_inputs(0)->mutable_generate_config()->set_calculate_loss(1);
    batch.mutable_inputs(0)->mutable_generate_config()->set_return_logits(true);
    batch.mutable_inputs(0)->mutable_generate_config()->set_return_hidden_states(true);
    for (auto& item : *batch.mutable_inputs()) {
        item.set_batch_group_size(2);
        item.mutable_batch_group_id()->set_value(77);
    }
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    ASSERT_TRUE(decode.BatchGenerateCall(&context, &batch, &response).ok());
    ASSERT_TRUE(waitUntil([&] { return prefill_rpc->service.completed_calls == 1; }));
    EXPECT_EQ(prefill_rpc->service.peer_calls, 2);
    EXPECT_EQ(prefill_rpc->service.batch_calls, 1);
    EXPECT_EQ(decode_engine->batch_calls, 1);
    EXPECT_EQ(prefill_engine->batch_calls, 1);
    ASSERT_EQ(response.results_size(), 2);
    ASSERT_EQ(decode_engine->made.size(), 2);
    for (int i = 0; i < 2; ++i) {
        auto& stream = decode_engine->made[i];
        EXPECT_EQ(stream->getPrefillTpSize(), 2);
        EXPECT_EQ(stream->getPrefillCpSize(), 1);
        const auto& sent = prefill_rpc->service.received.inputs(i);
        EXPECT_EQ(sent.generate_config().unique_key(), stream->uniqueKey());
        EXPECT_NE(sent.generate_config().unique_key(), "business-key");
        EXPECT_EQ(sent.generate_config().timeout_ms(), stream->generateConfig()->timeout_ms);
        EXPECT_EQ(stream->generateInput()->batch_group_size, 2);
        EXPECT_EQ(stream->generateInput()->batch_group_id, 77);
        EXPECT_EQ(sent.batch_group_size(), 2);
        EXPECT_EQ(sent.batch_group_id().value(), 77);
        EXPECT_EQ(response.results(i).final_output().request_id(), i + 1);
        auto tokens = QueryConverter::transTensor(response.results(i).final_output().flatten_output().output_ids());
        EXPECT_TRUE(torch::equal(tokens.flatten(), torch::tensor({i + 1, 10, 20}, torch::kInt32)));
    }
    EXPECT_NE(decode_engine->made[0]->uniqueKey(), decode_engine->made[1]->uniqueKey());
    EXPECT_EQ(batch.inputs(0).generate_config().unique_key(), "business-key");
    EXPECT_EQ(batch.inputs(1).generate_config().role_addrs_size(), 0);
    EXPECT_TRUE(decode.meta_->getEngineScheduleInfo(0).running_task_info_list.empty());
    EXPECT_TRUE(prefill.meta_->getEngineScheduleInfo(0).running_task_info_list.empty());
}

TEST_F(PDBatchRpcTest, FixedSizeBatchesRotateDpAsAWholeAndNeverCachePeerInfo) {
    BatchRpcHarness second(&prefill);
    ASSERT_NE(second.server, nullptr);
    prefill_rpc->service.dp_addrs = {prefill_rpc->address(), second.address()};
    auto                   batch  = request();
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    ASSERT_TRUE(decode.BatchGenerateCall(&context, &batch, &response).ok());
    ASSERT_TRUE(decode.BatchGenerateCall(&context, &batch, &response).ok());
    EXPECT_EQ(prefill_rpc->service.peer_calls, 4);
    EXPECT_EQ(prefill_rpc->service.batch_calls, 1);
    EXPECT_EQ(second.service.batch_calls, 1);
    ASSERT_EQ(decode_engine->made.size(), 4);
    for (int i = 2; i < 4; ++i) {
        const auto& roles = decode_engine->made[i]->generateConfig()->role_addrs;
        ASSERT_FALSE(roles.empty());
        EXPECT_EQ(roles[0].grpc_port, second.service.port);
    }
}

TEST(PDBatchEntryTest, EmptyBatchSucceedsWithoutEngineOrCaller) {
    DecodeRpcServerNew2    decode;
    PrefillRpcServerNew2   prefill;
    grpc::ServerContext    context;
    BatchGenerateInputPB   request;
    BatchGenerateOutputsPB response;
    EXPECT_TRUE(decode.BatchGenerateCall(&context, &request, &response).ok());
    EXPECT_TRUE(prefill.BatchGenerateCall(&context, &request, &response).ok());
    EXPECT_EQ(response.results_size(), 0);
}

TEST_F(PDBatchRpcTest, FailedPeerProbeNeverReusesEarlierBatchTopology) {
    auto                   batch = request();
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    ASSERT_TRUE(decode.BatchGenerateCall(&context, &batch, &response).ok());
    prefill_rpc->service.fail_peer = true;
    EXPECT_FALSE(decode.BatchGenerateCall(&context, &batch, &response).ok());
    EXPECT_EQ(prefill_rpc->service.peer_calls, 3);
    EXPECT_EQ(prefill_rpc->service.batch_calls, 1);
    EXPECT_EQ(decode_engine->batch_calls, 1);
}

TEST_F(PDBatchRpcTest, CallerRejectsIncompleteAndErrorResponses) {
    auto                batch = handoff();
    PrefillServerCaller client("test");
    prefill_rpc->service.wrong_count = true;
    auto call = std::move(client.callPrefillBatch(batch, prefill_rpc->address(), currentTimeMs() + 5000).value());
    ASSERT_NE(call, nullptr);
    ASSERT_TRUE(waitUntil([&] { return call->done(); }));
    EXPECT_EQ(call->status().error_message(), "prefill batch result count mismatch");
    prefill_rpc->service.wrong_count    = false;
    prefill_rpc->service.response_error = true;
    call = std::move(client.callPrefillBatch(batch, prefill_rpc->address(), currentTimeMs() + 5000).value());
    ASSERT_NE(call, nullptr);
    ASSERT_TRUE(waitUntil([&] { return call->done(); }));
    EXPECT_NE(call->status().error_message().find("injected result error"), std::string::npos);
    EXPECT_EQ(std::move(client.callPrefillBatch(batch, prefill_rpc->address(), currentTimeMs() - 1).value()), nullptr);
}

TEST_F(PDBatchRpcTest, DestroyingCallerCancelsAndDrainsOutstandingRpc) {
    prefill_engine->hold = true;
    PrefillServerCaller client("test");
    auto call = std::move(client.callPrefillBatch(handoff(), prefill_rpc->address(), currentTimeMs() + 5000).value());
    ASSERT_NE(call, nullptr);
    ASSERT_TRUE(waitUntil([&] {
        // This direct asynchronous caller owns its CQ; polling advances the RPC.
        (void)call->done();
        return prefill_engine->batch_calls > 0;
    }));
    call.reset();
    ASSERT_TRUE(waitUntil([&] { return prefill_rpc->service.completed_calls > 0; }));
    EXPECT_TRUE(prefill_engine->made[0]->hasError());
}

TEST_F(PDBatchRpcTest, RejectMixedBatchBeforeRpcOrEnqueue) {
    auto batch = request();
    batch.mutable_inputs(1)->mutable_generate_config()->set_max_new_tokens(1);
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    EXPECT_EQ(decode.BatchGenerateCall(&context, &batch, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(prefill.BatchGenerateCall(&context, &batch, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(prefill_rpc->service.peer_calls, 0);
    EXPECT_EQ(decode_engine->batch_calls, 0);
    EXPECT_EQ(prefill_engine->batch_calls, 0);
}

TEST_F(PDBatchRpcTest, AllNonPdRequestsForwardAsOneBatch) {
    auto batch = request();
    for (auto& item : *batch.mutable_inputs())
        item.mutable_generate_config()->set_can_use_pd_separation(false);
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    ASSERT_TRUE(decode.BatchGenerateCall(&context, &batch, &response).ok());
    EXPECT_EQ(response.results_size(), 2);
    EXPECT_EQ(decode_engine->batch_calls, 0);
    EXPECT_EQ(prefill_engine->batch_calls, 1);
    EXPECT_EQ(prefill_rpc->service.peer_calls, 0);
    EXPECT_EQ(prefill_rpc->service.batch_calls, 1);
}

TEST_F(PDBatchRpcTest, RejectConflictingPrefillTargetsBeforeRpc) {
    auto  batch = request();
    auto* addr  = batch.mutable_inputs(1)->mutable_generate_config()->add_role_addrs();
    addr->set_role(RoleAddrPB::PREFILL);
    addr->set_ip("127.0.0.2");
    addr->set_grpc_port(1234);
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    EXPECT_EQ(decode.BatchGenerateCall(&context, &batch, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(prefill_rpc->service.peer_calls, 0);
    EXPECT_EQ(decode_engine->batch_calls, 0);
}

TEST_F(PDBatchRpcTest, TopologyChangeRejectsWholeBatchBeforeEnqueue) {
    prefill_rpc->service.change_topology = true;
    auto                   batch         = request();
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    EXPECT_EQ(decode.BatchGenerateCall(&context, &batch, &response).error_code(),
              grpc::StatusCode::FAILED_PRECONDITION);
    EXPECT_EQ(prefill_rpc->service.peer_calls, 2);
    EXPECT_EQ(prefill_rpc->service.batch_calls, 0);
    EXPECT_TRUE(decode_engine->made.empty());
}

TEST_F(PDBatchRpcTest, DecodePreprocessingFailureDoesNotCreateOrEnqueueAnyStream) {
    auto batch = request();
    batch.mutable_inputs(1)->add_multimodal_inputs()->set_multimodal_url("missing-processor");
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    EXPECT_FALSE(decode.BatchGenerateCall(&context, &batch, &response).ok());
    EXPECT_TRUE(decode_engine->made.empty());
    EXPECT_EQ(prefill_rpc->service.batch_calls, 0);
}

TEST_F(PDBatchRpcTest, PrefillSnapshotMismatchDoesNotCreateOrEnqueueAnyStream) {
    auto batch = handoff();
    batch.mutable_inputs(1)->set_token_ids(0, 99);
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    EXPECT_FALSE(prefill.BatchGenerateCall(&context, &batch, &response).ok());
    EXPECT_TRUE(prefill_engine->made.empty());
    EXPECT_EQ(prefill_engine->batch_calls, 0);
}

TEST_F(PDBatchRpcTest, PrefillRejectsDuplicateKeysAndExpiredDeadline) {
    auto batch = handoff();
    batch.mutable_inputs(1)->mutable_generate_config()->set_unique_key(batch.inputs(0).generate_config().unique_key());
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    EXPECT_EQ(prefill.BatchGenerateCall(&context, &batch, &response).error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    batch = handoff();
    batch.mutable_inputs(1)->mutable_generate_config()->set_timeout_ms(0);
    EXPECT_EQ(prefill.BatchGenerateCall(&context, &batch, &response).error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_EQ(prefill_engine->batch_calls, 0);
}

TEST_F(PDBatchRpcTest, LaterItemFailureCancelsEarlierWaitingStreamAndRemoteBatch) {
    decode_engine->hold          = true;
    decode_engine->fail_id       = 2;
    prefill_engine->hold         = true;
    auto                   batch = request();
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    const auto             start  = currentTimeMs();
    auto                   status = decode.BatchGenerateCall(&context, &batch, &response);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_LT(currentTimeMs() - start, 3000);
    EXPECT_TRUE(decode_engine->made[0]->hasError());
    // The RPC may be cancelled before Prefill enters the handler.
    ASSERT_TRUE(waitUntil([&] { return prefill_rpc->service.completed_calls == prefill_rpc->service.batch_calls; }));
    EXPECT_TRUE(prefill.meta_->getEngineScheduleInfo(0).running_task_info_list.empty());
}

TEST_F(PDBatchRpcTest, PrefillFailureStopsWaitingDecodeWithoutTransferDeadline) {
    decode_engine->hold          = true;
    prefill_engine->fail_id      = 2;
    auto                   batch = request();
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    auto                   status = decode.BatchGenerateCall(&context, &batch, &response);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_TRUE(decode_engine->made[0]->hasError());
    EXPECT_TRUE(decode_engine->made[1]->hasError());
}

TEST_F(PDBatchRpcTest, CancellationOfOuterRpcCancelsBothSides) {
    decode_engine->hold  = true;
    prefill_engine->hold = true;
    BatchRpcHarness decode_rpc(&decode);
    ASSERT_NE(decode_rpc.server, nullptr);
    PrefillServerCaller client("test-client");
    auto call = std::move(client.callPrefillBatch(request(), decode_rpc.address(), currentTimeMs() + 5000).value());
    ASSERT_NE(call, nullptr);
    ASSERT_TRUE(waitUntil([&] {
        (void)call->done();
        return decode_engine->batch_calls > 0 && prefill_engine->batch_calls > 0;
    }));
    call->cancel();
    ASSERT_TRUE(waitUntil([&] { return call->done(); }));
    EXPECT_EQ(call->status().error_code(), grpc::StatusCode::CANCELLED);
    ASSERT_TRUE(
        waitUntil([&] { return decode_rpc.service.completed_calls > 0 && prefill_rpc->service.completed_calls > 0; }));
    EXPECT_TRUE(decode_engine->made[0]->hasError());
    EXPECT_TRUE(prefill_engine->made[0]->hasError());
    EXPECT_TRUE(decode.meta_->getEngineScheduleInfo(0).running_task_info_list.empty());
}

TEST_F(PDBatchRpcTest, IndividualDeadlineIsNotExtendedToBatchDeadline) {
    decode_engine->hold  = true;
    prefill_engine->hold = true;
    auto batch           = request();
    batch.mutable_inputs(0)->mutable_generate_config()->set_timeout_ms(300);
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    const auto             start = currentTimeMs();
    EXPECT_EQ(decode.BatchGenerateCall(&context, &batch, &response).error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_LT(currentTimeMs() - start, 3000);
}

TEST(BatchStreamOutputCollectorTest, ConcatenatesDeltasAndPreservesOptionalOutputs) {
    BatchStreamOutputCollector collector;
    auto                       first = chunk(1, {10, 11}, false);
    auto&                      item  = first.generate_outputs[0];
    item.loss                        = torch::tensor({0.5f});
    item.logits                      = torch::ones({1, 4});
    item.hidden_states               = torch::ones({1, 2});
    item.all_hidden_states           = torch::ones({2, 2});
    PromptLogitsOutput prompt;
    prompt.start_pos            = 3;
    item.prompt_logits          = prompt;
    item.aux_info.softmax_probs = torch::tensor({0.1f, 0.2f});
    ASSERT_TRUE(collector.add(std::move(first)).ok());
    auto last                                       = chunk(1, {12}, true);
    last.generate_outputs[0].aux_info.softmax_probs = torch::tensor({0.3f});
    ASSERT_TRUE(collector.add(std::move(last)).ok());
    auto output = collector.finish();
    ASSERT_EQ(output.generate_outputs.size(), 1);
    const auto& result = output.generate_outputs[0];
    EXPECT_TRUE(torch::equal(result.output_ids, torch::tensor({{10, 11, 12}}, torch::kInt32)));
    EXPECT_TRUE(result.finished);
    EXPECT_EQ(result.aux_info.step_output_len, 3);
    ASSERT_TRUE(result.loss.has_value());
    EXPECT_TRUE(result.logits.has_value());
    EXPECT_TRUE(result.hidden_states.has_value());
    EXPECT_TRUE(result.all_hidden_states.has_value());
    ASSERT_TRUE(result.prompt_logits.has_value());
    EXPECT_EQ(result.prompt_logits->start_pos, 3);
    EXPECT_TRUE(torch::allclose(*result.aux_info.softmax_probs, torch::tensor({0.1f, 0.2f, 0.3f})));
}

TEST(BatchStreamOutputCollectorTest, RejectsEmptyOutputAndChangedOutputCount) {
    BatchStreamOutputCollector collector;
    EXPECT_FALSE(collector.add(GenerateOutputs{}).ok());
    EXPECT_TRUE(collector.add(chunk(1, {1}, false)).ok());
    auto changed = chunk(1, {2}, true);
    changed.generate_outputs.push_back(changed.generate_outputs.front());
    EXPECT_FALSE(collector.add(std::move(changed)).ok());
}

TEST(FirstErrorTest, LaterCleanupCannotReplaceCauseAndCopiesRetainOrder) {
    FirstError first, later;
    EXPECT_TRUE(first.record(ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "Prefill VIT failed")));
    EXPECT_FALSE(first.record(ErrorInfo(ErrorCode::CANCELLED, "cleanup cancelled")));
    later.record(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "later deadline"));
    const auto selected = FirstError::earlier(later.snapshot(), first.snapshot());
    EXPECT_EQ(selected.error.code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_EQ(selected.error.ToString(), "Prefill VIT failed");
    EXPECT_EQ(selected.order, first.snapshot().order);
}

TEST(RpcFirstErrorTest, ApplicationCodesRoundTripWithoutNarrowing) {
    for (auto code : {ErrorCode::GENERATE_TIMEOUT,
                      ErrorCode::MALLOC_FAILED,
                      ErrorCode::MM_PROCESS_ERROR,
                      ErrorCode::EXCEEDS_KV_CACHE_MAX_LEN,
                      ErrorCode::P2P_CONNECTOR_WORKER_READ_BUFFER_MISMATCH}) {
        EXPECT_EQ(transRPCErrorCode(transErrorCodeToRPC(code)), code);
        const ErrorInfo original(code, "original cause");
        const auto      decoded = errorInfoFromGrpcStatus(grpcStatusFromErrorInfo(original), "relay");
        EXPECT_EQ(decoded.code(), code);
        EXPECT_EQ(decoded.ToString(), original.ToString());
    }
}

TEST_F(PDBatchRpcTest, BatchSelectsFirstFailedStreamInsteadOfLowestIndex) {
    auto a = decode_engine->makeStream(QueryConverter::transQuery(&request().inputs(0)));
    auto b = decode_engine->makeStream(QueryConverter::transQuery(&request().inputs(1)));
    b->reportError(ErrorCode::MM_PROCESS_ERROR, "first VIT failure item 1");
    a->reportError(ErrorCode::MALLOC_FAILED, "later allocation failure item 0");
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    const auto             status = decode.pollBatchStreamOutput(&context, {a, b}, &response);
    const auto             error  = errorInfoFromGrpcStatus(status);
    EXPECT_EQ(error.code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_EQ(error.ToString(), "first VIT failure item 1");
}

TEST_F(PDBatchRpcTest, RemoteFirstFailureSurvivesLocalFailureAndCancellation) {
    auto       batch  = request();
    auto       stream = decode_engine->makeStream(QueryConverter::transQuery(&batch.inputs(0)));
    FirstError remote;
    remote.record(ErrorInfo(ErrorCode::INVALID_PARAMS, "Prefill plan digest mismatch"));
    stream->reportError(ErrorCode::CANCELLED, "later local cancellation");
    grpc::ServerContext    context;
    BatchGenerateOutputsPB response;
    auto                   status = decode.pollBatchStreamOutput(&context, {stream}, &response, [&](bool& done) {
        done = true;
        return remote.snapshot();
    });
    EXPECT_EQ(errorInfoFromGrpcStatus(status).code(), ErrorCode::INVALID_PARAMS);
    EXPECT_EQ(errorInfoFromGrpcStatus(status).ToString(), "Prefill plan digest mismatch");
}

TEST_F(PDBatchRpcTest, SingleOutputWaitReportsPrefillFailureWithoutWaitingForToken) {
    auto       batch  = request();
    auto       stream = decode_engine->makeStream(QueryConverter::transQuery(&batch.inputs(0)));
    FirstError remote;
    remote.record(ErrorInfo(ErrorCode::MM_PROCESS_ERROR, "Prefill preprocessing failed"));
    grpc::ServerContext context;
    auto                status = decode.pollStreamOutput(&context, "single-error", nullptr, stream, [&](bool& done) {
        done = true;
        return remote.snapshot();
    });
    EXPECT_EQ(errorInfoFromGrpcStatus(status).code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_EQ(errorInfoFromGrpcStatus(status).ToString(), "Prefill preprocessing failed");
    stream->reportError(ErrorCode::GENERATE_TIMEOUT, "later timeout");
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::MM_PROCESS_ERROR);
}

TEST_F(PDBatchRpcTest, GetPeerInfoFailurePreservesTransportAndPeer) {
    prefill_rpc->service.fail_peer = true;
    PrefillServerCaller caller("first-error-test");
    auto                result = caller.getPrefillPeerInfo("127.0.0.1", prefill_rpc->service.port, 5000);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), ErrorCode::CONNECT_FAILED);
    EXPECT_NE(result.status().ToString().find("peer unavailable"), std::string::npos);
    EXPECT_NE(result.status().ToString().find(prefill_rpc->address()), std::string::npos);
    EXPECT_NE(result.status().ToString().find("grpc_code=14"), std::string::npos);
}

TEST_F(PDBatchRpcTest, BatchCallerPreservesPerItemApplicationCodeBeforeCountMismatch) {
    prefill_rpc->service.response_error = true;
    prefill_rpc->service.wrong_count    = true;
    PrefillServerCaller caller("first-error-test");
    auto                started = caller.callPrefillBatch(handoff(), prefill_rpc->address(), currentTimeMs() + 5000);
    ASSERT_TRUE(started.ok());
    auto call = std::move(started.value());
    ASSERT_TRUE(waitUntil([&] { return call->done(); }));
    const auto first = call->firstError();
    call->cancel();
    EXPECT_EQ(first.error.code(), ErrorCode::MM_PROCESS_ERROR);
    EXPECT_NE(first.error.ToString().find("injected result error"), std::string::npos);
    EXPECT_EQ(call->firstError().order, first.order);
}

}  // namespace
}  // namespace rtp_llm
