#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/model_rpc/PrefillBatchRpcServer.h"

namespace rtp_llm {
namespace {

class PartialEnqueueEngine: public EngineBase {
public:
    PartialEnqueueEngine(): EngineBase(EngineInitParams()) {}

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void enqueue(std::shared_ptr<GenerateStream>&) override {}
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>&) override {
        return {enqueue_successes, streams};
    }
    absl::Status stop() override {
        return absl::OkStatus();
    }
    absl::StatusOr<GenerateStreamPtr> preRun(const std::shared_ptr<GenerateInput>&, preRunMode) override {
        return absl::UnimplementedError("unused in test");
    }
    KVCacheInfo getCacheStatusInfo(int64_t, bool) override {
        return KVCacheInfo();
    }

    std::vector<bool>              enqueue_successes;
    std::vector<GenerateStreamPtr> streams;
};

class TestPrefillBatchRpcServer: public PrefillBatchRpcServer {
public:
    grpc::Status EnqueueGroup(grpc::ServerContext*,
                              const EnqueueGroupRequestPB* request,
                              EnqueueBatchResponsePB*      response) override {
        ++enqueue_group_calls;
        captured_group_request = *request;
        response->set_batch_id(request->batch_id());
        const int result_count = request->requests_size() - (omit_last_result ? 1 : 0);
        for (int i = 0; i < result_count; ++i) {
            const auto& group_input = request->requests(i);
            if (group_input.has_input()) {
                response->add_successes()->set_request_id(group_input.input().request_id());
            } else {
                auto* error = response->add_errors();
                error->set_request_id(0);
                error->mutable_error_info()->set_error_code(grpc::StatusCode::INVALID_ARGUMENT);
                error->mutable_error_info()->set_error_message("missing input");
            }
        }
        return grpc::Status::OK;
    }

    void setParallelism(int64_t dp_size, int64_t dp_rank) {
        maga_init_params_.parallelism_config.dp_size = dp_size;
        maga_init_params_.parallelism_config.dp_rank = dp_rank;
    }

    int                   enqueue_group_calls = 0;
    bool                  omit_last_result    = false;
    EnqueueGroupRequestPB captured_group_request;
};

EnqueueBatchExternalInputPB* addInput(EnqueueBatchDpSlotPB* slot, int64_t request_id) {
    auto* external_input = slot->add_requests();
    external_input->mutable_input()->set_request_id(request_id);
    return external_input;
}

std::set<int64_t> successIds(const EnqueueBatchResponsePB& response) {
    std::set<int64_t> ids;
    for (const auto& success : response.successes()) {
        ids.insert(success.request_id());
    }
    return ids;
}

std::shared_ptr<GenerateInput> makeGenerateInput(int64_t request_id) {
    auto input             = std::make_shared<GenerateInput>();
    input->request_id      = request_id;
    input->begin_time_us   = autil::TimeUtility::currentTimeInMicroSeconds();
    input->input_ids       = torch::tensor({1, 2, 3}, torch::kInt32);
    input->generate_config = std::make_shared<GenerateConfig>();
    return input;
}

GenerateStreamPtr makeGenerateStream(const std::shared_ptr<GenerateInput>& input) {
    ModelConfig model_config;
    model_config.max_seq_len = 128;
    RuntimeConfig runtime_config;
    return std::make_shared<NormalGenerateStream>(
        input, model_config, runtime_config, ResourceContext{}, /*metrics_reporter=*/nullptr);
}

void buildReadySlots(PrefillBatchRpcServer&                         server,
                     const std::vector<int64_t>&                    request_ids,
                     std::vector<PrefillBatchRpcServer::BatchSlot>& slots,
                     std::vector<PrefillBatchRpcServer::ReadySlot>& ready_slots) {
    slots.resize(request_ids.size());
    ready_slots.reserve(request_ids.size());
    for (size_t i = 0; i < request_ids.size(); ++i) {
        const auto request_id = request_ids[i];
        auto&      slot       = slots[i];
        slot.input            = std::make_shared<GenerateInputPB>();
        slot.input->set_request_id(request_id);
        RPCContext rpc_context{slot.input.get(), nullptr};
        slot.prefill_context                 = std::make_unique<PrefillGenerateContext>(&server.resource(),
                                                                        rpc_context,
                                                                        /*timeout_ms=*/0,
                                                                        /*server_context=*/nullptr,
                                                                        server.metrics_reporter_,
                                                                        server.meta_);
        slot.prefill_context->generate_input = makeGenerateInput(request_id);
        auto entry                           = server.response_registry_.reserve(request_id);
        ASSERT_NE(entry, nullptr);
        ready_slots.push_back(PrefillBatchRpcServer::ReadySlot{&slot, std::move(entry)});
    }
}

TEST(PrefillBatchRpcServerTest, FlattensLocalSlotsAndPreservesMissingInputResult) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);

    EnqueueBatchRequestPB request;
    request.set_batch_id(101);
    auto* first_slot = request.add_dp_slots();
    first_slot->set_dp_rank(0);
    addInput(first_slot, 11);
    first_slot->add_requests();
    auto* second_slot = request.add_dp_slots();
    second_slot->set_dp_rank(0);
    addInput(second_slot, 12);

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(nullptr, &request, &response).ok());

    EXPECT_EQ(server.enqueue_group_calls, 1);
    EXPECT_EQ(server.captured_group_request.batch_id(), 101);
    EXPECT_EQ(server.captured_group_request.dp_rank(), 0);
    ASSERT_EQ(server.captured_group_request.requests_size(), 3);
    EXPECT_EQ(server.captured_group_request.requests(0).input().request_id(), 11);
    EXPECT_FALSE(server.captured_group_request.requests(1).has_input());
    EXPECT_EQ(server.captured_group_request.requests(2).input().request_id(), 12);
    EXPECT_EQ(response.batch_id(), 101);
    EXPECT_EQ(successIds(response), (std::set<int64_t>{11, 12}));
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 0);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(response.successes_size() + response.errors_size(), 3);
}

TEST(PrefillBatchRpcServerTest, RejectsInvalidRankWithoutBlockingLocalRequests) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);

    EnqueueBatchRequestPB request;
    request.set_batch_id(102);
    auto* local_slot = request.add_dp_slots();
    local_slot->set_dp_rank(0);
    addInput(local_slot, 21);
    auto* invalid_slot = request.add_dp_slots();
    invalid_slot->set_dp_rank(1);
    addInput(invalid_slot, 22);

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(nullptr, &request, &response).ok());

    EXPECT_EQ(server.enqueue_group_calls, 1);
    ASSERT_EQ(server.captured_group_request.requests_size(), 1);
    EXPECT_EQ(server.captured_group_request.requests(0).input().request_id(), 21);
    EXPECT_EQ(successIds(response), (std::set<int64_t>{21}));
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 22);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(response.successes_size() + response.errors_size(), 2);
}

TEST(PrefillBatchRpcServerTest, RejectsWholeBatchWhenRequestIdIsDuplicatedAcrossSlots) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);

    EnqueueBatchRequestPB request;
    request.set_batch_id(103);
    auto* local_slot = request.add_dp_slots();
    local_slot->set_dp_rank(0);
    addInput(local_slot, 31);
    local_slot->add_requests();
    auto* invalid_slot = request.add_dp_slots();
    invalid_slot->set_dp_rank(9);
    addInput(invalid_slot, 31);

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(nullptr, &request, &response).ok());

    EXPECT_EQ(server.enqueue_group_calls, 0);
    EXPECT_EQ(response.successes_size(), 0);
    ASSERT_EQ(response.errors_size(), 3);
    EXPECT_EQ(response.errors(0).request_id(), 31);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::ALREADY_EXISTS);
    EXPECT_EQ(response.errors(1).request_id(), 0);
    EXPECT_EQ(response.errors(1).error_info().error_code(), grpc::StatusCode::INVALID_ARGUMENT);
    EXPECT_EQ(response.errors(2).request_id(), 31);
    EXPECT_EQ(response.errors(2).error_info().error_code(), grpc::StatusCode::ALREADY_EXISTS);
}

TEST(PrefillBatchRpcServerTest, FailsFastWhenMultiDpIsConfigured) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/2, /*dp_rank=*/0);
    EnqueueBatchRequestPB  request;
    EnqueueBatchResponsePB response;

    EXPECT_ANY_THROW(server.EnqueueBatch(nullptr, &request, &response));
    EXPECT_EQ(server.enqueue_group_calls, 0);
}

TEST(PrefillBatchRpcServerTest, FailsFastWhenEnqueueGroupOmitsAResult) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);
    server.omit_last_result = true;
    EnqueueBatchRequestPB request;
    auto*                 slot = request.add_dp_slots();
    slot->set_dp_rank(0);
    addInput(slot, 51);
    addInput(slot, 52);
    EnqueueBatchResponsePB response;

    EXPECT_ANY_THROW(server.EnqueueBatch(nullptr, &request, &response));
    EXPECT_EQ(server.enqueue_group_calls, 1);
    EXPECT_EQ(response.successes_size() + response.errors_size(), 1);
}

TEST(PrefillBatchRpcServerTest, AdmitGroupStripsLegacySchedulerMetadata) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);

    EnqueueGroupRequestPB request;
    request.set_dp_rank(0);
    auto* input = request.add_requests()->mutable_input();
    input->set_request_id(61);
    input->set_group_size(2);
    input->mutable_group_id()->set_value(7);
    EnqueueBatchResponsePB                        response;
    std::vector<PrefillBatchRpcServer::BatchSlot> slots;

    ASSERT_TRUE(server.admitGroup(&request, &response, slots).ok());
    ASSERT_EQ(slots.size(), 1);
    EXPECT_EQ(slots[0].input->group_size(), 0);
    EXPECT_FALSE(slots[0].input->has_group_id());
}

TEST(PrefillBatchRpcServerTest, PartialSchedulerRejectionCleansRejectedPrefillResources) {
    PrefillBatchRpcServer server;
    server.meta_   = std::make_shared<RpcServerRuntimeMeta>();
    auto engine    = std::make_shared<PartialEnqueueEngine>();
    server.engine_ = engine;

    std::vector<PrefillBatchRpcServer::BatchSlot> slots(2);
    std::vector<PrefillBatchRpcServer::ReadySlot> ready_slots;
    buildReadySlots(server, {1001, 1002}, slots, ready_slots);

    engine->streams = {
        makeGenerateStream(slots[0].prefill_context->generate_input),
        makeGenerateStream(slots[1].prefill_context->generate_input),
    };
    engine->streams[1]->reportError(ErrorCode::MALLOC_FAILED, "scheduler rejected request");
    engine->enqueue_successes  = {true, false};
    auto rejected_cancel_state = slots[1].prefill_context->cancel_state;

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.enqueueGroupStreams(ready_slots, &response).ok());

    ASSERT_EQ(ready_slots.size(), 1);
    EXPECT_EQ(slots[0].input->group_size(), 0);
    EXPECT_EQ(slots[1].input->group_size(), 0);
    EXPECT_EQ(engine->streams[0]->groupSize(), 1);
    EXPECT_EQ(engine->streams[1]->groupSize(), 1);
    EXPECT_EQ(ready_slots[0].slot, &slots[0]);
    EXPECT_NE(slots[0].prefill_context, nullptr);
    EXPECT_EQ(slots[1].prefill_context, nullptr);
    EXPECT_TRUE(rejected_cancel_state->load());
    EXPECT_EQ(engine->streams[1]->statusInfo().code(), ErrorCode::MALLOC_FAILED);
    EXPECT_EQ(server.response_registry_.size(), 1);
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 1002);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);

    auto schedule_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(schedule_info.running_task_info_list.size(), 1);
    EXPECT_EQ(schedule_info.running_task_info_list[0].request_id, 1001);
    ASSERT_EQ(schedule_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(schedule_info.finished_task_info_list[0].request_id, 1002);
    EXPECT_EQ(schedule_info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::MALLOC_FAILED));

    server.response_registry_.abort(1001, ready_slots[0].entry);
    slots[0].prefill_context->cancel_state->store(true);
    slots[0].prefill_context.reset();
    EXPECT_EQ(server.response_registry_.size(), 0);
}

TEST(PrefillBatchRpcServerTest, FailsFastWhenEnqueueMultipleReordersStreams) {
    PrefillBatchRpcServer server;
    server.meta_   = std::make_shared<RpcServerRuntimeMeta>();
    auto engine    = std::make_shared<PartialEnqueueEngine>();
    server.engine_ = engine;

    std::vector<PrefillBatchRpcServer::BatchSlot> slots;
    std::vector<PrefillBatchRpcServer::ReadySlot> ready_slots;
    buildReadySlots(server, {2001, 2002}, slots, ready_slots);

    engine->streams = {
        makeGenerateStream(slots[1].prefill_context->generate_input),
        makeGenerateStream(slots[0].prefill_context->generate_input),
    };
    engine->enqueue_successes = {true, true};

    EnqueueBatchResponsePB response;
    EXPECT_ANY_THROW(server.enqueueGroupStreams(ready_slots, &response));

    server.response_registry_.abort(2001, ready_slots[0].entry);
    server.response_registry_.abort(2002, ready_slots[1].entry);
}

}  // namespace
}  // namespace rtp_llm
