#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <set>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/model_rpc/PrefillBatchRpcServer.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm {
namespace {

class PartialEnqueueEngine: public EngineBase {
public:
    // EngineBase requires a scheduler; this inert implementation keeps the
    // batch-enqueue tests focused on their per-context cancellation path.
    class NoopScheduler: public SchedulerBase {
    public:
        absl::Status enqueue(const GenerateStreamPtr&) override {
            return absl::OkStatus();
        }
        std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
        enqueueGroup(const std::vector<GenerateStreamPtr>&) override {
            return {{}, {}};
        }
        absl::StatusOr<std::list<GenerateStreamPtr>> schedule() override {
            return std::list<GenerateStreamPtr>();
        }
        absl::Status stop() override {
            return absl::OkStatus();
        }
        bool empty() override {
            return true;
        }
        int64_t lastScheduleTime() override {
            return 0;
        }
        int64_t onflightStreams() override {
            return 0;
        }
    };

    PartialEnqueueEngine(): EngineBase(EngineInitParams()) {
        scheduler_ = std::make_unique<NoopScheduler>();
    }

    std::shared_ptr<GenerateStream> enqueue(const std::shared_ptr<GenerateInput>&) override {
        return nullptr;
    }
    void enqueue(std::shared_ptr<GenerateStream>&) override {}
    std::pair<std::vector<bool>, std::vector<GenerateStreamPtr>>
    enqueueMultiple(const std::vector<std::shared_ptr<GenerateInput>>&) override {
        ++enqueue_multiple_calls;
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
    int                            enqueue_multiple_calls{0};
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

    grpc::Status outwardStatus(PrefillGenerateContext& context, const grpc::Status& fallback) {
        return preferPriorityPreemption(context, fallback);
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

std::shared_ptr<DeferredPrefillContext> makeDeferred(PrefillBatchRpcServer& server, int64_t request_id) {
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    auto input   = std::make_shared<GenerateInputPB>();
    input->set_request_id(request_id);
    input->mutable_group_id()->set_value(99);
    RPCContext rpc_context{input.get(), nullptr};
    auto       context  = std::make_unique<PrefillGenerateContext>(&server.resource(),
                                                            rpc_context,
                                                            /*timeout_ms=*/0,
                                                            /*server_context=*/nullptr,
                                                            server.metrics_reporter_,
                                                            server.meta_);
    auto       deferred = std::make_shared<DeferredPrefillContext>();
    deferred->context   = std::move(context);
    deferred->input     = std::move(input);
    return deferred;
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
        auto       context      = std::make_unique<PrefillGenerateContext>(&server.resource(),
                                                                rpc_context,
                                                                /*timeout_ms=*/0,
                                                                /*server_context=*/nullptr,
                                                                server.metrics_reporter_,
                                                                server.meta_);
        context->generate_input = makeGenerateInput(request_id);
        slot.deferred           = std::make_shared<DeferredPrefillContext>();
        slot.deferred->context  = std::move(context);
        slot.deferred->input    = slot.input;
        ready_slots.push_back(PrefillBatchRpcServer::ReadySlot{&slot, slot.deferred});
    }
}

TEST(PrefillBatchRpcServerTest, FlattensLocalSlotsAndPropagatesFetchLease) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);

    EnqueueBatchRequestPB request;
    request.set_batch_id(101);
    request.set_fetch_attach_timeout_ms(4321);
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
    EXPECT_EQ(server.captured_group_request.fetch_attach_timeout_ms(), 4321);
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
    auto*                 local_slot = request.add_dp_slots();
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
    auto*                 first = request.add_dp_slots();
    first->set_dp_rank(0);
    addInput(first, 31);
    auto* second = request.add_dp_slots();
    second->set_dp_rank(0);
    addInput(second, 31);

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.EnqueueBatch(nullptr, &request, &response).ok());
    EXPECT_EQ(server.enqueue_group_calls, 0);
    EXPECT_EQ(response.successes_size(), 0);
    ASSERT_EQ(response.errors_size(), 2);
    EXPECT_EQ(response.errors(0).request_id(), 31);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::ALREADY_EXISTS);
    EXPECT_EQ(response.errors(1).request_id(), 31);
    EXPECT_EQ(response.errors(1).error_info().error_code(), grpc::StatusCode::ALREADY_EXISTS);
}

TEST(PrefillBatchRpcServerTest, FailsFastWhenMultiDpIsConfigured) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/2, /*dp_rank=*/0);
    EnqueueBatchRequestPB  request;
    EnqueueBatchResponsePB response;
    EXPECT_DEATH({ (void)server.EnqueueBatch(nullptr, &request, &response); }, "");
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
    EXPECT_DEATH({ (void)server.EnqueueBatch(nullptr, &request, &response); }, "");
}

TEST(PrefillBatchRpcServerTest, AdmitGroupCopiesBatchMetadataAndLease) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);
    EnqueueGroupRequestPB request;
    request.set_batch_id(7);
    request.set_dp_rank(0);
    request.set_fetch_attach_timeout_ms(999);
    request.add_requests()->mutable_input()->set_request_id(61);
    request.add_requests()->mutable_input()->set_request_id(62);
    EnqueueBatchResponsePB                        response;
    std::vector<PrefillBatchRpcServer::BatchSlot> slots;

    ASSERT_TRUE(server.admitGroup(&request, &response, slots).ok());
    ASSERT_EQ(slots.size(), 2);
    EXPECT_EQ(slots[0].input->group_size(), 2);
    ASSERT_TRUE(slots[0].input->has_group_id());
    EXPECT_EQ(slots[0].input->group_id().value(), 7);
    EXPECT_EQ(slots[0].fetch_attach_timeout_ms, 999);
    EXPECT_EQ(slots[1].input->group_size(), 2);
    ASSERT_TRUE(slots[1].input->has_group_id());
    EXPECT_EQ(slots[1].input->group_id().value(), 7);
    EXPECT_EQ(slots[1].fetch_attach_timeout_ms, 999);
}

TEST(PrefillBatchRpcServerTest, ContextCapturesAdmittedEnvelopeBeforeQueryConversion) {
    TestPrefillBatchRpcServer server;
    server.setParallelism(/*dp_size=*/1, /*dp_rank=*/0);
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    EnqueueGroupRequestPB request;
    request.set_batch_id(8);
    request.set_dp_rank(0);
    request.add_requests()->mutable_input()->set_request_id(63);
    EnqueueBatchResponsePB                        response;
    std::vector<PrefillBatchRpcServer::BatchSlot> slots;

    ASSERT_TRUE(server.admitGroup(&request, &response, slots).ok());
    ASSERT_EQ(slots.size(), 1);
    server.buildSlotContexts(slots);
    ASSERT_NE(slots[0].deferred, nullptr);
    ASSERT_EQ(slots[0].deferred->context->generate_input, nullptr);

    EXPECT_EQ(slots[0].deferred->context->requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);
    auto canceling = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceling.running_task_info_list.size(), 1);
    EXPECT_EQ(canceling.running_task_info_list[0].request_id, 63);
    EXPECT_EQ(canceling.running_task_info_list[0].batch_id, 8);
    EXPECT_EQ(canceling.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);
}

TEST(PrefillBatchRpcServerTest, PartialSchedulerRejectionCleansRejectedPrefillResources) {
    PrefillBatchRpcServer server;
    server.meta_   = std::make_shared<RpcServerRuntimeMeta>();
    auto engine    = std::make_shared<PartialEnqueueEngine>();
    server.engine_ = engine;

    std::vector<PrefillBatchRpcServer::BatchSlot> slots;
    std::vector<PrefillBatchRpcServer::ReadySlot> ready_slots;
    buildReadySlots(server, {1001, 1002}, slots, ready_slots);

    engine->streams = {
        makeGenerateStream(ready_slots[0].deferred->context->generate_input),
        makeGenerateStream(ready_slots[1].deferred->context->generate_input),
    };
    engine->streams[1]->reportError(ErrorCode::MALLOC_FAILED, "scheduler rejected request");
    engine->enqueue_successes  = {true, false};
    auto accepted_deferred     = ready_slots[0].deferred;
    auto rejected_cancel_state = ready_slots[1].deferred->context->cancel_state;

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.enqueueGroupStreams(ready_slots, &response).ok());

    ASSERT_EQ(ready_slots.size(), 1);
    EXPECT_EQ(ready_slots[0].slot, &slots[0]);
    EXPECT_EQ(ready_slots[0].deferred, accepted_deferred);
    EXPECT_TRUE(rejected_cancel_state->load());
    EXPECT_EQ(engine->streams[1]->statusInfo().code(), ErrorCode::MALLOC_FAILED);
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 1002);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);

    auto schedule_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(schedule_info.running_task_info_list.size(), 1);
    EXPECT_EQ(schedule_info.running_task_info_list[0].request_id, 1001);
    ASSERT_EQ(schedule_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(schedule_info.finished_task_info_list[0].request_id, 1002);
    EXPECT_EQ(schedule_info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::MALLOC_FAILED));

    accepted_deferred->context->cancel_state->store(true);
    accepted_deferred.reset();
}

TEST(PrefillBatchRpcServerTest, LatchedPriorityCancelBeforeEnqueuePreservesRaw8429) {
    PrefillBatchRpcServer server;
    server.meta_   = std::make_shared<RpcServerRuntimeMeta>();
    auto engine    = std::make_shared<PartialEnqueueEngine>();
    server.engine_ = engine;

    std::vector<PrefillBatchRpcServer::BatchSlot> slots;
    std::vector<PrefillBatchRpcServer::ReadySlot> ready_slots;
    buildReadySlots(server, {1013}, slots, ready_slots);
    ASSERT_EQ(ready_slots[0].deferred->context->requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.enqueueGroupStreams(ready_slots, &response).ok());

    EXPECT_TRUE(ready_slots.empty());
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 1013);
    EXPECT_EQ(response.errors(0).error_info().error_code(), static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(PrefillBatchRpcServerTest, RejectsWhenEnqueueMultipleReordersStreams) {
    PrefillBatchRpcServer server;
    server.meta_   = std::make_shared<RpcServerRuntimeMeta>();
    auto engine    = std::make_shared<PartialEnqueueEngine>();
    server.engine_ = engine;

    std::vector<PrefillBatchRpcServer::BatchSlot> slots;
    std::vector<PrefillBatchRpcServer::ReadySlot> ready_slots;
    buildReadySlots(server, {2001, 2002}, slots, ready_slots);

    engine->streams = {
        makeGenerateStream(ready_slots[1].deferred->context->generate_input),
        makeGenerateStream(ready_slots[0].deferred->context->generate_input),
    };
    engine->enqueue_successes = {true, true};

    EnqueueBatchResponsePB response;
    const auto             status = server.enqueueGroupStreams(ready_slots, &response);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
    EXPECT_NE(status.error_message().find("result order mismatch"), std::string::npos);
    for (auto& ready_slot : ready_slots) {
        server.rejectSlot(ready_slot, status, &response);
    }
    EXPECT_EQ(response.errors_size(), 2);
}

TEST(PrefillBatchRpcServerTest, DoesNotAckSuccessAfterRequestDeadlineExpires) {
    PrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();
    std::vector<PrefillBatchRpcServer::BatchSlot> slots;
    std::vector<PrefillBatchRpcServer::ReadySlot> ready_slots;
    buildReadySlots(server, {2010}, slots, ready_slots);
    auto deferred                            = ready_slots[0].deferred;
    deferred->context->request_timeout_ms    = 1;
    deferred->context->request_begin_time_us = currentTimeUs() - 10 * 1000;
    auto cancel_state                        = deferred->context->cancel_state;
    ASSERT_TRUE(server.deferred_contexts_->store(2010, deferred).ok());

    EnqueueBatchResponsePB response;
    server.publishSlot(ready_slots[0], &response);

    EXPECT_EQ(response.successes_size(), 0);
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
    EXPECT_TRUE(cancel_state->load());
    EXPECT_EQ(server.deferred_contexts_->size(), 0);
}

TEST(PrefillBatchRpcServerTest, DeferredContextMapTakesAndRemovesContext) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3001);

    ASSERT_TRUE(contexts->store(3001, deferred).ok());
    ASSERT_TRUE(contexts->armTtl(3001, deferred, std::chrono::seconds(1)).ok());
    EXPECT_FALSE(deferred->finishOperation());
    EXPECT_EQ(contexts->size(), 1);
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3001, claimed).ok());
    EXPECT_EQ(claimed, deferred);
    EXPECT_EQ(contexts->size(), 0);

    std::shared_ptr<DeferredPrefillContext> duplicate_fetch;
    const auto                              duplicate_fetch_status = contexts->take(3001, duplicate_fetch);
    EXPECT_EQ(duplicate_fetch_status.error_code(), grpc::StatusCode::NOT_FOUND);
    EXPECT_EQ(duplicate_fetch, nullptr);
}

TEST(PrefillBatchRpcServerTest, ActiveOperationOwnsPriorityFinalizationOnExit) {
    DeferredPrefillContext deferred;
    EXPECT_FALSE(deferred.requestPriorityFinalization());
    EXPECT_TRUE(deferred.finishOperation());
    EXPECT_FALSE(deferred.finishOperation());
}

TEST(PrefillBatchRpcServerTest, IdleContextCanBeFinalizedWithoutOperationWaiter) {
    DeferredPrefillContext deferred;
    EXPECT_FALSE(deferred.finishOperation());
    EXPECT_TRUE(deferred.requestPriorityFinalization());
    EXPECT_FALSE(deferred.requestPriorityFinalization());
}

TEST(PrefillBatchRpcServerTest, PriorityPreemptionCancelRemainsRoutableAfterFetchTakesContext) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3009);

    ASSERT_TRUE(contexts->registerActive(3009, deferred).ok());
    ASSERT_TRUE(contexts->store(3009, deferred).ok());
    EXPECT_FALSE(deferred->finishOperation());
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3009, claimed).ok());

    ASSERT_EQ(contexts->cancelByPriorityPreemption(3009), PriorityCancelResult::ACCEPTED);
    EXPECT_TRUE(claimed->context->cancel_state->load());
    EXPECT_TRUE(claimed->context->isPriorityPreempted());

    contexts->finish(3009, claimed.get());
    // A retry joins the already-installed weak-ACK latch.
    EXPECT_EQ(contexts->cancelByPriorityPreemption(3009), PriorityCancelResult::ACCEPTED);
}

TEST(PrefillBatchRpcServerTest, LateCancelAfterNaturalFinishDoesNotInstallAbsentFence) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3025);

    ASSERT_TRUE(contexts->registerActive(3025, deferred).ok());
    ASSERT_TRUE(contexts->store(3025, deferred).ok());
    EXPECT_FALSE(deferred->finishOperation());
    std::shared_ptr<DeferredPrefillContext> fetched;
    ASSERT_TRUE(contexts->take(3025, fetched).ok());
    contexts->finish(3025, fetched.get());

    EXPECT_EQ(contexts->cancelByPriorityPreemption(3025), PriorityCancelResult::NOT_FOUND);

    // NOT_FOUND is deliberately conservative: unlike TOMBSTONED it does not
    // claim that cancel-before-enqueue was fenced, and therefore must not
    // poison a later registration with a synthetic 8429.
    auto replacement = makeDeferred(server, 3025);
    EXPECT_TRUE(contexts->registerActive(3025, replacement).ok());
}

TEST(PrefillBatchRpcServerTest, NaturalFinishAndCancelHaveOneLinearizedOutcome) {
    PrefillBatchRpcServer server;

    for (int64_t request_id = 3200; request_id < 3300; ++request_id) {
        auto contexts = std::make_shared<DeferredPrefillContextMap>();
        auto deferred = makeDeferred(server, request_id);
        ASSERT_TRUE(contexts->registerActive(request_id, deferred).ok());

        std::atomic<int>     ready{0};
        std::atomic<bool>    start{false};
        PriorityCancelResult cancel_result = PriorityCancelResult::TOMBSTONED;
        std::thread          finish_thread([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            contexts->finish(request_id, deferred.get());
        });
        std::thread          cancel_thread([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            cancel_result = contexts->cancelByPriorityPreemption(request_id);
        });
        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);
        finish_thread.join();
        cancel_thread.join();

        EXPECT_NE(cancel_result, PriorityCancelResult::TOMBSTONED);
        EXPECT_TRUE(cancel_result == PriorityCancelResult::ACCEPTED
                    || cancel_result == PriorityCancelResult::NOT_FOUND);
    }
}

TEST(PrefillBatchRpcServerTest, PreparingContextIsVisibleToCancelBeforeStore) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3010);

    ASSERT_TRUE(contexts->registerActive(3010, deferred).ok());
    ASSERT_EQ(contexts->cancelByPriorityPreemption(3010), PriorityCancelResult::ACCEPTED);

    EXPECT_TRUE(deferred->context->isPriorityPreempted());
    EXPECT_TRUE(deferred->context->cancel_state->load());
    auto status_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(status_info.running_task_info_list.size(), 1);
    EXPECT_EQ(status_info.running_task_info_list[0].priority_preemption_progress,
              PriorityPreemptionProgress::CANCELING);
    contexts->finish(3010, deferred.get());
}

TEST(PrefillBatchRpcServerTest, CancelBeforeRegisterInstallsTombstoneAndRejectsEnqueueWith8429) {
    PrefillBatchRpcServer server;
    auto                  engine = std::make_shared<PartialEnqueueEngine>();
    server.engine_               = engine;
    CancelRequestPB request;
    request.set_request_id(3011);
    CancelResponsePB response;

    ASSERT_TRUE(server.Cancel(nullptr, &request, &response).ok());
    ASSERT_EQ(response.status(), CancelStatusPB::CANCEL_STATUS_TOMBSTONED);
    CancelResponsePB retry_response;
    ASSERT_TRUE(server.Cancel(nullptr, &request, &retry_response).ok());
    EXPECT_EQ(retry_response.status(), CancelStatusPB::CANCEL_STATUS_TOMBSTONED);

    std::vector<PrefillBatchRpcServer::BatchSlot> slots;
    PrefillBatchRpcServer::BatchSlot              slot;
    slot.input = std::make_shared<GenerateInputPB>();
    slot.input->set_request_id(3011);
    slots.push_back(std::move(slot));
    EnqueueBatchResponsePB enqueue_response;
    ASSERT_TRUE(server.acceptGroup(std::move(slots), &enqueue_response).ok());

    EXPECT_EQ(engine->enqueue_multiple_calls, 0);
    ASSERT_EQ(enqueue_response.errors_size(), 1);
    EXPECT_EQ(enqueue_response.errors(0).request_id(), 3011);
    EXPECT_EQ(enqueue_response.errors(0).error_info().error_code(),
              static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(PrefillBatchRpcServerTest, CancelRejectsZeroRequestId) {
    PrefillBatchRpcServer server;
    auto                  engine = std::make_shared<PartialEnqueueEngine>();
    server.engine_               = engine;
    CancelRequestPB  request;
    CancelResponsePB response;

    auto status = server.Cancel(nullptr, &request, &response);

    EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(PrefillBatchRpcServerTest, CancelActiveRequestLatches8429WithoutSchedulerIntent) {
    PrefillBatchRpcServer server;
    auto                  engine = std::make_shared<PartialEnqueueEngine>();
    server.engine_               = engine;
    auto deferred                = makeDeferred(server, 3012);
    ASSERT_TRUE(server.deferred_contexts_->registerActive(3012, deferred).ok());
    CancelRequestPB request;
    request.set_request_id(3012);
    CancelResponsePB response;

    ASSERT_TRUE(server.Cancel(nullptr, &request, &response).ok());

    EXPECT_EQ(response.status(), CancelStatusPB::CANCEL_STATUS_ACCEPTED);
    EXPECT_TRUE(deferred->context->isPriorityPreempted());

    // A tombstone retry acknowledges the already-installed weak latch but
    // must not create a scheduler intent.
    CancelResponsePB retry_response;
    ASSERT_TRUE(server.Cancel(nullptr, &request, &retry_response).ok());
    EXPECT_EQ(retry_response.status(), CancelStatusPB::CANCEL_STATUS_ACCEPTED);
    auto status_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(status_info.running_task_info_list.size(), 1);
    EXPECT_EQ(status_info.running_task_info_list[0].priority_preemption_progress,
              PriorityPreemptionProgress::CANCELING);
    server.deferred_contexts_->finish(3012, deferred.get());
}

TEST(PrefillBatchRpcServerTest, TypedPriorityTerminalDowngradesActiveCancelAckToTombstone) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3017);
    ASSERT_TRUE(contexts->registerActive(3017, deferred).ok());

    EXPECT_EQ(contexts->cancelByPriorityPreemption(3017), PriorityCancelResult::ACCEPTED);
    contexts->publishPriorityPreemptionCanceled(3017, deferred.get());

    EXPECT_EQ(contexts->cancelByPriorityPreemption(3017), PriorityCancelResult::TOMBSTONED);
    auto replacement = makeDeferred(server, 3017);
    auto status      = contexts->registerActive(3017, replacement);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
}

TEST(PrefillBatchRpcServerTest, CancelAndRegisterHaveOneLinearizedOutcome) {
    PrefillBatchRpcServer server;

    for (int64_t request_id = 3100; request_id < 3200; ++request_id) {
        auto contexts = std::make_shared<DeferredPrefillContextMap>();
        auto deferred = makeDeferred(server, request_id);

        std::atomic<int>     ready{0};
        std::atomic<bool>    start{false};
        grpc::Status         registration_status;
        PriorityCancelResult cancel_result = PriorityCancelResult::NOT_FOUND;

        std::thread register_thread([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            registration_status = contexts->registerActive(request_id, deferred);
        });
        std::thread cancel_thread([&] {
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            cancel_result = contexts->cancelByPriorityPreemption(request_id);
        });
        while (ready.load(std::memory_order_acquire) != 2) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);
        register_thread.join();
        cancel_thread.join();

        if (registration_status.ok()) {
            EXPECT_EQ(cancel_result, PriorityCancelResult::ACCEPTED);
            EXPECT_TRUE(deferred->context->isPriorityPreempted());
            continue;
        }

        EXPECT_EQ(cancel_result, PriorityCancelResult::TOMBSTONED);
        EXPECT_EQ(registration_status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
        ErrorDetailsPB details;
        ASSERT_TRUE(details.ParseFromString(registration_status.error_details()));
        EXPECT_EQ(details.error_code(), static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
        EXPECT_FALSE(deferred->context->isPriorityPreempted());
    }
}

TEST(PrefillBatchRpcServerTest, CancelStatusWireNumbersRemainBackwardCompatible) {
    EXPECT_EQ(CancelStatusPB::CANCEL_STATUS_UNSPECIFIED, 0);
    EXPECT_EQ(CancelStatusPB::CANCEL_STATUS_ACCEPTED, 1);
    EXPECT_EQ(CancelStatusPB::CANCEL_STATUS_NOT_FOUND, 2);
    EXPECT_EQ(CancelStatusPB::CANCEL_STATUS_TOMBSTONED, 3);
}

TEST(PrefillBatchRpcServerTest, AcceptedPriorityPreemptionOverridesPrepareFailure) {
    TestPrefillBatchRpcServer server;
    auto                      contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                      deferred = makeDeferred(server, 3014);
    ASSERT_TRUE(contexts->registerActive(3014, deferred).ok());

    ASSERT_EQ(contexts->cancelByPriorityPreemption(3014), PriorityCancelResult::ACCEPTED);
    contexts->finish(3014, deferred.get());
    auto           outward = server.outwardStatus(*deferred->context,
                                        grpc::Status(grpc::StatusCode::INTERNAL, "prepareAllocateResource failed"));
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(outward.error_details()));
    EXPECT_EQ(details.error_code(), static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    EXPECT_EQ(contexts->cancelByPriorityPreemption(3014), PriorityCancelResult::ACCEPTED);
}

TEST(PrefillBatchRpcServerTest, OtherTerminalBeforePriorityCancelReturnsNotFound) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3018);
    ASSERT_TRUE(contexts->registerActive(3018, deferred).ok());

    ASSERT_TRUE(deferred->context->tryMarkOtherTerminal());
    bool                                    newly_installed = true;
    std::shared_ptr<DeferredPrefillContext> canceled;
    EXPECT_EQ(contexts->cancelByPriorityPreemption(3018, canceled, &newly_installed), PriorityCancelResult::NOT_FOUND);
    EXPECT_FALSE(newly_installed);
    EXPECT_EQ(canceled, nullptr);
    EXPECT_EQ(deferred->context->terminalCause(), PrefillTerminalCause::OTHER);
}

TEST(PrefillBatchRpcServerTest, PriorityCancelBeforeOtherTerminalPreserves8429) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3019);
    ASSERT_TRUE(contexts->registerActive(3019, deferred).ok());

    bool                                    newly_installed = false;
    std::shared_ptr<DeferredPrefillContext> canceled;
    ASSERT_EQ(contexts->cancelByPriorityPreemption(3019, canceled, &newly_installed), PriorityCancelResult::ACCEPTED);
    EXPECT_TRUE(newly_installed);
    EXPECT_FALSE(deferred->context->tryMarkOtherTerminal());
    EXPECT_EQ(deferred->context->terminalCause(), PrefillTerminalCause::PRIORITY_PREEMPTION);
}

TEST(PrefillBatchRpcServerTest, PriorityAndOtherTerminalBarrierHasExactlyOneWinner) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3020);
    ASSERT_TRUE(contexts->registerActive(3020, deferred).ok());

    std::atomic<int>  ready{0};
    std::atomic<bool> start{false};
    bool              other_won       = false;
    bool              priority_won    = false;
    bool              newly_installed = false;
    std::thread       other_thread([&] {
        ready.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        other_won = deferred->context->tryMarkOtherTerminal();
    });
    std::thread       priority_thread([&] {
        ready.fetch_add(1);
        while (!start.load()) {
            std::this_thread::yield();
        }
        std::shared_ptr<DeferredPrefillContext> canceled;
        priority_won =
            contexts->cancelByPriorityPreemption(3020, canceled, &newly_installed) == PriorityCancelResult::ACCEPTED;
    });
    while (ready.load() != 2) {
        std::this_thread::yield();
    }
    start.store(true);
    other_thread.join();
    priority_thread.join();

    EXPECT_NE(other_won, priority_won);
    EXPECT_EQ(priority_won, newly_installed);
    EXPECT_EQ(deferred->context->terminalCause(),
              priority_won ? PrefillTerminalCause::PRIORITY_PREEMPTION : PrefillTerminalCause::OTHER);
}

TEST(PrefillBatchRpcServerTest, CanceledPrepareSlotCanFinalizeBeforeSiblingLeavesPrepare) {
    PrefillBatchRpcServer server;
    auto                  canceled = makeDeferred(server, 3021);
    auto                  sibling  = makeDeferred(server, 3022);

    EXPECT_EQ(canceled->context->requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);
    EXPECT_FALSE(canceled->requestPriorityFinalization());
    EXPECT_TRUE(canceled->finishOperation());

    // The sibling still owns PREPARE, proving finalizer ownership is per slot.
    EXPECT_FALSE(sibling->tryStartOperation().started);
    EXPECT_FALSE(sibling->finishOperation());
    EXPECT_TRUE(sibling->tryStartOperation().started);
}

TEST(PrefillBatchRpcServerTest, TerminalCauseAloneCannotStartFinalizerBeforeCancelRegistersIt) {
    PrefillBatchRpcServer server;
    auto                  deferred = makeDeferred(server, 3023);

    EXPECT_EQ(deferred->context->requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);
    // The operation may exit after the terminal CAS but before Cancel has
    // registered finalization. It must not claim the finalizer prematurely.
    EXPECT_FALSE(deferred->finishOperation());
    EXPECT_TRUE(deferred->requestPriorityFinalization());
}

TEST(PrefillBatchRpcServerTest, FetchAfterAcceptedPriorityCancelReturns8429Tombstone) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3015);
    ASSERT_TRUE(contexts->registerActive(3015, deferred).ok());
    ASSERT_TRUE(contexts->store(3015, deferred).ok());
    EXPECT_FALSE(deferred->finishOperation());

    ASSERT_EQ(contexts->cancelByPriorityPreemption(3015), PriorityCancelResult::ACCEPTED);

    std::shared_ptr<DeferredPrefillContext> fetched;
    auto                                    status = contexts->take(3015, fetched);
    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
    EXPECT_EQ(fetched, nullptr);
}

TEST(PrefillBatchRpcServerTest, PriorityPreemptionReturnsRaw8429InErrorDetails) {
    TestPrefillBatchRpcServer server;
    auto                      deferred = makeDeferred(server, 3013);
    deferred->context->requestPriorityPreempt();

    auto status =
        server.outwardStatus(*deferred->context, grpc::Status(grpc::StatusCode::CANCELLED, "downstream cancelled"));
    EXPECT_EQ(status.error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    ErrorDetailsPB details;
    ASSERT_TRUE(details.ParseFromString(status.error_details()));
    EXPECT_EQ(details.error_code(), static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(PrefillBatchRpcServerTest, PrefillFinalizerPublishesCanceled8429ExactlyOnce) {
    PrefillBatchRpcServer server;
    auto                  deferred = makeDeferred(server, 3016);
    deferred->context->requestPriorityPreempt();

    // Cancel wins before QueryConverter creates generate_input or a local
    // stream reaches RuntimeMeta. The deferred batch envelope is the only
    // available source for batch identity in this window.
    auto canceling = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceling.running_task_info_list.size(), 1);
    EXPECT_EQ(canceling.running_task_info_list[0].batch_id, 99);
    EXPECT_EQ(canceling.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);

    EXPECT_TRUE(deferred->context->finalizePriorityPreemption());
    EXPECT_TRUE(deferred->context->finalizePriorityPreemption());

    auto status_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(status_info.running_task_info_list.empty());
    ASSERT_EQ(status_info.finished_task_info_list.size(), 1);
    const auto& task = status_info.finished_task_info_list[0];
    EXPECT_EQ(task.request_id, 3016);
    EXPECT_EQ(task.batch_id, 99);
    EXPECT_EQ(task.priority_preemption_progress, PriorityPreemptionProgress::CANCELED);
    EXPECT_EQ(task.error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(PrefillBatchRpcServerTest, PriorityFirstCauseSuppressesOrdinaryDequeueTerminal) {
    PrefillBatchRpcServer server;
    auto                  deferred    = makeDeferred(server, 3024);
    auto                  input       = makeGenerateInput(3024);
    input->group_id                   = 99;
    auto stream                       = makeGenerateStream(input);
    deferred->context->generate_input = input;
    deferred->context->setStream(stream);
    deferred->context->setLocalStreamSchedulerOwned(false);

    ASSERT_EQ(deferred->context->requestPriorityPreempt(), PriorityPreemptionRequestResult::INSTALLED);
    deferred->context->dequeueStreamFromRuntimeMeta();

    auto canceling = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(canceling.running_task_info_list.size(), 1);
    EXPECT_TRUE(canceling.finished_task_info_list.empty());
    EXPECT_EQ(canceling.running_task_info_list[0].batch_id, 99);
    EXPECT_EQ(canceling.running_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELING);

    ASSERT_TRUE(deferred->context->finalizePriorityPreemption());
    auto canceled = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(canceled.running_task_info_list.empty());
    ASSERT_EQ(canceled.finished_task_info_list.size(), 1);
    EXPECT_EQ(canceled.finished_task_info_list[0].batch_id, 99);
    EXPECT_EQ(canceled.finished_task_info_list[0].priority_preemption_progress, PriorityPreemptionProgress::CANCELED);
}

TEST(PrefillBatchRpcServerTest, PriorityFinalizerDoesNotWaitForSchedulerRejectedStream) {
    PrefillBatchRpcServer server;
    auto                  deferred    = makeDeferred(server, 3017);
    auto                  input       = makeGenerateInput(3017);
    auto                  stream      = makeGenerateStream(input);
    deferred->context->generate_input = input;
    deferred->context->setStream(stream);
    deferred->context->setLocalStreamSchedulerOwned(false);
    deferred->context->requestPriorityPreempt();

    EXPECT_TRUE(deferred->context->finalizePriorityPreemption());

    auto status_info = server.meta_->getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_TRUE(status_info.running_task_info_list.empty());
    ASSERT_EQ(status_info.finished_task_info_list.size(), 1);
    EXPECT_EQ(status_info.finished_task_info_list[0].priority_preemption_progress,
              PriorityPreemptionProgress::CANCELED);
    EXPECT_EQ(status_info.finished_task_info_list[0].error_code, static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED));
}

TEST(PrefillBatchRpcServerTest, DeferredContextMapRejectsDuplicateRequestId) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  first    = makeDeferred(server, 3002);
    auto                  second   = makeDeferred(server, 3002);

    ASSERT_TRUE(contexts->store(3002, first).ok());
    const auto duplicate_status = contexts->store(3002, second);
    EXPECT_EQ(duplicate_status.error_code(), grpc::StatusCode::ALREADY_EXISTS);
    EXPECT_EQ(duplicate_status.error_message(), "request already exists in deferred context map");
    ASSERT_TRUE(contexts->armTtl(3002, first, std::chrono::seconds(1)).ok());
    EXPECT_FALSE(first->finishOperation());
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3002, claimed).ok());
    EXPECT_EQ(claimed, first);
}

TEST(PrefillBatchRpcServerTest, ConcurrentStoreAllowsExactlyOneContextPerRequestId) {
    constexpr int                                        kThreadCount = 16;
    PrefillBatchRpcServer                                server;
    auto                                                 contexts = std::make_shared<DeferredPrefillContextMap>();
    std::vector<std::shared_ptr<DeferredPrefillContext>> candidates;
    candidates.reserve(kThreadCount);
    for (int i = 0; i < kThreadCount; ++i) {
        candidates.push_back(makeDeferred(server, 3010));
    }

    std::atomic<int>              ready{0};
    std::atomic<bool>             start{false};
    std::vector<grpc::StatusCode> codes(kThreadCount);
    std::vector<std::thread>      threads;
    threads.reserve(kThreadCount);
    for (int i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([&, i] {
            ready.fetch_add(1);
            while (!start.load()) {
                std::this_thread::yield();
            }
            codes[i] = contexts->store(3010, candidates[i]).error_code();
        });
    }
    while (ready.load() != kThreadCount) {
        std::this_thread::yield();
    }
    start.store(true);
    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(std::count(codes.begin(), codes.end(), grpc::StatusCode::OK), 1);
    EXPECT_EQ(std::count(codes.begin(), codes.end(), grpc::StatusCode::ALREADY_EXISTS), kThreadCount - 1);
    EXPECT_EQ(contexts->size(), 1);
}

TEST(PrefillBatchRpcServerTest, StaleRollbackCannotRemoveAReplacementContext) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  first    = makeDeferred(server, 3011);
    auto                  second   = makeDeferred(server, 3011);

    ASSERT_TRUE(contexts->store(3011, first).ok());
    EXPECT_EQ(contexts->remove(3011, first.get()), first);
    ASSERT_TRUE(contexts->store(3011, second).ok());
    EXPECT_EQ(contexts->remove(3011, first.get()), nullptr);

    ASSERT_TRUE(contexts->armTtl(3011, second, std::chrono::seconds(1)).ok());
    EXPECT_FALSE(second->finishOperation());
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3011, claimed).ok());
    EXPECT_EQ(claimed, second);
}

TEST(PrefillBatchRpcServerTest, ConcurrentFetchAndRollbackHaveSingleOwner) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3012);
    ASSERT_TRUE(contexts->store(3012, deferred).ok());
    ASSERT_TRUE(contexts->armTtl(3012, deferred, std::chrono::seconds(1)).ok());
    EXPECT_FALSE(deferred->finishOperation());

    std::atomic<bool>                       start{false};
    grpc::Status                            take_status;
    std::shared_ptr<DeferredPrefillContext> fetched;
    std::shared_ptr<DeferredPrefillContext> rolled_back;
    std::thread                             fetch_thread([&] {
        while (!start.load()) {
            std::this_thread::yield();
        }
        take_status = contexts->take(3012, fetched);
    });
    std::thread                             rollback_thread([&] {
        while (!start.load()) {
            std::this_thread::yield();
        }
        rolled_back = contexts->remove(3012, deferred.get());
    });
    start.store(true);
    fetch_thread.join();
    rollback_thread.join();

    EXPECT_EQ(static_cast<int>(take_status.ok()) + static_cast<int>(rolled_back != nullptr), 1);
    EXPECT_EQ(fetched ? fetched : rolled_back, deferred);
    EXPECT_EQ(contexts->size(), 0);
}

TEST(PrefillBatchRpcServerTest, StopAcceptingPreservesPublishedContextsForFetch) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3014);
    ASSERT_TRUE(contexts->store(3014, deferred).ok());
    ASSERT_TRUE(contexts->armTtl(3014, deferred, std::chrono::seconds(1)).ok());
    EXPECT_FALSE(deferred->finishOperation());

    contexts->stopAccepting();
    auto rejected = makeDeferred(server, 3015);
    EXPECT_EQ(contexts->store(3015, rejected).error_code(), grpc::StatusCode::UNAVAILABLE);

    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3014, claimed).ok());
    EXPECT_EQ(claimed, deferred);
}

TEST(PrefillBatchRpcServerTest, DeferredContextMapExpiresAndCancelsUnfetchedContext) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3003);

    ASSERT_TRUE(contexts->store(3003, deferred).ok());
    ASSERT_TRUE(contexts->armTtl(3003, deferred, std::chrono::milliseconds(10)).ok());
    for (int i = 0; i < 100 && contexts->size() != 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    EXPECT_EQ(contexts->size(), 0);
    EXPECT_TRUE(deferred->context->cancel_state->load());
    EXPECT_EQ(deferred->context->error_status.error_code(), grpc::StatusCode::DEADLINE_EXCEEDED);
}

TEST(PrefillBatchRpcServerTest, TakingContextCancelsItsTtlWithoutCancellingRequest) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3004);

    ASSERT_TRUE(contexts->store(3004, deferred).ok());
    ASSERT_TRUE(contexts->armTtl(3004, deferred, std::chrono::milliseconds(20)).ok());
    EXPECT_FALSE(deferred->finishOperation());
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3004, claimed).ok());
    ASSERT_EQ(claimed, deferred);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    EXPECT_FALSE(deferred->context->cancel_state->load());
}

TEST(PrefillBatchRpcServerTest, TakingContextDoesNotLeakItsAlarm) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  deferred = makeDeferred(server, 3005);

    ASSERT_TRUE(contexts->store(3005, deferred).ok());
    ASSERT_TRUE(contexts->armTtl(3005, deferred, std::chrono::seconds(1)).ok());
    EXPECT_FALSE(deferred->finishOperation());
    std::weak_ptr<grpc::Alarm>              alarm = deferred->ttl_alarm;
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(3005, claimed).ok());
    claimed.reset();
    deferred.reset();
    EXPECT_TRUE(alarm.expired());
}

TEST(PrefillBatchRpcServerTest, CancelAllClearsAndCancelsDeferredContexts) {
    PrefillBatchRpcServer server;
    auto                  contexts = std::make_shared<DeferredPrefillContextMap>();
    auto                  first    = makeDeferred(server, 3006);
    auto                  second   = makeDeferred(server, 3007);

    ASSERT_TRUE(contexts->store(3006, first).ok());
    ASSERT_TRUE(contexts->store(3007, second).ok());
    ASSERT_TRUE(contexts->armTtl(3006, first, std::chrono::seconds(1)).ok());
    ASSERT_TRUE(contexts->armTtl(3007, second, std::chrono::seconds(1)).ok());
    contexts->cancelAll(grpc::Status(grpc::StatusCode::UNAVAILABLE, "shutdown"));

    EXPECT_EQ(contexts->size(), 0);
    EXPECT_TRUE(first->context->cancel_state->load());
    EXPECT_TRUE(second->context->cancel_state->load());
    EXPECT_EQ(first->context->error_status.error_code(), grpc::StatusCode::UNAVAILABLE);

    auto       after_shutdown  = makeDeferred(server, 3008);
    const auto shutdown_status = contexts->store(3008, after_shutdown);
    EXPECT_EQ(shutdown_status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(shutdown_status.error_message(), "Prefill batch server is shutting down");
}

TEST(PrefillBatchRpcServerTest, CancelPendingRequestsCancelsActiveContextBeforeItBecomesFetchable) {
    PrefillBatchRpcServer server;
    auto                  deferred = makeDeferred(server, 3009);

    ASSERT_TRUE(server.deferred_contexts_->registerActive(3009, deferred).ok());
    ASSERT_FALSE(deferred->context->cancel_state->load());

    server.beginShutdown();
    EXPECT_FALSE(deferred->context->cancel_state->load());

    server.cancelPendingRequests();
    EXPECT_TRUE(deferred->context->cancel_state->load());
    EXPECT_EQ(deferred->context->terminalCause(), PrefillTerminalCause::OTHER);
    EXPECT_EQ(deferred->context->error_status.error_code(), grpc::StatusCode::UNAVAILABLE);
    EXPECT_EQ(deferred->context->error_status.error_message(), "Prefill batch server is shutting down");

    auto       after_shutdown  = makeDeferred(server, 3010);
    const auto shutdown_status = server.deferred_contexts_->registerActive(3010, after_shutdown);
    EXPECT_EQ(shutdown_status.error_code(), grpc::StatusCode::UNAVAILABLE);
}

}  // namespace
}  // namespace rtp_llm
