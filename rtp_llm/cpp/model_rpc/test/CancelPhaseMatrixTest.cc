// AutoTPM Cancel: phase × mode matrix for the cancel-intent design.
//
//   A group — QUEUE/streaming mode (GenerateStreamCall, no fetch):
//     A1 cancel before arrival, A2 cancel while queued, A3 cancel while
//     running, A4 cancel after finish.
//   B group — BATCH mode (EnqueueBatch, then FetchResponse):
//     B6 cancel before enqueue (slot filter), B7/B8 cancel while queued /
//     running then fetch, B9 cancel after finish then fetch, B10 fetch-take
//     vs. intent-consume interleavings.
//   C group — attribution and visibility:
//     C11 8429 attribution, C12 finished-records visibility.
//
// Layering: A/C cases run at the scheduler layer (real FIFOScheduler +
// NormalGenerateStream + KVCacheManager); B fetch semantics are asserted at
// the DeferredPrefillContextMap::take() + stream terminal-state level — the
// real FetchResponse only serializes what the stream already carries, so the
// stream's terminal state at take() is the observable fetch outcome. TTL is
// exercised by calling sweepExpired() with an injected clock, never by
// sleeping.

#include <memory>
#include <string>
#include <vector>
#include "torch/all.h"
#include "gtest/gtest.h"

#include "rtp_llm/cpp/engine_base/schedulers/CancelIntentMap.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/model_rpc/PrefillBatchRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RpcServerRuntimeMeta.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

namespace rtp_llm {

namespace {

// Minimal scheduler stub: carries the CancelIntentMap for the batch-R1 tests
// where enqueueGroupStreams only touches engine_->getScheduler().cancelIntentMap().
class StubScheduler: public SchedulerBase {
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

// Engine stub with scripted enqueueMultiple results and a StubScheduler.
class StubBatchEngine: public EngineBase {
public:
    StubBatchEngine(): EngineBase(EngineInitParams()) {
        scheduler_ = std::make_unique<StubScheduler>();
    }

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

}  // namespace

class CancelPhaseMatrixTest: public DeviceTestBase {
protected:
    CancelPhaseMatrixTest(): perf_scope("PERF_TEST", "1") {}

    void SetUp() override {
        DeviceTestBase::SetUp();
        cache_config_ = test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/21, /*tokens_per_block=*/2, rtp_llm::DataType::TYPE_INT8);
        cache_manager_ = std::make_shared<KVCacheManager>(cache_config_);
        ASSERT_TRUE(cache_manager_->init());
    }

    std::shared_ptr<FIFOScheduler> createScheduler() {
        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;
        runtime_config.max_generate_batch_size                     = 100;
        runtime_config.fifo_scheduler_config.max_batch_tokens_size = 8192;
        PDSepConfig         pd_sep_config;
        ParallelismConfig   parallelism_config;
        ModelSpecificConfig model_specific_config;
        return std::make_shared<FIFOScheduler>(
            runtime_config, model_config, pd_sep_config, parallelism_config, model_specific_config, cache_manager_);
    }

    GenerateStreamPtr createStream(int64_t request_id, const std::vector<int>& input_tokens = {1, 2, 3}) {
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager_;

        ModelConfig model_config;
        model_config.max_seq_len = 8192;
        RuntimeConfig runtime_config;

        auto query             = std::make_shared<GenerateInput>();
        auto generate_config   = std::make_shared<GenerateConfig>();
        query->request_id      = request_id;
        query->input_ids       = torch::tensor(input_tokens, torch::kInt32);
        query->generate_config = generate_config;
        return std::make_shared<NormalGenerateStream>(query, model_config, runtime_config, resource_context, nullptr);
    }

    size_t initialFreeBlocks() const {
        return cache_manager_->freeBlocksNum();
    }

    // Schedule until the enqueued stream(s) reach RUNNING (multi-round state machine).
    absl::StatusOr<std::list<GenerateStreamPtr>> scheduleToRunning(std::shared_ptr<FIFOScheduler>& scheduler) {
        auto result1 = scheduler->schedule();
        if (!result1.ok() || result1.value().size() > 0) {
            return result1;
        }
        auto result2 = scheduler->schedule();
        if (!result2.ok() || result2.value().size() > 0) {
            return result2;
        }
        return scheduler->schedule();
    }

    // Deferred context holding a live stream, as EnqueueGroup leaves it for FetchResponse.
    std::shared_ptr<DeferredPrefillContext> makeDeferredWithStream(PrefillBatchRpcServer&   server,
                                                                   const GenerateStreamPtr& stream) {
        auto input = std::make_shared<GenerateInputPB>();
        input->set_request_id(stream->streamId());
        RPCContext rpc_context{input.get(), nullptr};
        auto       context      = std::make_unique<PrefillGenerateContext>(&server.resource(),
                                                                rpc_context,
                                                                /*timeout_ms=*/0,
                                                                /*server_context=*/nullptr,
                                                                server.metrics_reporter_,
                                                                server.meta_);
        context->generate_input = stream->generateInput();
        context->setStream(stream);
        auto deferred     = std::make_shared<DeferredPrefillContext>();
        deferred->context = std::move(context);
        deferred->input   = input;
        return deferred;
    }

    static int64_t nowMs() {
        return autil::TimeUtility::currentTimeInMilliSeconds();
    }

protected:
    autil::EnvGuard                 perf_scope;
    CacheConfig                     cache_config_;
    std::shared_ptr<KVCacheManager> cache_manager_;
};

// ============================================================================
// A1. QUEUE mode: cancel lands before the request arrives — the R1 enqueue
// checkpoint (tryConsume before stream creation) rejects it and consumes the
// entry; the scheduler never sees a stream.
// ============================================================================
TEST_F(CancelPhaseMatrixTest, A1CancelBeforeArrivalRejectedAtEnqueueCheckpoint) {
    auto  scheduler = createScheduler();
    auto& intents   = *scheduler->cancelIntentMap();

    intents.registerCancel(/*request_id=*/101, ErrorCode::CANCELLED, nowMs());

    // R1 checkpoint as run by PrefillRpcServer::enqueueRequest.
    auto intent = intents.tryConsume(101);
    ASSERT_TRUE(intent.has_value());
    EXPECT_EQ(intent->terminal_code, ErrorCode::CANCELLED);
    // Request rejected: no stream is ever created or enqueued.
    EXPECT_TRUE(intents.empty());
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    EXPECT_EQ(scheduler->runningStreamsSize(), 0);
}

// ============================================================================
// A2. QUEUE mode: cancel while the stream waits in the queue — the next
// schedule() round (R2) stops it, removes it and consumes the entry.
// ============================================================================
TEST_F(CancelPhaseMatrixTest, A2CancelWhileQueuedStopsStreamAndConsumesIntent) {
    auto scheduler   = createScheduler();
    auto stream      = createStream(102);
    auto free_before = initialFreeBlocks();

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_EQ(scheduler->waitingStreamsSize(), 1);

    scheduler->cancelIntentMap()->registerCancel(102, ErrorCode::CANCELLED, nowMs());

    auto result = scheduler->schedule();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result.value().size(), 0);
    EXPECT_EQ(scheduler->waitingStreamsSize(), 0);
    EXPECT_EQ(scheduler->runningStreamsSize(), 0);
    EXPECT_TRUE(stream->isFinished());
    EXPECT_TRUE(stream->hasError());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
    EXPECT_NE(stream->stopReason().find("cancelled by master"), std::string::npos);
    EXPECT_TRUE(scheduler->cancelIntentMap()->empty());
    EXPECT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

// ============================================================================
// A3. QUEUE mode: cancel while the stream is running — the in-flight forward
// round is untouched (intent registration alone never touches the stream);
// the next schedule() round evicts it and consumes the entry.
// ============================================================================
TEST_F(CancelPhaseMatrixTest, A3CancelWhileRunningTakesEffectNextTickOnly) {
    auto scheduler   = createScheduler();
    auto stream      = createStream(103);
    auto free_before = initialFreeBlocks();

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);
    ASSERT_EQ(result1.value().front().get(), stream.get());
    ASSERT_TRUE(stream->getStatus() == StreamState::RUNNING);

    scheduler->cancelIntentMap()->registerCancel(103, ErrorCode::CANCELLED, nowMs());

    // Registering the intent must not disturb the batch the engine is forwarding.
    EXPECT_TRUE(stream->getStatus() == StreamState::RUNNING);
    EXPECT_FALSE(stream->hasError());

    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    // The withdrawn stream is absent from the next batch.
    EXPECT_EQ(result2.value().size(), 0);
    EXPECT_EQ(scheduler->runningStreamsSize(), 0);
    EXPECT_TRUE(stream->isFinished());
    EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
    EXPECT_TRUE(scheduler->cancelIntentMap()->empty());
    EXPECT_EQ(cache_manager_->freeBlocksNum(), free_before);
}

// ============================================================================
// A4. QUEUE mode: cancel after the stream already finished — R2 never sees the
// stream, the finished state is untouched, and the entry stays until the R3
// TTL sweep (driven here with an injected clock, no sleeping).
// ============================================================================
TEST_F(CancelPhaseMatrixTest, A4CancelAfterFinishLeavesStreamAndEntryUntilTtl) {
    auto scheduler = createScheduler();
    auto stream    = createStream(104);
    auto keepalive = createStream(999, {7, 8});

    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(scheduler->enqueue(keepalive).ok());
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 2);

    // Finish the target normally.
    stream->reportEvent(StreamEvents::GenerateDone);
    auto result2 = scheduler->schedule();
    ASSERT_TRUE(result2.ok());
    ASSERT_TRUE(stream->isFinished());
    ASSERT_FALSE(stream->hasError());

    // Late cancel: several R2 rounds must neither revive nor re-terminate it.
    auto& intents = *scheduler->cancelIntentMap();
    intents.registerCancel(104, ErrorCode::CANCELLED, nowMs());
    for (int i = 0; i < 3; ++i) {
        ASSERT_TRUE(scheduler->schedule().ok());
    }
    EXPECT_TRUE(stream->isFinished());
    EXPECT_FALSE(stream->hasError());
    EXPECT_EQ(intents.size(), 1u);

    // R3 with an injected clock past the TTL drops the unconsumed entry.
    intents.sweepExpired(nowMs() + CancelIntentMap::kTtlMs + 1);
    EXPECT_TRUE(intents.empty());

    keepalive->reportError(ErrorCode::CANCELLED, "test teardown");
    ASSERT_TRUE(scheduler->schedule().ok());
}

// ============================================================================
// B6. BATCH mode: cancel before enqueue — enqueueGroupStreams' R1 slot filter
// rejects the cancelled slot with the intent's attribution, consumes the
// entry, and admits the surviving slot normally.
// ============================================================================
TEST_F(CancelPhaseMatrixTest, B6BatchEnqueueFiltersCancelledSlotAndAdmitsSurvivor) {
    PrefillBatchRpcServer server;
    server.meta_   = std::make_shared<RpcServerRuntimeMeta>();
    auto engine    = std::make_shared<StubBatchEngine>();
    server.engine_ = engine;

    // Two ready slots; request 7001 has a matching cancel intent.
    std::vector<PrefillBatchRpcServer::BatchSlot> slots(2);
    std::vector<PrefillBatchRpcServer::ReadySlot> ready_slots;
    const std::vector<int64_t>                    request_ids = {7001, 7002};
    for (size_t i = 0; i < request_ids.size(); ++i) {
        auto& slot = slots[i];
        slot.input = std::make_shared<GenerateInputPB>();
        slot.input->set_request_id(request_ids[i]);
        RPCContext rpc_context{slot.input.get(), nullptr};
        slot.prefill_context                 = std::make_unique<PrefillGenerateContext>(&server.resource(),
                                                                        rpc_context,
                                                                        /*timeout_ms=*/0,
                                                                        /*server_context=*/nullptr,
                                                                        server.metrics_reporter_,
                                                                        server.meta_);
        auto input                           = std::make_shared<GenerateInput>();
        input->request_id                    = request_ids[i];
        input->begin_time_us                 = autil::TimeUtility::currentTimeInMicroSeconds();
        input->input_ids                     = torch::tensor({1, 2, 3}, torch::kInt32);
        input->generate_config               = std::make_shared<GenerateConfig>();
        slot.prefill_context->generate_input = input;
        auto deferred                        = std::make_shared<DeferredPrefillContext>();
        deferred->context                    = std::move(slot.prefill_context);
        deferred->input                      = slot.input;
        ready_slots.push_back(PrefillBatchRpcServer::ReadySlot{&slot, std::move(deferred)});
    }

    engine->getScheduler().cancelIntentMap()->registerCancel(7001, ErrorCode::PRIORITY_PREEMPTED, nowMs());

    // The engine only sees the surviving input.
    engine->streams            = {createStream(7002)};
    engine->enqueue_successes  = {true};
    auto rejected_cancel_state = ready_slots[0].deferred->context->cancel_state;

    EnqueueBatchResponsePB response;
    ASSERT_TRUE(server.enqueueGroupStreams(ready_slots, &response).ok());

    // Survivor admitted, cancelled slot rejected with 8429 attribution.
    ASSERT_EQ(ready_slots.size(), 1);
    EXPECT_EQ(ready_slots[0].slot->input->request_id(), 7002);
    ASSERT_EQ(response.errors_size(), 1);
    EXPECT_EQ(response.errors(0).request_id(), 7001);
    EXPECT_EQ(response.errors(0).error_info().error_code(), grpc::StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_NE(response.errors(0).error_info().error_message().find("cancelled before enqueue"), std::string::npos);
    EXPECT_TRUE(rejected_cancel_state->load());
    // Intent consumed at R1.
    EXPECT_TRUE(engine->getScheduler().cancelIntentMap()->empty());

    // Teardown: mark the admitted context cancelled before destruction.
    ready_slots[0].deferred->context->cancel_state->store(true);
    ready_slots.clear();
}

// ============================================================================
// B7. BATCH mode: cancel after enqueue, before the stream runs — the R2 round
// terminates it, and a later fetch (DeferredPrefillContextMap::take) observes
// the cancel terminal state instead of hanging or succeeding.
// ============================================================================
TEST_F(CancelPhaseMatrixTest, B7CancelWhileQueuedThenFetchSeesCancelTerminal) {
    PrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();

    auto scheduler = createScheduler();
    auto contexts  = std::make_shared<DeferredPrefillContextMap>();
    auto stream    = createStream(8001);
    auto deferred  = makeDeferredWithStream(server, stream);

    ASSERT_TRUE(contexts->store(8001, deferred).ok());
    ASSERT_TRUE(scheduler->enqueue(stream).ok());

    scheduler->cancelIntentMap()->registerCancel(8001, ErrorCode::PRIORITY_PREEMPTED, nowMs());
    ASSERT_TRUE(scheduler->schedule().ok());

    // Fetch attaches after the cancel: it claims the context exactly once and
    // finds the stream in the cancel terminal state (what finishStream would
    // serialize back), not pending and not successful.
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(8001, claimed).ok());
    ASSERT_EQ(claimed, deferred);
    auto& fetched_stream = claimed->context->getStream();
    ASSERT_NE(fetched_stream, nullptr);
    EXPECT_TRUE(fetched_stream->isFinished());
    EXPECT_TRUE(fetched_stream->hasError());
    EXPECT_EQ(fetched_stream->statusInfo().code(), ErrorCode::PRIORITY_PREEMPTED);
    EXPECT_TRUE(scheduler->cancelIntentMap()->empty());
    EXPECT_EQ(contexts->size(), 0);
}

// ============================================================================
// B8. BATCH mode: cancel while the stream runs — same fetch contract as B7,
// with the stream stopped from RUNNING and its KV blocks reclaimed.
// ============================================================================
TEST_F(CancelPhaseMatrixTest, B8CancelWhileRunningThenFetchSeesCancelTerminal) {
    PrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();

    auto scheduler   = createScheduler();
    auto contexts    = std::make_shared<DeferredPrefillContextMap>();
    auto stream      = createStream(8002);
    auto deferred    = makeDeferredWithStream(server, stream);
    auto free_before = initialFreeBlocks();

    ASSERT_TRUE(contexts->store(8002, deferred).ok());
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);
    ASSERT_TRUE(stream->getStatus() == StreamState::RUNNING);

    scheduler->cancelIntentMap()->registerCancel(8002, ErrorCode::CANCELLED, nowMs());
    ASSERT_TRUE(scheduler->schedule().ok());
    EXPECT_EQ(scheduler->runningStreamsSize(), 0);
    EXPECT_EQ(cache_manager_->freeBlocksNum(), free_before);

    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(8002, claimed).ok());
    auto& fetched_stream = claimed->context->getStream();
    ASSERT_NE(fetched_stream, nullptr);
    EXPECT_TRUE(fetched_stream->isFinished());
    EXPECT_EQ(fetched_stream->statusInfo().code(), ErrorCode::CANCELLED);
    EXPECT_TRUE(scheduler->cancelIntentMap()->empty());
}

// ============================================================================
// B9. BATCH mode: cancel arrives after the stream finished — the late intent
// must not corrupt the completed result: fetch still observes the clean
// terminal state and the entry ages out through TTL.
// ============================================================================
TEST_F(CancelPhaseMatrixTest, B9CancelAfterFinishThenFetchReturnsNormalResult) {
    PrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();

    auto scheduler = createScheduler();
    auto contexts  = std::make_shared<DeferredPrefillContextMap>();
    auto stream    = createStream(8003);
    auto keepalive = createStream(998, {7, 8});
    auto deferred  = makeDeferredWithStream(server, stream);

    ASSERT_TRUE(contexts->store(8003, deferred).ok());
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    ASSERT_TRUE(scheduler->enqueue(keepalive).ok());
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 2);

    stream->reportEvent(StreamEvents::GenerateDone);
    ASSERT_TRUE(scheduler->schedule().ok());
    ASSERT_TRUE(stream->isFinished());
    ASSERT_FALSE(stream->hasError());

    // Late cancel after completion.
    auto& intents = *scheduler->cancelIntentMap();
    intents.registerCancel(8003, ErrorCode::CANCELLED, nowMs());
    ASSERT_TRUE(scheduler->schedule().ok());

    // Fetch returns the normal result: finished, no error.
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(8003, claimed).ok());
    auto& fetched_stream = claimed->context->getStream();
    ASSERT_NE(fetched_stream, nullptr);
    EXPECT_TRUE(fetched_stream->isFinished());
    EXPECT_FALSE(fetched_stream->hasError());

    // The unconsumed entry is dropped by the TTL sweep, not by consumption.
    EXPECT_EQ(intents.size(), 1u);
    intents.sweepExpired(nowMs() + CancelIntentMap::kTtlMs + 1);
    EXPECT_TRUE(intents.empty());

    keepalive->reportError(ErrorCode::CANCELLED, "test teardown");
    ASSERT_TRUE(scheduler->schedule().ok());
}

// ============================================================================
// B10. BATCH mode: fetch-take vs. intent-consume ordering. Both sequential
// interleavings must end in the same safe state: stream terminated once,
// context claimed exactly once, no dangling map entries, blocks reclaimed.
// (B7/B8 already cover consume-then-take; this covers take-then-consume.)
// ============================================================================
TEST_F(CancelPhaseMatrixTest, B10FetchTakeBeforeIntentConsumeIsSafe) {
    PrefillBatchRpcServer server;
    server.meta_ = std::make_shared<RpcServerRuntimeMeta>();

    auto scheduler   = createScheduler();
    auto contexts    = std::make_shared<DeferredPrefillContextMap>();
    auto stream      = createStream(8004);
    auto deferred    = makeDeferredWithStream(server, stream);
    auto free_before = initialFreeBlocks();

    ASSERT_TRUE(contexts->store(8004, deferred).ok());
    ASSERT_TRUE(scheduler->enqueue(stream).ok());
    auto result1 = scheduleToRunning(scheduler);
    ASSERT_TRUE(result1.ok());
    ASSERT_EQ(result1.value().size(), 1);

    // Fetch claims the context while the stream is still running...
    std::shared_ptr<DeferredPrefillContext> claimed;
    ASSERT_TRUE(contexts->take(8004, claimed).ok());
    EXPECT_EQ(contexts->size(), 0);

    // ...then the cancel intent lands and R2 consumes it.
    scheduler->cancelIntentMap()->registerCancel(8004, ErrorCode::PRIORITY_PREEMPTED, nowMs());
    ASSERT_TRUE(scheduler->schedule().ok());

    // The claimed context observes the terminal state; nothing leaks.
    auto& fetched_stream = claimed->context->getStream();
    ASSERT_NE(fetched_stream, nullptr);
    EXPECT_TRUE(fetched_stream->isFinished());
    EXPECT_EQ(fetched_stream->statusInfo().code(), ErrorCode::PRIORITY_PREEMPTED);
    EXPECT_TRUE(scheduler->cancelIntentMap()->empty());
    EXPECT_EQ(scheduler->runningStreamsSize(), 0);
    EXPECT_EQ(cache_manager_->freeBlocksNum(), free_before);

    // A duplicate fetch after the take must fail cleanly, not crash.
    std::shared_ptr<DeferredPrefillContext> duplicate;
    EXPECT_EQ(contexts->take(8004, duplicate).error_code(), grpc::StatusCode::NOT_FOUND);
}

// ============================================================================
// C11. Attribution: a PRIORITY_PREEMPTED intent terminates the stream with
// exactly 8429; a plain cancel keeps the ordinary CANCELLED code.
// ============================================================================
TEST_F(CancelPhaseMatrixTest, C11PreemptionAttributionIs8429AndPlainCancelIsCancelled) {
    auto scheduler        = createScheduler();
    auto preempted_stream = createStream(9001, {1, 2});
    auto cancelled_stream = createStream(9002, {3, 4});

    ASSERT_TRUE(scheduler->enqueue(preempted_stream).ok());
    ASSERT_TRUE(scheduler->enqueue(cancelled_stream).ok());

    auto& intents = *scheduler->cancelIntentMap();
    intents.registerCancel(9001, ErrorCode::PRIORITY_PREEMPTED, nowMs());
    intents.registerCancel(9002, ErrorCode::CANCELLED, nowMs());

    ASSERT_TRUE(scheduler->schedule().ok());

    EXPECT_TRUE(preempted_stream->isFinished());
    EXPECT_EQ(preempted_stream->statusInfo().code(), ErrorCode::PRIORITY_PREEMPTED);
    EXPECT_EQ(static_cast<int>(preempted_stream->statusInfo().code()), 8429);
    EXPECT_TRUE(cancelled_stream->isFinished());
    EXPECT_EQ(cancelled_stream->statusInfo().code(), ErrorCode::CANCELLED);
    EXPECT_TRUE(intents.empty());
}

// ============================================================================
// C12. Finished visibility: a cancel-terminated stream must surface in the
// RpcServerRuntimeMeta finished records with its attribution — the master's
// release decision depends on this WorkerStatus contract (the RPC layer calls
// meta dequeue via GenerateContext::stopStream on the same path).
// ============================================================================
TEST_F(CancelPhaseMatrixTest, C12CancelledStreamAppearsInFinishedRecords) {
    RpcServerRuntimeMeta meta;
    auto                 stream = createStream(9101);

    meta.enqueue(9101, stream);
    auto before = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    ASSERT_EQ(before.running_task_info_list.size(), 1);
    ASSERT_EQ(before.finished_task_info_list.size(), 0);

    // R2-style termination, then the RPC layer dequeues the stream.
    stream->reportError(ErrorCode::PRIORITY_PREEMPTED, "cancelled by master: PRIORITY_PREEMPTED");
    stream->moveToNext();
    ASSERT_TRUE(stream->isFinished());
    meta.dequeue(9101, stream);

    auto after = meta.getEngineScheduleInfo(/*latest_finished_version=*/-1);
    EXPECT_EQ(after.running_task_info_list.size(), 0);
    ASSERT_EQ(after.finished_task_info_list.size(), 1);
    EXPECT_EQ(after.finished_task_info_list[0].request_id, 9101);
    EXPECT_EQ(after.finished_task_info_list[0].error_code, 8429);
}

}  // namespace rtp_llm
