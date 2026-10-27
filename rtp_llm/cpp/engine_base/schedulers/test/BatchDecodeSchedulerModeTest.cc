#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <thread>
#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"
#include "rtp_llm/cpp/engine_base/schedulers/BatchDecodeScheduler.h"

namespace rtp_llm {

TEST(BatchDecodeSchedulerModeTest, LegacyConstructorDefaultsToDecode) {
    RuntimeConfig runtime;
    runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size  = 1;
    runtime.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type = 1;
    BatchDecodeScheduler scheduler(runtime, nullptr, nullptr);
    EXPECT_EQ(scheduler.scheduler_type_, BatchDecodeScheduler::kBatchDecode);
    EXPECT_EQ(scheduler.batch_size_, 1);
}

TEST(BatchDecodeSchedulerModeTest, PrefillRoleDoesNotDependOnSyntheticWarmupType) {
    for (int warmup_type : {0, 1}) {
        RuntimeConfig runtime;
        runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size  = 1;
        runtime.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type = warmup_type;
        BatchDecodeScheduler scheduler(runtime, nullptr, nullptr, 0, RoleType::PREFILL);
        EXPECT_EQ(scheduler.scheduler_type_, BatchDecodeScheduler::kBatchPrefill);
        EXPECT_EQ(scheduler.batch_size_, 1);
    }
}

TEST(BatchDecodeSchedulerModeTest, OtherRolesRetainDecodeMode) {
    for (RoleType role : {RoleType::DECODE, RoleType::PDFUSION}) {
        for (int warmup_type : {0, 1}) {
            RuntimeConfig runtime;
            runtime.batch_decode_scheduler_config.batch_decode_scheduler_warmup_type = warmup_type;
            BatchDecodeScheduler scheduler(runtime, nullptr, nullptr, 0, role);
            EXPECT_EQ(scheduler.scheduler_type_, BatchDecodeScheduler::kBatchDecode);
        }
    }
}

TEST(BatchDecodeSchedulerModeTest, RuntimeControlStillOverridesInitialModeAndBatchSize) {
    RuntimeConfig runtime;
    runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 1;
    BatchDecodeScheduler scheduler(runtime, nullptr, nullptr, 3, RoleType::PREFILL);
    scheduler.updateSchedulerInfo(R"({"batch_size":16,"mode":"decode"})");
    EXPECT_EQ(scheduler.scheduler_type_, BatchDecodeScheduler::kBatchDecode);
    EXPECT_EQ(scheduler.batch_size_, 16);
    EXPECT_EQ(scheduler.dp_rank_, 3);
    scheduler.updateSchedulerInfo(R"({"batch_size":32,"mode":"prefill"})");
    EXPECT_EQ(scheduler.scheduler_type_, BatchDecodeScheduler::kBatchPrefill);
    EXPECT_EQ(scheduler.batch_size_, 32);
    EXPECT_EQ(scheduler.dp_rank_, 3);
}

namespace {

template<class Stream = NormalGenerateStream>
std::shared_ptr<Stream> makeShutdownStream(StreamState state = StreamState::WAITING) {
    ModelConfig model;
    model.max_seq_len                  = 64;
    model.vocab_size                   = 128;
    model.attn_config.tokens_per_block = 2;
    RuntimeConfig   runtime;
    ResourceContext resource;
    resource.role_type                     = RoleType::PREFILL;
    auto input                             = std::make_shared<GenerateInput>();
    input->input_ids                       = torch::tensor({1, 2}, torch::kInt32);
    input->generate_config                 = std::make_shared<GenerateConfig>();
    input->generate_config->max_new_tokens = 1;
    auto stream                            = std::make_shared<Stream>(input, model, runtime, resource, nullptr);
    stream->generate_status_->status.store(state);
    return stream;
}

// Pause the actual scheduler polling expression, with a genuinely pending load.
// This does not fake moveToNext(), loadCacheDone(), or the state-machine result.
class GatedLoadingStream: public NormalGenerateStream {
public:
    using NormalGenerateStream::NormalGenerateStream;
    StreamState getStatus() const override {
        if (armed.exchange(false)) {
            polling.set_value();
            release.wait();
        }
        return NormalGenerateStream::getStatus();
    }
    mutable std::atomic<bool>  armed{false};
    mutable std::promise<void> polling;
    std::shared_future<void>   release;
};

std::shared_ptr<LoadAsyncContext> makePendingShutdownLoad() {
    auto coordinator = std::make_shared<LoadContextCoordinator>(
        [](const std::shared_ptr<LoadAsyncContext>&) { return true; }, [](LoadAsyncContext&) {});
    std::vector<TransferDescriptor> descriptors{
        TransferDescriptor{nullptr, 0, 0, Tier::HOST, Tier::DEVICE, BlockIndicesType{1}}};
    auto context = coordinator->create(std::move(descriptors), {false}, 1);
    if (!coordinator->registerContext(context)) {
        throw std::runtime_error("cannot register controlled pending load");
    }
    return context;
}

TEST(BatchDecodeSchedulerStopTest, StopRetainsAllQueuesUntilFinalizeAndPreservesCompletedResults) {
    RuntimeConfig        runtime;
    BatchDecodeScheduler scheduler(runtime, nullptr, nullptr, 0, RoleType::PREFILL);
    auto                 waiting   = makeShutdownStream();
    auto                 loading   = makeShutdownStream(StreamState::LOADING_CACHE);
    auto                 running   = makeShutdownStream(StreamState::RUNNING);
    auto                 completed = makeShutdownStream(StreamState::RUNNING);
    auto                 failed    = makeShutdownStream(StreamState::RUNNING);
    auto                 finished  = makeShutdownStream(StreamState::FINISHED);
    completed->reportEvent(StreamEvents::GenerateDone);
    failed->reportError(ErrorCode::INVALID_PARAMS, "original request error");
    scheduler.waiting_streams_       = {waiting};
    scheduler.loading_cache_streams_ = {loading};
    scheduler.running_streams_       = {running, completed, failed, finished};
    scheduler.draining_mixed_modes_  = true;

    EXPECT_TRUE(absl::IsFailedPrecondition(scheduler.finalizeStop()));
    EXPECT_EQ(scheduler.onflightStreams(), 6);
    ASSERT_TRUE(scheduler.stop().ok());
    ASSERT_TRUE(scheduler.stop().ok());
    EXPECT_EQ(scheduler.onflightStreams(), 6);
    EXPECT_EQ(waiting->getStatus(), StreamState::WAITING);
    EXPECT_EQ(loading->getStatus(), StreamState::LOADING_CACHE);
    EXPECT_EQ(running->getStatus(), StreamState::RUNNING);
    EXPECT_FALSE(waiting->hasError());
    EXPECT_FALSE(loading->hasError());
    EXPECT_FALSE(running->hasError());
    EXPECT_FALSE(completed->hasError());
    EXPECT_EQ(failed->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    auto scheduled = scheduler.schedule();
    ASSERT_TRUE(scheduled.ok());
    EXPECT_TRUE(scheduled->empty());
    EXPECT_EQ(scheduler.onflightStreams(), 6);

    ASSERT_TRUE(scheduler.finalizeStop().ok());
    for (const auto& stream : {waiting, loading, running}) {
        EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
    }
    EXPECT_EQ(completed->getStatus(), StreamState::FINISHED);
    EXPECT_TRUE(completed->hasEvent(StreamEvents::GenerateDone));
    EXPECT_FALSE(completed->hasError());
    EXPECT_EQ(failed->getStatus(), StreamState::FINISHED);
    EXPECT_EQ(failed->statusInfo().code(), ErrorCode::INVALID_PARAMS);
    EXPECT_FALSE(finished->hasError());
    EXPECT_EQ(scheduler.onflightStreams(), 0);
    EXPECT_FALSE(scheduler.draining_mixed_modes_);
    EXPECT_TRUE(scheduler.finalizeStop().ok());
    EXPECT_EQ(failed->statusInfo().code(), ErrorCode::INVALID_PARAMS);
}

TEST(BatchDecodeSchedulerStopTest, RejectedEnqueueAndGroupAreAlreadyTerminal) {
    RuntimeConfig        runtime;
    BatchDecodeScheduler scheduler(runtime, nullptr, nullptr);
    ASSERT_TRUE(scheduler.stop().ok());
    auto single = makeShutdownStream();
    EXPECT_TRUE(absl::IsCancelled(scheduler.enqueue(single)));
    EXPECT_EQ(single->getStatus(), StreamState::FINISHED);
    EXPECT_EQ(single->statusInfo().code(), ErrorCode::CANCELLED);
    std::vector<GenerateStreamPtr> group{makeShutdownStream(), makeShutdownStream()};
    const auto                     result = scheduler.enqueueGroup(group);
    EXPECT_EQ(result.first, (std::vector<bool>{false, false}));
    EXPECT_EQ(result.second, group);
    for (const auto& stream : group) {
        EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
        EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
    }
    EXPECT_EQ(scheduler.onflightStreams(), 0);
    EXPECT_TRUE(scheduler.finalizeStop().ok());
}

TEST(BatchDecodeSchedulerStopTest, IdleScheduleWakesOnStop) {
    RuntimeConfig runtime;
    runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 1;
    BatchDecodeScheduler scheduler(runtime, nullptr, nullptr);
    std::promise<void>   entering;
    auto                 entered   = entering.get_future();
    auto                 scheduled = std::async(std::launch::async, [&] {
        entering.set_value();
        return scheduler.schedule();
    });
    EXPECT_EQ(entered.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_EQ(scheduled.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
    EXPECT_TRUE(scheduler.stop().ok());
    EXPECT_EQ(scheduled.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto result = scheduled.get();
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->empty());
    EXPECT_TRUE(scheduler.finalizeStop().ok());
}

TEST(BatchDecodeSchedulerStopTest, StopDuringLoadingKeepsEverySelectedStreamForFinalize) {
    // In particular, stopping at the middle selected item must not lose earlier
    // RUNNING items or later WAITING items from the scheduler's ownership lists.
    for (int gated_index : {0, 1}) {
        SCOPED_TRACE(gated_index);
        RuntimeConfig runtime;
        runtime.batch_decode_scheduler_config.batch_decode_scheduler_batch_size = 3;
        BatchDecodeScheduler           scheduler(runtime, nullptr, nullptr, 0, RoleType::PREFILL);
        std::vector<GenerateStreamPtr> streams;
        auto                           gated = makeShutdownStream<GatedLoadingStream>(StreamState::LOADING_CACHE);
        auto                           load  = makePendingShutdownLoad();
        ASSERT_FALSE(load->done());
        gated->stream_cache_resource_->allocator_load_context_ = load;
        std::promise<void> release;
        gated->release = release.get_future().share();
        auto polling   = gated->polling.get_future();
        for (int i = 0; i < 3; ++i) {
            streams.push_back(i == gated_index ?
                                  gated :
                                  makeShutdownStream(i < gated_index ? StreamState::RUNNING : StreamState::WAITING));
            ASSERT_TRUE(scheduler.enqueue(streams.back()).ok());
        }
        gated->armed   = true;
        auto scheduled = std::async(std::launch::async, [&] { return scheduler.schedule(); });
        EXPECT_EQ(polling.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        auto       stopped  = std::async(std::launch::async, [&] { return scheduler.stop(); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!scheduler.stop_requested_.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        EXPECT_TRUE(scheduler.stop_requested_.load());
        // Never leave a worker blocked on a test gate, even if an EXPECT failed.
        release.set_value();
        EXPECT_EQ(scheduled.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_EQ(stopped.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_TRUE(stopped.get().ok());
        auto result = scheduled.get();
        ASSERT_TRUE(result.ok());
        EXPECT_TRUE(result->empty());
        EXPECT_EQ(scheduler.waiting_streams_.size(), 3);
        EXPECT_EQ(scheduler.onflightStreams(), 3);
        EXPECT_FALSE(load->done());
        for (const auto& stream : streams) {
            EXPECT_FALSE(stream->hasError());
        }
        ASSERT_TRUE(scheduler.finalizeStop().ok());
        EXPECT_EQ(scheduler.onflightStreams(), 0);
        for (const auto& stream : streams) {
            EXPECT_EQ(stream->getStatus(), StreamState::FINISHED);
            EXPECT_EQ(stream->statusInfo().code(), ErrorCode::CANCELLED);
        }
        // This CPU scheduler fixture has no KV manager. Real KV release/drain is
        // checked in EngineStopTest; retire its controlled load explicitly here.
        EXPECT_TRUE(load->abortPending());
    }
}

}  // namespace

}  // namespace rtp_llm
