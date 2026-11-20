#include <chrono>
#include <future>
#include <memory>

#include "gtest/gtest.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/normal_engine/NormalGenerateStream.h"

namespace rtp_llm {
namespace {

class FIFOSchedulerLoadRegistrationTest: public ::testing::Test {
protected:
    void SetUp() override {
        initLogger();
        ModelConfig model_config;
        model_config.max_seq_len = 16;
        RuntimeConfig runtime_config;
        // These tests only exercise host-side lifetimes: no runtime init or device allocation.
        auto cache_manager = std::make_shared<KVCacheManager>(test::makeSimpleMhaCacheConfig(
            /*layer_num=*/1, /*block_num=*/4, /*tokens_per_block=*/2, DataType::TYPE_INT8));
        scheduler_         = std::make_shared<FIFOScheduler>(
            runtime_config, model_config, PDSepConfig{}, ParallelismConfig{}, ModelSpecificConfig{}, cache_manager);
        ResourceContext resource_context;
        resource_context.cache_manager = cache_manager;
        auto input                     = std::make_shared<GenerateInput>();
        input->generate_config         = std::make_shared<GenerateConfig>();
        input->input_ids               = torch::tensor(std::vector<int>{1, 2}, torch::kInt32);
        stream_ =
            std::make_shared<NormalGenerateStream>(input, model_config, runtime_config, resource_context, nullptr);
    }

    std::shared_ptr<LoadAsyncContext> makeContext(bool joined = false) {
        auto coordinator = std::make_shared<LoadContextCoordinator>(
            [](const std::shared_ptr<LoadAsyncContext>&) { return true; }, [](LoadAsyncContext&) {});
        std::vector<TransferDescriptor> descriptors{
            TransferDescriptor{nullptr, 0, 0, Tier::HOST, Tier::DEVICE, BlockIndicesType{1}}};
        auto context = coordinator->create(std::move(descriptors), {joined}, /*matched_blocks=*/1);
        EXPECT_TRUE(coordinator->registerContext(context));
        EXPECT_TRUE(context->commit());
        return context;
    }

    void registerContext(const std::shared_ptr<LoadAsyncContext>& context) {
        stream_->streamCacheResource().allocator_load_context_ = context;
        std::lock_guard<std::mutex> lock(scheduler_->lock_);
        scheduler_->onLoadingStream(stream_);
    }

    std::shared_ptr<FIFOScheduler> scheduler_;
    GenerateStreamPtr              stream_;
};

TEST_F(FIFOSchedulerLoadRegistrationTest, CancelledCommittedJoinedLoadReturnsCountAndWakesScheduler) {
    auto context = makeContext(/*joined=*/true);
    ASSERT_FALSE(context->done());
    ASSERT_FALSE(context->abortPending());
    bool completion_called = false;
    context->onDone([&](ErrorInfo) { completion_called = true; });
    registerContext(context);
    auto state = scheduler_->scheduler_sync_state_;
    ASSERT_EQ(state->pending_cache_load_count.load(), 1);
    std::weak_ptr<LoadAsyncContext> weak_context = context;
    context.reset();

    std::promise<void> checked;
    auto               waiter = std::async(std::launch::async, [&] {
        std::unique_lock<std::mutex> lock(scheduler_->lock_);
        scheduler_->schedule_trigger_ = false;
        bool first                    = true;
        state->wait(
            lock,
            [&] {
                if (first) {
                    first = false;
                    checked.set_value();
                }
                return scheduler_->waitPredicate();
            },
            std::chrono::seconds(2));
        return scheduler_->waitPredicate();
    });
    checked.get_future().wait();
    stream_->reportError(ErrorCode::CANCELLED, "cancel pending joined load");
    EXPECT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    EXPECT_TRUE(weak_context.expired());
    EXPECT_FALSE(completion_called);
    EXPECT_EQ(state->pending_cache_load_count.load(), 0);
    EXPECT_EQ(waiter.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    EXPECT_TRUE(waiter.get());
}

TEST_F(FIFOSchedulerLoadRegistrationTest, CancellationDoesNotCompleteAContextStillOwnedByATask) {
    auto context = makeContext();
    registerContext(context);
    stream_->reportError(ErrorCode::CANCELLED, "cancel while task owns context");
    EXPECT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    EXPECT_FALSE(context->done());
    EXPECT_EQ(scheduler_->scheduler_sync_state_->pending_cache_load_count.load(), 1);

    ASSERT_TRUE(context->completeTransfers(1, true));
    EXPECT_EQ(scheduler_->scheduler_sync_state_->pending_cache_load_count.load(), 0);
}

TEST_F(FIFOSchedulerLoadRegistrationTest, CompletionThenDestructionReturnsCountOnlyOnce) {
    for (const bool success : {true, false}) {
        SCOPED_TRACE(success);
        auto context = makeContext();
        registerContext(context);
        EXPECT_EQ(scheduler_->scheduler_sync_state_->pending_cache_load_count.load(), 1);
        ASSERT_TRUE(context->completeTransfers(1, success));
        EXPECT_EQ(scheduler_->scheduler_sync_state_->pending_cache_load_count.load(), 0);
        context.reset();
        stream_->streamCacheResource().allocator_load_context_.reset();
        EXPECT_EQ(scheduler_->scheduler_sync_state_->pending_cache_load_count.load(), 0);
    }
}

TEST_F(FIFOSchedulerLoadRegistrationTest, AlreadyCompletedContextReturnsCountInline) {
    auto context = makeContext();
    ASSERT_TRUE(context->completeTransfers(1, true));
    ASSERT_TRUE(context->done());
    registerContext(context);
    EXPECT_EQ(scheduler_->scheduler_sync_state_->pending_cache_load_count.load(), 0);
    EXPECT_TRUE(scheduler_->scheduler_sync_state_->cache_load_completion_trigger);
}

TEST_F(FIFOSchedulerLoadRegistrationTest, MissingContextDoesNotLeaveAPendingRegistration) {
    registerContext(nullptr);
    EXPECT_EQ(scheduler_->scheduler_sync_state_->pending_cache_load_count.load(), 0);
}

TEST_F(FIFOSchedulerLoadRegistrationTest, DiscardedCallbackAfterSchedulerDestructionIsSafe) {
    auto context = makeContext(/*joined=*/true);
    registerContext(context);
    std::weak_ptr<SchedulerSynchronizationState> weak_state = scheduler_->scheduler_sync_state_;
    scheduler_.reset();
    EXPECT_FALSE(weak_state.expired());
    context.reset();
    stream_->reportError(ErrorCode::CANCELLED, "release after scheduler destruction");
    EXPECT_EQ(stream_->moveToNext(), StreamState::FINISHED);
    EXPECT_TRUE(weak_state.expired());
}

}  // namespace
}  // namespace rtp_llm
