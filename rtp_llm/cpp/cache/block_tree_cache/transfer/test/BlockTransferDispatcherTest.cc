#include <gtest/gtest.h>

#include <deque>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/group_set/FullGroupSet.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"

namespace rtp_llm {
namespace {

std::shared_ptr<AsyncContext> okContext() {
    return std::make_shared<CompletedAsyncContext>(ErrorInfo::OkStatus());
}

std::shared_ptr<AsyncContext> failedContext() {
    return std::make_shared<CompletedAsyncContext>(
        ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "scripted transfer failure"));
}

class ScriptedPerRankEngine final: public PerRankBlockTransferEngine {
public:
    explicit ScriptedPerRankEngine(std::deque<std::shared_ptr<AsyncContext>> contexts = {}):
        PerRankBlockTransferEngine(std::vector<GroupSetPtr>{}), contexts_(std::move(contexts)) {}

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>& descriptors) override {
        ++submit_count_;
        submitted_batches_.push_back(descriptors);
        if (contexts_.empty()) {
            return okContext();
        }
        std::shared_ptr<AsyncContext> context = contexts_.front();
        contexts_.pop_front();
        return context;
    }

    size_t submittedBatchCount() const {
        return submit_count_;
    }

    const std::vector<std::vector<TransferDescriptor>>& submittedBatches() const {
        return submitted_batches_;
    }

private:
    std::deque<std::shared_ptr<AsyncContext>> contexts_;
    size_t                                    submit_count_{0};
    std::vector<std::vector<TransferDescriptor>> submitted_batches_;
};

TransferDescriptor descriptor(size_t group_id) {
    return TransferDescriptor::hostToDisk(group_id, 1, 1);
}

TEST(BlockTransferDispatcherTest, DescriptorVectorUsesPerRankEntry) {
    std::shared_ptr<ScriptedPerRankEngine> engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher                dispatcher(engine);

    auto context = dispatcher.executePerRank({TransferDescriptor::hostToDisk(0, 1, 1)});
    context->waitDone();
    EXPECT_TRUE(context->success());
    EXPECT_EQ(engine->submittedBatchCount(), 1u);
}

TEST(BlockTransferDispatcherTest, EmptyBatchSucceedsWithoutAnEngine) {
    BlockTransferDispatcher dispatcher(nullptr);
    auto context = dispatcher.executeMultiRank({}, 0);
    context->waitDone();
    EXPECT_TRUE(context->success());
}

TEST(BlockTransferDispatcherTest, PerRankBatchUsesOneSubmit) {
    auto engine = std::make_shared<ScriptedPerRankEngine>(
        std::deque<std::shared_ptr<AsyncContext>>{okContext(), failedContext(), okContext()});
    BlockTransferDispatcher dispatcher(engine);

    auto context = dispatcher.executeMultiRank({descriptor(0), descriptor(1), descriptor(2)}, 100);
    context->waitDone();
    EXPECT_TRUE(context->success());
    EXPECT_EQ(engine->submittedBatchCount(), 1u);
}

TEST(BlockTransferDispatcherTest, MultiRankFailureDoesNotFallbackToPerRank) {
    auto per_rank_engine = std::make_shared<ScriptedPerRankEngine>(
        std::deque<std::shared_ptr<AsyncContext>>{okContext()});
    auto group_set = std::make_shared<FullGroupSet>(std::vector<DeviceBlockPoolPtr>{}, nullptr, nullptr);
    auto multi_rank_engine =
        std::make_shared<MultiRankBlockTransferEngine>(std::vector<GroupSetPtr>{group_set}, nullptr);
    BlockTransferDispatcher dispatcher(per_rank_engine, multi_rank_engine);

    const TransferDescriptor unsupported;
    auto context = dispatcher.executeMultiRank({unsupported}, 100);
    context->waitDone();
    EXPECT_FALSE(context->success());
    EXPECT_EQ(per_rank_engine->submittedBatchCount(), 0u);
}

TEST(BlockTransferDispatcherTest, ReturnsPendingPerRankContextWithoutWaiting) {
    auto pending = std::make_shared<TransferBatchAsyncContext>();
    auto engine  = std::make_shared<ScriptedPerRankEngine>(
        std::deque<std::shared_ptr<AsyncContext>>{pending});
    BlockTransferDispatcher dispatcher(engine);

    auto context = dispatcher.executeMultiRank({descriptor(0)}, 100);

    EXPECT_EQ(context, pending);
    EXPECT_FALSE(context->done());
    pending->complete(ErrorInfo::OkStatus());
}

TEST(BlockTransferDispatcherTest, SynchronousRunTransferSubmitsSingletonsInOrder) {
    auto engine = std::make_shared<ScriptedPerRankEngine>();
    BlockTransferDispatcher dispatcher(engine);

    EXPECT_TRUE(dispatcher.runTransfer({descriptor(0), descriptor(1), descriptor(2)}, 100));
    ASSERT_EQ(engine->submittedBatches().size(), 3u);
    for (size_t index = 0; index < 3; ++index) {
        ASSERT_EQ(engine->submittedBatches()[index].size(), 1u);
        EXPECT_EQ(engine->submittedBatches()[index].front().group_set_id, index);
    }
}

TEST(BlockTransferDispatcherTest, SynchronousRunTransferStopsAtFirstFailure) {
    auto engine = std::make_shared<ScriptedPerRankEngine>(
        std::deque<std::shared_ptr<AsyncContext>>{okContext(), failedContext(), okContext()});
    BlockTransferDispatcher dispatcher(engine);

    EXPECT_FALSE(dispatcher.runTransfer({descriptor(0), descriptor(1), descriptor(2)}, 100));
    EXPECT_EQ(engine->submittedBatchCount(), 2u);
}

TEST(BlockTransferDispatcherTest, AsynchronousRunTransferGroupsAndWaitsForEveryBatch) {
    auto first  = std::make_shared<TransferBatchAsyncContext>();
    auto second = std::make_shared<TransferBatchAsyncContext>();
    auto third  = std::make_shared<TransferBatchAsyncContext>();
    auto engine = std::make_shared<ScriptedPerRankEngine>(
        std::deque<std::shared_ptr<AsyncContext>>{first, second, third});
    BlockTransferDispatcher dispatcher(engine, nullptr, 2);

    size_t    callback_count = 0;
    ErrorInfo final_error    = ErrorInfo::OkStatus();
    dispatcher.runTransfer({descriptor(0), descriptor(1), descriptor(0), descriptor(0)},
                           100,
                           [&](ErrorInfo error) {
                               ++callback_count;
                               final_error = std::move(error);
                           });

    ASSERT_EQ(engine->submittedBatches().size(), 3u);
    EXPECT_EQ(engine->submittedBatches()[0].size(), 2u);
    EXPECT_EQ(engine->submittedBatches()[0][0].group_set_id, 0u);
    EXPECT_EQ(engine->submittedBatches()[0][1].group_set_id, 0u);
    EXPECT_EQ(engine->submittedBatches()[1].size(), 1u);
    EXPECT_EQ(engine->submittedBatches()[1][0].group_set_id, 0u);
    EXPECT_EQ(engine->submittedBatches()[2].size(), 1u);
    EXPECT_EQ(engine->submittedBatches()[2][0].group_set_id, 1u);

    second->complete(ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "second failed"));
    first->complete(ErrorInfo::OkStatus());
    EXPECT_EQ(callback_count, 0u);
    third->complete(ErrorInfo::OkStatus());
    EXPECT_EQ(callback_count, 1u);
    EXPECT_FALSE(final_error.ok());
}

}  // namespace
}  // namespace rtp_llm
