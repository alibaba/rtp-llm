#include <gtest/gtest.h>

#include <deque>
#include <memory>

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

    std::shared_ptr<AsyncContext> submit(const std::vector<TransferDescriptor>&) override {
        ++submit_count_;
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

private:
    std::deque<std::shared_ptr<AsyncContext>> contexts_;
    size_t                                    submit_count_{0};
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

}  // namespace
}  // namespace rtp_llm
