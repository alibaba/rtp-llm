#include <gtest/gtest.h>

#include <deque>
#include <memory>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/BlockTransferDispatcher.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

namespace rtp_llm {
namespace {

class ScriptedPerRankEngine final: public PerRankBlockTransferEngine {
public:
    explicit ScriptedPerRankEngine(std::deque<TransferStatus> statuses):
        PerRankBlockTransferEngine({}, std::make_shared<const std::vector<Component>>()),
        statuses_(std::move(statuses)) {}

    TransferHandle submit(const TransferDescriptor&) override {
        ++submit_count_;
        const TransferStatus status = statuses_.empty() ? TransferStatus::OK : statuses_.front();
        if (!statuses_.empty()) {
            statuses_.pop_front();
        }
        return TransferHandle::completed(status);
    }

    size_t submitCount() const {
        return submit_count_;
    }

private:
    std::deque<TransferStatus> statuses_;
    size_t                     submit_count_{0};
};

TransferDescriptor descriptor(int group_id) {
    return TransferDescriptor::hostToDisk(group_id, 1, 1);
}

TEST(BlockTransferDispatcherTest, EmptyBatchSucceedsWithoutAnEngine) {
    BlockTransferDispatcher dispatcher(nullptr);
    EXPECT_TRUE(dispatcher.executeMultiRank({}, 0));
}

TEST(BlockTransferDispatcherTest, PerRankBatchStopsAtFirstFailure) {
    auto engine = std::make_shared<ScriptedPerRankEngine>(
        std::deque<TransferStatus>{TransferStatus::OK, TransferStatus::DISK_IO_ERROR, TransferStatus::OK});
    BlockTransferDispatcher dispatcher(engine);

    EXPECT_FALSE(dispatcher.executeMultiRank({descriptor(0), descriptor(1), descriptor(2)}, 100));
    EXPECT_EQ(engine->submitCount(), 2u);
}

TEST(BlockTransferDispatcherTest, MultiRankFailureDoesNotFallbackToPerRank) {
    auto per_rank_engine   = std::make_shared<ScriptedPerRankEngine>(std::deque<TransferStatus>{TransferStatus::OK});
    auto multi_rank_engine = std::make_shared<MultiRankBlockTransferEngine>(std::vector<ComponentGroupPtr>{}, nullptr);
    BlockTransferDispatcher dispatcher(per_rank_engine, multi_rank_engine);

    EXPECT_FALSE(dispatcher.executeMultiRank({descriptor(0)}, 100));
    EXPECT_EQ(per_rank_engine->submitCount(), 0u);
}

}  // namespace
}  // namespace rtp_llm
