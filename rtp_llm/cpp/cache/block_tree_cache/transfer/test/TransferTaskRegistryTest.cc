#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskRegistry.h"
#include <gtest/gtest.h>
#include <set>
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferTaskId.h"

namespace rtp_llm {
TEST(TransferTaskRegistryTest, MixedQueriesPreserveMissingRecords) {
    TransferTaskRegistry registry;
    using State = TransferTaskRegistry::State;
    ASSERT_TRUE(registry.start("running"));
    ASSERT_TRUE(registry.start("finished"));
    ASSERT_TRUE(registry.finish("finished"));
    const State expected[] = {State::RUNNING, State::FINISHED, State::NOT_FOUND};
    size_t i = 0;
    for (const auto* id : {"running", "finished", "missing"}) EXPECT_EQ(registry.query(id), expected[i++]);
    EXPECT_EQ(registry.query("missing"), State::NOT_FOUND);
}

TEST(TransferTaskRegistryTest, ExecutionIdsAreDistinct) {
    std::set<std::string> ids;
    for (int i = 0; i < 1000; ++i) {
        auto id = nextTransferTaskId();
        EXPECT_FALSE(id.empty());
        EXPECT_TRUE(ids.insert(id).second);
    }
}
TEST(TransferTaskRegistryTest, TracksCompletionWithoutRestartingTask) {
    TransferTaskRegistry registry;
    using State = TransferTaskRegistry::State;
    EXPECT_EQ(registry.query("task"), State::NOT_FOUND);
    EXPECT_FALSE(registry.finish("task"));
    EXPECT_EQ(registry.query("task"), State::NOT_FOUND);
    EXPECT_FALSE(registry.start(""));
    ASSERT_TRUE(registry.start("task"));
    EXPECT_FALSE(registry.start("task"));
    EXPECT_EQ(registry.query("task"), State::RUNNING);
    ASSERT_TRUE(registry.start("other"));
    ASSERT_TRUE(registry.finish("task"));
    EXPECT_TRUE(registry.finish("task"));
    EXPECT_FALSE(registry.start("task"));
    EXPECT_EQ(registry.query("task"), State::FINISHED);
    EXPECT_EQ(registry.query("other"), State::RUNNING);
}
}  // namespace rtp_llm
