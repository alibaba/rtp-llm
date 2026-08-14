#include "rtp_llm/cpp/cache/events/KVCMLogicalMirror.h"

#include <gtest/gtest.h>
#include <vector>

namespace rtp_llm::detail {
namespace {

TEST(KVCMLogicalMirrorTest, MaintainsSortedBoundedFinalState) {
    KVCMLogicalMirror mirror(/*max_keys=*/3);
    ASSERT_TRUE(mirror.seed(KVCacheSnapshot{{30, 10}}));
    ASSERT_TRUE(mirror.apply({
        {KVCacheEventType::BLOCK_ADD, 20, 1},
        {KVCacheEventType::BLOCK_DELETE, 10, 2},
        {KVCacheEventType::BLOCK_ADD, 40, 3},
    }));
    EXPECT_EQ((std::vector<int64_t>{20, 30, 40}), mirror.snapshot().block_keys);

    // Re-adding an existing key is idempotent at the ceiling; a new key is
    // rejected without inserting it.
    EXPECT_TRUE(mirror.apply({{KVCacheEventType::BLOCK_ADD, 30, 4}}));
    EXPECT_FALSE(mirror.apply({{KVCacheEventType::BLOCK_ADD, 50, 5}}));
    EXPECT_EQ((std::vector<int64_t>{20, 30, 40}), mirror.snapshot().block_keys);

    mirror.release();
    EXPECT_TRUE(mirror.snapshot().block_keys.empty());
}

TEST(KVCMLogicalMirrorTest, SeedEnforcesInputResourceCeiling) {
    KVCMLogicalMirror mirror(/*max_keys=*/2);
    ASSERT_TRUE(mirror.seed(KVCacheSnapshot{{1, 2}}));
    EXPECT_FALSE(mirror.seed(KVCacheSnapshot{{10, 10, 10}}));
    EXPECT_EQ((std::vector<int64_t>{1, 2}), mirror.snapshot().block_keys);

    ASSERT_TRUE(mirror.seed(KVCacheSnapshot{{7}}));
    EXPECT_EQ((std::vector<int64_t>{7}), mirror.snapshot().block_keys);
}

TEST(KVCMLogicalMirrorTest, StreamsSortedSetDifferenceInBoundedBatches) {
    const std::vector<int64_t> source{1, 3, 4};
    const std::vector<int64_t> target{2, 3, 5};
    size_t                     source_index = 0;
    size_t                     target_index = 0;

    const auto first =
        KVCMLogicalMirror::nextMutationBatch(source, target, source_index, target_index, /*max_batch_size=*/2);
    ASSERT_EQ(2u, first.size());
    EXPECT_EQ(KVCacheEventType::BLOCK_DELETE, first[0].type);
    EXPECT_EQ(1, first[0].block_key);
    EXPECT_EQ(KVCacheEventType::BLOCK_ADD, first[1].type);
    EXPECT_EQ(2, first[1].block_key);

    const auto second =
        KVCMLogicalMirror::nextMutationBatch(source, target, source_index, target_index, /*max_batch_size=*/2);
    ASSERT_EQ(2u, second.size());
    EXPECT_EQ(KVCacheEventType::BLOCK_DELETE, second[0].type);
    EXPECT_EQ(4, second[0].block_key);
    EXPECT_EQ(KVCacheEventType::BLOCK_ADD, second[1].type);
    EXPECT_EQ(5, second[1].block_key);
    EXPECT_EQ(source.size(), source_index);
    EXPECT_EQ(target.size(), target_index);
}

}  // namespace
}  // namespace rtp_llm::detail
