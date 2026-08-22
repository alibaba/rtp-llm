#include "rtp_llm/cpp/cache/KdaShadowCache.h"

#include <memory>
#include <optional>
#include <vector>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

class FakeShadowAllocator final: public KdaShadowBlockAllocator {
public:
    bool reserve(int seq_len, BlockIndicesType& blocks) override {
        ++reserve_calls;
        if (fail_reserve) {
            return false;
        }
        blocks = {next_block++, next_block++};
        live_blocks += blocks.size();
        last_seq_len = seq_len;
        return true;
    }

    bool release(const BlockIndicesType& blocks) override {
        ++release_calls;
        if (fail_release) {
            return false;
        }
        live_blocks -= blocks.size();
        return true;
    }

    bool   fail_reserve{false};
    bool   fail_release{false};
    int    reserve_calls{0};
    int    release_calls{0};
    int    last_seq_len{0};
    size_t live_blocks{0};

private:
    BlockIdxType next_block{100};
};

KdaShadowCommand command(KdaShadowCommandType type, KdaShadowKey key, int seq_len = 0) {
    return {type, key, seq_len, {}};
}

TEST(KdaShadowCacheTest, HappyPathAndIdempotency) {
    auto              allocator = std::make_shared<FakeShadowAllocator>();
    KdaShadowRegistry registry(allocator);
    KdaShadowKey      key{7, 11};

    auto reserve = registry.apply(command(KdaShadowCommandType::RESERVE, key, 128));
    ASSERT_TRUE(reserve.success);
    EXPECT_EQ(reserve.state, KdaShadowState::RESERVED);
    EXPECT_EQ(reserve.blocks, (BlockIndicesType{100, 101}));
    EXPECT_EQ(allocator->last_seq_len, 128);

    auto duplicate_reserve = registry.apply(command(KdaShadowCommandType::RESERVE, key, 128));
    EXPECT_TRUE(duplicate_reserve.success);
    EXPECT_TRUE(duplicate_reserve.idempotent);
    EXPECT_EQ(allocator->reserve_calls, 1);

    EXPECT_TRUE(registry.apply(command(KdaShadowCommandType::LOAD, key)).success);
    EXPECT_TRUE(registry.apply(command(KdaShadowCommandType::LOAD, key)).idempotent);
    EXPECT_TRUE(registry.apply(command(KdaShadowCommandType::COMMIT, key)).success);
    EXPECT_TRUE(registry.apply(command(KdaShadowCommandType::COMMIT, key)).idempotent);

    ASSERT_TRUE(registry.readyRecord(key).has_value());
    auto rows = registry.buildReadyBlockRows({key, std::nullopt, key});
    ASSERT_EQ(rows.size(), 3);
    EXPECT_EQ(rows[0], (BlockIndicesType{100, 101}));
    EXPECT_TRUE(rows[1].empty());
    EXPECT_EQ(rows[2], (BlockIndicesType{100, 101}));

    auto release = registry.apply(command(KdaShadowCommandType::RELEASE, key));
    EXPECT_TRUE(release.success);
    EXPECT_EQ(release.state, KdaShadowState::RELEASED);
    EXPECT_TRUE(release.blocks.empty());
    EXPECT_TRUE(registry.apply(command(KdaShadowCommandType::RELEASE, key)).idempotent);
    EXPECT_EQ(allocator->release_calls, 1);
    EXPECT_EQ(allocator->live_blocks, 0);
    EXPECT_EQ(registry.liveRecordCount(), 0);
    EXPECT_EQ(registry.liveBlockCount(), 0);
}

TEST(KdaShadowCacheTest, GenerationEpochPreventsStaleRelease) {
    auto              allocator = std::make_shared<FakeShadowAllocator>();
    KdaShadowRegistry registry(allocator);
    KdaShadowKey      old_key{42, 1};
    KdaShadowKey      new_key{42, 2};

    ASSERT_TRUE(registry.apply(command(KdaShadowCommandType::RESERVE, old_key, 64)).success);
    ASSERT_TRUE(registry.apply(command(KdaShadowCommandType::RESERVE, new_key, 64)).success);
    EXPECT_EQ(registry.liveRecordCount(), 2);

    ASSERT_TRUE(registry.apply(command(KdaShadowCommandType::RELEASE, old_key)).success);
    auto current = registry.record(new_key);
    ASSERT_TRUE(current.has_value());
    EXPECT_EQ(current->state, KdaShadowState::RESERVED);
    EXPECT_EQ(current->blocks, (BlockIndicesType{102, 103}));
    EXPECT_EQ(registry.liveRecordCount(), 1);
    EXPECT_EQ(allocator->live_blocks, 2);
}

TEST(KdaShadowCacheTest, FailureRollbackAndAllocationFailureDoNotLeak) {
    auto              allocator = std::make_shared<FakeShadowAllocator>();
    KdaShadowRegistry registry(allocator);
    KdaShadowKey      load_key{9, 3};

    ASSERT_TRUE(registry.apply(command(KdaShadowCommandType::RESERVE, load_key, 32)).success);
    ASSERT_TRUE(registry.apply(command(KdaShadowCommandType::LOAD, load_key)).success);
    auto fail = command(KdaShadowCommandType::FAIL, load_key);
    fail.error = "P-to-D shard checksum failure";
    ASSERT_TRUE(registry.apply(fail).success);
    EXPECT_EQ(registry.record(load_key)->state, KdaShadowState::ERROR);
    ASSERT_TRUE(registry.apply(command(KdaShadowCommandType::ROLLBACK, load_key)).success);
    EXPECT_EQ(registry.record(load_key)->state, KdaShadowState::RELEASED);
    EXPECT_EQ(allocator->live_blocks, 0);

    allocator->fail_reserve = true;
    KdaShadowKey alloc_key{10, 1};
    auto         reserve = registry.apply(command(KdaShadowCommandType::RESERVE, alloc_key, 32));
    EXPECT_FALSE(reserve.success);
    EXPECT_EQ(reserve.state, KdaShadowState::ERROR);
    EXPECT_TRUE(reserve.blocks.empty());
    EXPECT_TRUE(registry.apply(command(KdaShadowCommandType::ROLLBACK, alloc_key)).success);
    EXPECT_EQ(allocator->live_blocks, 0);
}

TEST(KdaShadowCacheTest, InvalidTransitionsAndNotReadyRowsFailClosed) {
    auto              allocator = std::make_shared<FakeShadowAllocator>();
    KdaShadowRegistry registry(allocator);
    KdaShadowKey      key{12, 1};

    EXPECT_FALSE(registry.apply(command(KdaShadowCommandType::LOAD, key)).success);
    ASSERT_TRUE(registry.apply(command(KdaShadowCommandType::RESERVE, key, 16)).success);
    EXPECT_FALSE(registry.apply(command(KdaShadowCommandType::COMMIT, key)).success);
    EXPECT_FALSE(registry.apply(command(KdaShadowCommandType::RESERVE, key, 17)).success);
    EXPECT_THROW(registry.buildReadyBlockRows({key}), std::runtime_error);
}

TEST(KdaShadowCacheTest, DestructorReleasesLiveRecords) {
    auto allocator = std::make_shared<FakeShadowAllocator>();
    {
        KdaShadowRegistry registry(allocator);
        ASSERT_TRUE(registry.apply(command(KdaShadowCommandType::RESERVE, {1, 1}, 16)).success);
        ASSERT_TRUE(registry.apply(command(KdaShadowCommandType::RESERVE, {2, 1}, 16)).success);
        EXPECT_EQ(allocator->live_blocks, 4);
    }
    EXPECT_EQ(allocator->live_blocks, 0);
    EXPECT_EQ(allocator->release_calls, 2);
}

}  // namespace
}  // namespace rtp_llm
