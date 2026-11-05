#include "rtp_llm/cpp/cache/events/KVCacheEventState.h"
#include <gtest/gtest.h>

namespace rtp_llm {
namespace {
class StateTestPublisher: public KVCacheEventPublisher {
public:
    bool start() noexcept override {
        return true;
    }
    void            stop() noexcept override {}
    PublisherStatus status() const noexcept override {
        return {};
    }
    PublishResult tryPublish(KVCacheEvent event) noexcept override {
        events.push_back(event);
        return result;
    }
    std::vector<KVCacheEvent> events;
    PublishResult             result = PublishResult::ACCEPTED;
};

TEST(KVCacheEventStateTest, LayersAndGroupsHaveIndependentLifetimes) {
    KVCacheEventState  state;
    StateTestPublisher publisher;
    state.update(10, KVCacheEventMedium::HBM, {0}, publisher);
    state.update(10, KVCacheEventMedium::HOST, {1}, publisher);
    state.update(10, KVCacheEventMedium::DISK, {2}, publisher);
    const auto before = state.snapshot();
    ASSERT_EQ(before.host_block_groups.count(10), 1u);
    ASSERT_EQ(before.disk_block_groups.count(10), 1u);
    EXPECT_EQ(before.block_groups.at(10), (std::vector<size_t>{0}));
    EXPECT_EQ(before.host_block_groups.at(10), (std::vector<size_t>{1}));
    EXPECT_EQ(before.disk_block_groups.at(10), (std::vector<size_t>{2}));
    state.update(10, KVCacheEventMedium::HOST, {1}, publisher);
    EXPECT_EQ(state.snapshot().version, before.version);
    ASSERT_EQ(publisher.events.size(), 3u);
    state.update(10, KVCacheEventMedium::HOST, {}, publisher);
    EXPECT_TRUE(state.snapshot().host_block_groups.empty());
    EXPECT_EQ(state.snapshot().block_groups, before.block_groups);
    EXPECT_EQ(state.snapshot().disk_block_groups, before.disk_block_groups);
    EXPECT_EQ(publisher.events.back().medium, KVCacheEventMedium::HOST);
    EXPECT_EQ(publisher.events.back().type, KVCacheEventType::BLOCK_DELETE);
    state.erase(10, publisher);
    EXPECT_TRUE(state.snapshot().block_groups.empty());
    EXPECT_TRUE(state.snapshot().disk_block_groups.empty());
    ASSERT_EQ(publisher.events.size(), 6u);
}

TEST(KVCacheEventStateTest, SnapshotTracksDroppedDeltasAndRestoredSources) {
    KVCacheEventState  state;
    StateTestPublisher publisher;
    publisher.result = PublishResult::QUEUE_FULL;
    state.update(20, KVCacheEventMedium::HOST, {0, 1}, publisher);
    const auto before = state.snapshot();
    ASSERT_EQ(before.host_block_groups.count(20), 1u);
    state.update(20, KVCacheEventMedium::HOST, {}, publisher);
    EXPECT_TRUE(state.snapshot().host_block_groups.empty());
    // A failed/cancelled transfer makes its source usable again.
    state.update(20, KVCacheEventMedium::HOST, {0, 1}, publisher);
    EXPECT_EQ(state.snapshot().host_block_groups, before.host_block_groups);
    EXPECT_GT(state.snapshot().version, before.version);
    state.update(20, KVCacheEventMedium::HOST, {1}, publisher);
    EXPECT_EQ(publisher.events.back().group_index, 0u);
    EXPECT_EQ(publisher.events.back().type, KVCacheEventType::BLOCK_DELETE);
    EXPECT_EQ(state.snapshot().host_block_groups.at(20), (std::vector<size_t>{1}));
}
}  // namespace
}  // namespace rtp_llm
