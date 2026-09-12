#include <gtest/gtest.h>

#include <atomic>
#include <numeric>
#include <thread>

#include "rtp_llm/cpp/cache/DSV41GpuCheckpointCache.h"

namespace rtp_llm::test {
namespace {

DSV41GpuCheckpointData checkpoint(size_t count = 1, DSV41ReplayMode mode = DSV41ReplayMode::FULL) {
    DSV41GpuCheckpointData data;
    data.reuse_unit       = 128;
    auto& meta            = data.metadata;
    meta.identity         = {"model-r1", "layout-r1", mode, 1, 128, 136};
    meta.materialized_end = meta.encoder_materialized_end = meta.decoder_checkpoint_end = count * 128;
    meta.aux_valid_start                                                                = meta.materialized_end - 128;
    meta.aux_valid_end                                                                  = meta.materialized_end;
    meta.global_entries = meta.index_entries = {
        meta.materialized_end / 2, meta.materialized_end / 2, meta.materialized_end / 2, meta.materialized_end};
    for (size_t layer = 0; layer < meta.swa.size(); ++layer) {
        const auto floor = mode == DSV41ReplayMode::BOUNDED_CHECKPOINT_V1 && layer > 20 ? meta.aux_valid_start : 0;
        meta.swa[layer]  = {meta.aux_valid_start, meta.materialized_end, floor};
    }
    meta.history_ready = meta.draft_committed = meta.pair_empty = true;
    data.keys.resize(count);
    std::iota(data.keys.begin(), data.keys.end(), 100);
    for (size_t group = 0; group < data.blocks.size(); ++group) {
        data.blocks[group].resize(group < 4 ? count : 1);
        std::iota(data.blocks[group].begin(), data.blocks[group].end(), 1);
    }
    return data;
}

struct Counter {
    std::atomic<int>                held{0};
    std::atomic<int>                acquired{0};
    DSV41GpuCheckpointCache::Retain retain() {
        return [this](const auto&) {
            ++held;
            ++acquired;
            return std::shared_ptr<void>(new int(0), [this](void* value) {
                delete static_cast<int*>(value);
                --held;
            });
        };
    }
};

}  // namespace

TEST(DSV41GpuCheckpointCacheTest, SameTokenKeysHaveIndependentCompleteModeAndLayoutIdentities) {
    Counter                 refs;
    DSV41GpuCheckpointCache cache;
    auto                    full    = checkpoint();
    auto                    bounded = checkpoint(1, DSV41ReplayMode::BOUNDED_CHECKPOINT_V1);
    bounded.blocks[5][0]            = 8;
    ASSERT_TRUE(cache.publish(full, refs.retain()));
    ASSERT_TRUE(cache.publish(bounded, refs.retain()));
    ASSERT_EQ(cache.size(), 2);
    EXPECT_EQ(cache.match(full.metadata.identity, full.keys, 128, 1)->data.blocks[5][0], 1);
    EXPECT_EQ(cache.match(bounded.metadata.identity, full.keys, 128, 1)->data.blocks[5][0], 8);
    auto other           = full.metadata.identity;
    other.model_revision = "model-r2";
    EXPECT_FALSE(cache.match(other, full.keys, 128, 1));
    other                    = full.metadata.identity;
    other.layout_fingerprint = "layout-r2";
    EXPECT_FALSE(cache.match(other, full.keys, 128, 1));
}

TEST(DSV41GpuCheckpointCacheTest, IntermediateGlobalChainNeedsOnlyExplicitCompleteBoundaries) {
    Counter                 refs;
    DSV41GpuCheckpointCache cache;
    auto                    first = checkpoint();
    auto                    third = checkpoint(3);
    ASSERT_TRUE(cache.publish(first, refs.retain()));
    ASSERT_TRUE(cache.publish(third, refs.retain()));
    EXPECT_EQ(cache.match(third.metadata.identity, third.keys, 128, 2)->data.keys.size(), 1);
    EXPECT_EQ(cache.match(third.metadata.identity, third.keys, 128, 3)->data.keys.size(), 3);
    auto broken = third.keys;
    broken[1]   = 999;
    EXPECT_EQ(cache.match(third.metadata.identity, broken, 128, 3)->data.keys.size(), 1);
}

TEST(DSV41GpuCheckpointCacheTest, MissingRegionsAndUnwrittenAuxNeverAcquireBacking) {
    Counter                 refs;
    DSV41GpuCheckpointCache cache;
    auto                    missing = checkpoint(2);
    missing.blocks[2].pop_back();
    EXPECT_THROW(cache.publish(missing, refs.retain()), std::invalid_argument);
    missing                            = checkpoint(2);
    missing.metadata.swa[42].valid_end = 0;
    EXPECT_THROW(cache.publish(missing, refs.retain()), std::invalid_argument);
    missing                        = checkpoint(2);
    missing.metadata.aux_valid_end = 128;
    EXPECT_THROW(cache.publish(missing, refs.retain()), std::invalid_argument);
    missing              = checkpoint(2);
    missing.blocks[0][1] = missing.blocks[0][0];
    EXPECT_THROW(cache.publish(missing, refs.retain()), std::invalid_argument);
    EXPECT_EQ(refs.acquired, 0);
    EXPECT_EQ(cache.size(), 0);
}

TEST(DSV41GpuCheckpointCacheTest, MatchLeaseProtectsWholePrefixTreeAcrossConcurrentEviction) {
    Counter                 refs;
    DSV41GpuCheckpointCache cache;
    const auto              first = checkpoint();
    const auto              third = checkpoint(3);
    ASSERT_TRUE(cache.publish(first, refs.retain()));
    ASSERT_TRUE(cache.publish(third, refs.retain()));
    auto lease = cache.match(third.metadata.identity, third.keys, 128, 1);
    ASSERT_TRUE(lease);
    std::thread evict([&] {
        EXPECT_TRUE(cache.takeOldestJointEvictable(5).empty());
        EXPECT_TRUE(cache.takeOldestJointEvictable(0).empty());
    });
    evict.join();
    EXPECT_EQ(refs.held, 2);
    lease.reset();
    auto removed = cache.takeOldestJointEvictable(0);
    EXPECT_EQ(removed.size(), 2);
    EXPECT_EQ(cache.size(), 0);
    EXPECT_EQ(refs.held, 2);
    removed.clear();
    EXPECT_EQ(refs.held, 0);
}

TEST(DSV41GpuCheckpointCacheTest, FailedTransferKeepsGpuEntryUntilDestinationCommit) {
    Counter                 refs;
    DSV41GpuCheckpointCache cache;
    auto                    data = checkpoint();
    ASSERT_TRUE(cache.publish(data, refs.retain()));
    auto transfer = cache.leaseOldestForTransfer();
    ASSERT_TRUE(transfer);
    EXPECT_TRUE(cache.takeOldestJointEvictable().empty());
    transfer.reset();
    EXPECT_TRUE(cache.match(data.metadata.identity, data.keys, 128, 1));
    transfer = cache.leaseOldestForTransfer();
    ASSERT_TRUE(transfer);
    cache.commitTransfer(transfer);
    EXPECT_FALSE(cache.match(data.metadata.identity, data.keys, 128, 1));
    EXPECT_EQ(refs.held, 1);
    transfer.reset();
    EXPECT_EQ(refs.held, 0);
}

TEST(DSV41GpuCheckpointCacheTest, StaleNMetadataCannotPublishTLiveRings) {
    auto            data = checkpoint(2);
    DSV41CacheState state(data.metadata.identity);
    state.advanceEncoder(256);
    state.completeDecoder(data.metadata, 128);
    state.requireProtectedPrefix(256, 385);
    state.protect(std::make_shared<DSV41CheckpointSnapshot>(data.metadata));
    state.advanceEncoder(385);
    state.completeHandoff(385, true, true, true);
    state.finish(385);
    EXPECT_THROW(data.validateProducer(state.view()), std::invalid_argument);
}

TEST(DSV41GpuCheckpointCacheTest, CpuAndGpuPublicationDoNotConsumeEachOthersBacking) {
    auto            data = checkpoint();
    DSV41CacheState state(data.metadata.identity);
    state.advanceEncoder(128);
    state.completeDecoder(data.metadata, 128);
    state.protect(std::make_shared<DSV41CheckpointSnapshot>(data.metadata));
    state.finish(128);
    ASSERT_TRUE(state.publishGpuCheckpoint([&](const auto& view) {
        data.validateProducer(view);
        EXPECT_EQ(view.snapshots.size(), 1);
        return true;
    }));
    EXPECT_FALSE(state.view().published);
    EXPECT_TRUE(state.view().gpu_published);
    ASSERT_TRUE(state.publishSnapshots([](const auto& view) { return view.snapshots.size() == 1; }));
    EXPECT_TRUE(state.view().published);
    EXPECT_TRUE(state.view().gpu_published);
    EXPECT_THROW(state.cancel(), std::logic_error);
}

TEST(DSV41GpuCheckpointCacheTest, RefAllocationFailureAndConflictingMetadataDoNotPartiallyPublish) {
    Counter                 refs;
    DSV41GpuCheckpointCache cache;
    const auto              data = checkpoint();
    EXPECT_FALSE(cache.publish(data, [](const auto&) { return std::shared_ptr<void>{}; }));
    EXPECT_EQ(cache.size(), 0);
    EXPECT_THROW(
        cache.publish(data,
                      [](const auto&) -> std::shared_ptr<void> { throw std::runtime_error("allocation failure"); }),
        std::runtime_error);
    EXPECT_EQ(cache.size(), 0);
    ASSERT_TRUE(cache.publish(data, refs.retain()));
    auto conflicting                          = data;
    conflicting.metadata.history_token_ids[0] = 73;
    EXPECT_FALSE(cache.publish(conflicting, refs.retain()));
    conflicting              = data;
    conflicting.blocks[5][0] = 9;
    EXPECT_FALSE(cache.publish(conflicting, refs.retain()));
    EXPECT_TRUE(cache.publish(data, refs.retain()));
    EXPECT_EQ(refs.acquired, 1);
    EXPECT_EQ(refs.held, 1);
}

}  // namespace rtp_llm::test
