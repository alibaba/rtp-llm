#include "rtp_llm/cpp/cache/writeback/PdKvWritebackManifest.h"

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(PdKvWritebackManifestTest, DropsFinalPartialBlock) {
    PdKvWritebackSnapshot snapshot;
    snapshot.seq_size_per_block = 16;
    snapshot.final_token_count  = 39;
    snapshot.cache_keys         = {101, 102, 103};
    snapshot.group_block_ids    = {{1, 2, 3}};

    auto manifest = buildPdKvWritebackManifest(snapshot);

    ASSERT_TRUE(manifest.ok()) << manifest.status();
    EXPECT_EQ(manifest.value().reusable_block_count, 2);
    EXPECT_EQ(manifest.value().cache_keys, CacheKeysType({101, 102}));
    ASSERT_EQ(manifest.value().group_block_ids.size(), 1);
    EXPECT_EQ(manifest.value().group_block_ids[0], BlockIndicesType({1, 2}));
}

TEST(PdKvWritebackManifestTest, KeepsDecodeContinuedFullBlock) {
    PdKvWritebackSnapshot snapshot;
    snapshot.seq_size_per_block  = 16;
    snapshot.final_token_count   = 48;
    snapshot.prefill_token_count = 19;
    snapshot.cache_keys          = {201, 202, 203};
    snapshot.group_block_ids     = {{11, 12, 13}};

    auto manifest = buildPdKvWritebackManifest(snapshot);

    ASSERT_TRUE(manifest.ok()) << manifest.status();
    EXPECT_EQ(manifest.value().reusable_block_count, 3);
    EXPECT_EQ(manifest.value().cache_keys, CacheKeysType({201, 202, 203}));
    ASSERT_EQ(manifest.value().group_block_ids.size(), 1);
    EXPECT_EQ(manifest.value().group_block_ids[0], BlockIndicesType({11, 12, 13}));
}

TEST(PdKvWritebackManifestTest, RejectsMultimodalOverlap) {
    PdKvWritebackSnapshot snapshot;
    snapshot.seq_size_per_block = 16;
    snapshot.final_token_count  = 64;
    snapshot.cache_keys         = {301, 302, 303, 304};
    snapshot.group_block_ids    = {{21, 22, 23, 24}};
    snapshot.mm_intervals       = {{20, 28}};

    auto manifest = buildPdKvWritebackManifest(snapshot);

    ASSERT_TRUE(manifest.ok()) << manifest.status();
    EXPECT_EQ(manifest.value().reusable_block_count, 1);
    EXPECT_EQ(manifest.value().cache_keys, CacheKeysType({301}));
    ASSERT_EQ(manifest.value().group_block_ids.size(), 1);
    EXPECT_EQ(manifest.value().group_block_ids[0], BlockIndicesType({21}));
}

TEST(PdKvWritebackManifestTest, RejectsShortGroupBlocks) {
    PdKvWritebackSnapshot snapshot;
    snapshot.seq_size_per_block = 16;
    snapshot.final_token_count  = 32;
    snapshot.cache_keys         = {401, 402};
    snapshot.group_block_ids    = {{31}};

    auto manifest = buildPdKvWritebackManifest(snapshot);

    EXPECT_FALSE(manifest.ok());
}

TEST(PdKvWritebackManifestTest, UsesRequestIdFallbackWhenRequestKeyIsEmpty) {
    PdKvWritebackSnapshot snapshot;
    snapshot.request_id         = 12345;
    snapshot.seq_size_per_block = 16;
    snapshot.final_token_count  = 16;
    snapshot.cache_keys         = {501};
    snapshot.group_block_ids    = {{41}};

    auto manifest = buildPdKvWritebackManifest(snapshot);

    ASSERT_TRUE(manifest.ok()) << manifest.status();
    EXPECT_EQ(manifest.value().request_key, "pd_kv_writeback_12345");
}

}  // namespace
}  // namespace rtp_llm
