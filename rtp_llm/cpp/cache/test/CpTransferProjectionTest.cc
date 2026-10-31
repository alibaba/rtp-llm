#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CachePhysicalRegion.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/distribute/context_parallel/CpTransferProjection.h"

namespace rtp_llm::test {
namespace {

using KeySlot = std::pair<size_t, size_t>;

std::vector<KeySlot> rankKeySlots(size_t key_count, int cp_rank, int cp_size) {
    auto policy             = defaultCacheGroupPolicy(CacheGroupType::FULL);
    policy.cp_mapping       = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    auto spec               = std::make_shared<MHAKVCacheSpec>("full",
                                                 /*seq_size_per_block=*/4,
                                                 /*kernel_seq_size_per_block=*/4);
    spec->dtype_            = DataType::TYPE_FP32;
    spec->per_token_k_elems = 2;
    CacheConfig config;
    config.layer_num          = 1;
    config.seq_size_per_block = 4;
    config.setTopology({GroupBase{"full", spec, policy}}, {{0, {"full"}}});
    ParallelismConfig warmup_parallelism;
    warmup_parallelism.tp_size                            = 4;
    warmup_parallelism.tp_rank                            = 3;
    warmup_parallelism.prefill_cp_config.kv_cache_sharded = true;
    config.finalizeStorageLayout(warmup_parallelism, /*warmup=*/true);
    EXPECT_FALSE(config.request_projection_required);
    EXPECT_EQ(config.canonicalSizePerBlock(), 4u);
    EXPECT_EQ(config.globalSizePerBlock("full"), 4u);

    std::vector<int32_t> block_ids(key_count);
    for (size_t i = 0; i < block_ids.size(); ++i) {
        block_ids[i] = static_cast<int32_t>(i);
    }
    std::vector<uint8_t> storage((key_count + 1) * spec->block_size_bytes());
    BlockInfo            base;
    base.addr       = storage.data();
    base.size_bytes = storage.size();

    std::vector<KeySlot> pairs;
    const auto           pages = selectProducerTransferPages(policy,
                                                   key_count,
                                                   /*use_hybrid=*/false,
                                                   /*compact_tail=*/false);
    for (const auto page : pages) {
        const auto regions = queryProducerCacheBlockRegions(config,
                                                            0,
                                                            "full",
                                                            key_count,
                                                            page,
                                                            cp_rank,
                                                            cp_size,
                                                            /*opaque_wire=*/true,
                                                            block_ids.data(),
                                                            block_ids.size(),
                                                            {base});
        if (!regions.empty()) {
            pairs.emplace_back(regions.front().key_index, regions.front().local_slot);
        }
    }
    return pairs;
}

TEST(CpTransferProjectionTest, RankLocalKeysAndSlotsPreserveP1P2P4Vectors) {
    EXPECT_EQ(rankKeySlots(4, 0, 1), (std::vector<KeySlot>{{0, 0}, {1, 1}, {2, 2}, {3, 3}}));
    EXPECT_EQ(rankKeySlots(8, 1, 2), (std::vector<KeySlot>{{1, 0}, {3, 1}, {5, 2}, {7, 3}}));
    EXPECT_EQ(rankKeySlots(4, 0, 2), (std::vector<KeySlot>{{0, 0}, {2, 1}}));
    EXPECT_EQ(rankKeySlots(10, 3, 4), (std::vector<KeySlot>{{3, 0}, {7, 1}}));
}

TEST(CpTransferProjectionTest, ShortTailAndPaddingHaveNoProducerKey) {
    EXPECT_TRUE(rankKeySlots(0, 0, 4).empty());
    EXPECT_TRUE(rankKeySlots(1, 3, 4).empty());
    EXPECT_EQ(rankKeySlots(1, 0, 4), (std::vector<KeySlot>{{0, 0}}));
    EXPECT_EQ(rankKeySlots(5, 0, 4), (std::vector<KeySlot>{{0, 0}, {4, 1}}));
}

TEST(CpTransferProjectionTest, SourceQueryRejectsInvalidProducerTopology) {
    EXPECT_THROW((void)rankKeySlots(4, 2, 2), std::exception);
}

TEST(CpTransferProjectionTest, ExactSwaHandoffSkipsOnlyHistoryOutsideTheWindow) {
    auto policy                = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.cp_mapping          = CpBlockMappingMode::COMPACT_LAST_RANK;
    policy.sliding_window_size = 128;
    // CP2 has 512-token physical slots. At N1320, all 127 prior tokens are
    // in slot 2; immediately after N1024, the window still crosses slots.
    EXPECT_EQ(firstRequiredSwaHandoffPage(policy, 512, 1320), 2u);
    EXPECT_EQ(firstRequiredSwaHandoffPage(policy, 512, 1025), 1u);
    EXPECT_EQ(firstRequiredSwaHandoffPage(policy, 512, 1425), 2u);
    EXPECT_EQ(firstRequiredSwaHandoffPage(policy, 512, 3745), 7u);
    // Non-CP 256-token slots at N1320 need both tail slots.
    policy.cp_mapping = CpBlockMappingMode::NONE;
    EXPECT_EQ(firstRequiredSwaHandoffPage(policy, 256, 1320), 4u);
    policy.active_tail_blocks = 1;  // HCA_STATE has its own one-slot rule.
    EXPECT_EQ(firstRequiredSwaHandoffPage(policy, 512, 1320), 0u);
}

}  // namespace
}  // namespace rtp_llm::test
