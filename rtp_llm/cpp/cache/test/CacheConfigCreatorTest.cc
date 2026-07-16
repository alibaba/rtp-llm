#include <gtest/gtest.h>

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

namespace rtp_llm::test {
namespace {

std::vector<int> layerRange(int begin, int end) {
    std::vector<int> layers(static_cast<size_t>(end - begin));
    std::iota(layers.begin(), layers.end(), begin);
    return layers;
}

TEST(CacheConfigCreatorTest, MergeMtpAlignsUniqueFullGroupWhenTagsDiffer) {
    constexpr int kMainLayerNum   = 64;
    constexpr int kGroupLayerNum  = 16;
    constexpr int kMtpGlobalLayer = kMainLayerNum;
    constexpr int kBlockNum       = 8;
    constexpr int kTokensPerBlock = 1024;

    CacheConfig score_config;
    score_config.dtype                     = DataType::TYPE_BF16;
    score_config.layer_num                 = kMainLayerNum;
    score_config.layer_all_num             = kMainLayerNum + 1;
    score_config.block_num                 = kBlockNum;
    score_config.seq_size_per_block        = kTokensPerBlock;
    score_config.kernel_seq_size_per_block = 16;
    score_config.group_layer_num           = kGroupLayerNum;

    // Qwen3.5-27B on TP2 has 2 local full-attention KV heads of width 256.
    auto full_spec    = makeMhaSpec("full", kTokensPerBlock, DataType::TYPE_BF16, 2, 256);
    auto linear0_spec = makeLinearSpec("linear0", kTokensPerBlock, DataType::TYPE_BF16, 24, 128);
    auto linear1_spec = makeLinearSpec("linear1", kTokensPerBlock, DataType::TYPE_BF16, 24, 128);
    auto linear2_spec = makeLinearSpec("linear2", kTokensPerBlock, DataType::TYPE_BF16, 24, 128);
    score_config.fromGroupedSpecs(
        {full_spec, linear0_spec, linear1_spec, linear2_spec},
        {layerRange(0, 16), layerRange(16, 32), layerRange(32, 48), layerRange(48, 64)},
        {CacheGroupType::FULL, CacheGroupType::LINEAR, CacheGroupType::LINEAR, CacheGroupType::LINEAR},
        {"full", "linear0", "linear1", "linear2"});
    score_config.layer_to_block_stride_bytes.assign(kMainLayerNum + 1, 1);

    // The standalone Qwen2 draft has 4 local KV heads of width 128 on TP2 and uses "default".
    auto propose_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, kBlockNum, kTokensPerBlock, DataType::TYPE_BF16, /*local_head_num_kv=*/4, /*head_dim=*/128);

    auto sub_config = score_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/kMainLayerNum);

    EXPECT_EQ(score_config.groupTagsSnapshot(), std::vector<std::string>({"full", "linear0", "linear1", "linear2"}));
    ASSERT_EQ(score_config.layerIdsForGroup(0).size(), kGroupLayerNum + 1);
    EXPECT_EQ(score_config.layerIdsForGroup(0).back(), kMtpGlobalLayer);
    EXPECT_EQ(score_config.groupIdsForLayer(kMtpGlobalLayer), std::vector<int>({0}));
    EXPECT_EQ(score_config.groupIdForLayerTag(kMtpGlobalLayer, "full"), 0);

    ASSERT_NE(sub_config, nullptr);
    ASSERT_EQ(sub_config->groupNums(), 4);
    EXPECT_EQ(sub_config->groupTagsSnapshot(), std::vector<std::string>({"default", "linear0", "linear1", "linear2"}));
    EXPECT_EQ(sub_config->layerIdsForGroup(0), std::vector<int>({0}));
    EXPECT_TRUE(sub_config->layerIdsForGroup(1).empty());
    EXPECT_TRUE(sub_config->layerIdsForGroup(2).empty());
    EXPECT_TRUE(sub_config->layerIdsForGroup(3).empty());
    EXPECT_EQ(sub_config->groupIdsForLayer(0), std::vector<int>({0}));
    EXPECT_EQ(sub_config->groupIdForLayerTag(0, "default"), 0);
}

TEST(CacheConfigCreatorTest, MergeMtpRejectsAmbiguousTypeOnlyMapping) {
    constexpr int kMainLayerNum   = 2;
    constexpr int kBlockNum       = 8;
    constexpr int kTokensPerBlock = 1024;

    CacheConfig score_config;
    score_config.dtype                     = DataType::TYPE_BF16;
    score_config.layer_num                 = kMainLayerNum;
    score_config.layer_all_num             = kMainLayerNum + 1;
    score_config.block_num                 = kBlockNum;
    score_config.seq_size_per_block        = kTokensPerBlock;
    score_config.kernel_seq_size_per_block = 16;
    score_config.group_layer_num           = 1;
    score_config.fromGroupedSpecs({makeMhaSpec("full0", kTokensPerBlock, DataType::TYPE_BF16, 2, 256),
                                   makeMhaSpec("full1", kTokensPerBlock, DataType::TYPE_BF16, 2, 256)},
                                  {{0}, {1}},
                                  {CacheGroupType::FULL, CacheGroupType::FULL},
                                  {"full0", "full1"});
    score_config.layer_to_block_stride_bytes.assign(kMainLayerNum + 1, 1);

    auto propose_config = makeSimpleMhaCacheConfig(
        /*layer_num=*/1, kBlockNum, kTokensPerBlock, DataType::TYPE_BF16, /*local_head_num_kv=*/4, /*head_dim=*/128);

    EXPECT_THROW(score_config.mergeMTPModule(propose_config, /*module_index=*/0, /*main_layer_num=*/kMainLayerNum),
                 std::runtime_error);
}

}  // namespace
}  // namespace rtp_llm::test
