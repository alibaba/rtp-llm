#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"

namespace rtp_llm {
namespace test {
namespace {

class FixedMLASpec final: public KVCacheSpec {
public:
    explicit FixedMLASpec(size_t block_size_bytes): block_size_bytes_(block_size_bytes) {
        type               = KVCacheSpecType::MultiHeadLatentAttention;
        tag                = "default";
        seq_size_per_block = 4;
    }

    size_t block_size() const override {
        return block_size_bytes_ / 2;
    }

    size_t k_block_size() const override {
        return block_size() - v_block_size();
    }

    size_t v_block_size() const override {
        return 4;
    }

    size_t block_size_bytes() const override {
        return block_size_bytes_;
    }

    size_t k_block_size_bytes() const override {
        return block_size_bytes_ - v_block_size_bytes();
    }

    size_t v_block_size_bytes() const override {
        return 8;
    }

    DataType memoryLayoutDType() const override {
        return DataType::TYPE_BF16;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<FixedMLASpec>(*this);
    }

    std::string debugString(size_t) const override {
        return "FixedMLASpec";
    }

private:
    size_t block_size_bytes_;
};

CacheConfig makeCacheConfig(uint32_t layer_num, bool sparse, size_t scale_stride_bytes) {
    CacheConfig config;
    config.dtype                     = DataType::TYPE_BF16;
    config.layer_num                 = layer_num;
    config.layer_all_num             = layer_num;
    config.use_mla                   = true;
    config.is_sparse                 = sparse;
    config.block_num                 = 4;
    config.seq_size_per_block        = 4;
    config.kernel_seq_size_per_block = 4;
    config.kv_block_stride_bytes     = 64;
    config.kv_scale_stride_bytes     = scale_stride_bytes;

    GroupBase group;
    group.spec                  = std::make_shared<FixedMLASpec>(config.kv_block_stride_bytes);
    group.policy                = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.local_kv_head_num     = 1;
    group.kv_block_stride_bytes = config.kv_block_stride_bytes;
    group.kv_scale_stride_bytes = config.kv_scale_stride_bytes;
    for (uint32_t layer_id = 0; layer_id < layer_num; ++layer_id) {
        group.layer_ids.push_back(static_cast<int>(layer_id));
    }
    config.groups.push_back(std::move(group));
    return config;
}

TEST(BlockPoolConfigHelperTest, MTPSparseIndexerUsesProposeScaleStride) {
    auto score_config = makeCacheConfig(/*layer_num=*/2, /*sparse=*/true, /*scale_stride_bytes=*/32);
    auto propose_config =
        std::make_shared<CacheConfig>(makeCacheConfig(/*layer_num=*/1, /*sparse=*/true, /*scale_stride_bytes=*/128));
    score_config.mtp_sub_configs.push_back(propose_config);

    ASSERT_TRUE(propose_config->is_sparse);
    ASSERT_EQ(propose_config->specForGroup(0)->scale_block_size_bytes(), 0u);
    ASSERT_GT(propose_config->kv_scale_stride_bytes, 0u);

    const auto pool_config = BlockPoolConfigHelper::createConfig(score_config);
    ASSERT_EQ(pool_config.memory_layouts.size(), 2u);
    const auto& mtp_layout = pool_config.memory_layouts[1];
    EXPECT_TRUE(mtp_layout.is_mla);
    EXPECT_TRUE(mtp_layout.hasScale());
    EXPECT_EQ(mtp_layout.kv_scale_stride_bytes, propose_config->kv_scale_stride_bytes);
    EXPECT_GT(mtp_layout.kv_scale_pool_size_bytes, 0u);
}

}  // namespace
}  // namespace test
}  // namespace rtp_llm
