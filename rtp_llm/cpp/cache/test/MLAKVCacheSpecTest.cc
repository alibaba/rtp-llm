#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/config/StaticConfig.h"
#include "rtp_llm/cpp/utils/Exception.h"

namespace rtp_llm::test {
namespace {

std::shared_ptr<MLAKVCacheSpec>
makeMlaSpec(DataType dtype, bool is_sparse, size_t kv_lora_rank = 512) {
    AttentionConfigs attn{};
    attn.kv_lora_rank  = kv_lora_rank;
    attn.rope_head_dim = 64;
    attn.is_sparse     = is_sparse;

    KVCacheSpecDesc desc;
    desc.tag        = "mla";
    desc.cache_type = KVCacheSpecType::MultiHeadLatentAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype              = dtype;
    ctx.seq_size_per_block = 1;
    ctx.attn_config        = &attn;

    return std::dynamic_pointer_cast<MLAKVCacheSpec>(
        MLAKVCacheSpec::build(desc, ctx));
}

void expectBlockSizeBytesIfSupported(const MLAKVCacheSpec& spec, size_t expected_elems) {
    const size_t dtype_size = getTypeSize(spec.memoryLayoutDType());
    if (dtype_size == 0) {
        return;
    }
    EXPECT_EQ(spec.block_size_bytes(), expected_elems * dtype_size);
}

class MLAKVCacheSpecTest: public ::testing::Test {
protected:
    void SetUp() override {
        old_core_dump_on_exception_                  = StaticConfig::user_ft_core_dump_on_exception;
        StaticConfig::user_ft_core_dump_on_exception  = false;
    }
    void TearDown() override {
        StaticConfig::user_ft_core_dump_on_exception  = old_core_dump_on_exception_;
    }
private:
    bool old_core_dump_on_exception_;
};

TEST_F(MLAKVCacheSpecTest, Bf16LayoutDoesNotDependOnSparseMode) {
    auto dense_spec  = makeMlaSpec(DataType::TYPE_BF16, false);
    auto sparse_spec = makeMlaSpec(DataType::TYPE_BF16, true);

    constexpr size_t expected_bytes = (512 + 64) * 2;
    EXPECT_EQ(dense_spec->block_size_bytes(), expected_bytes);
    EXPECT_EQ(sparse_spec->block_size_bytes(), expected_bytes);
}

TEST_F(MLAKVCacheSpecTest, DenseFp8UsesNativeLayout) {
    auto spec = makeMlaSpec(DataType::TYPE_FP8_E4M3, false);

    constexpr size_t expected_elems = 512 + 512 / 128 * 4 + 64 * 2;
    EXPECT_EQ(spec->block_size(), expected_elems);
    expectBlockSizeBytesIfSupported(*spec, expected_elems);
}

TEST_F(MLAKVCacheSpecTest, SparseFp8UsesPlatformLayout) {
    auto spec = makeMlaSpec(DataType::TYPE_FP8_E4M3, true);

#if USING_ROCM
    constexpr size_t expected_elems = 512 + 64;
    EXPECT_EQ(spec->block_size(), expected_elems);
    EXPECT_EQ(spec->k_block_size() + spec->v_block_size(), spec->block_size());
#else
    constexpr size_t expected_elems = 512 + 512 / 128 * 4 + 64 * 2;
    EXPECT_EQ(spec->block_size(), expected_elems);
#endif
    expectBlockSizeBytesIfSupported(*spec, expected_elems);
}

TEST_F(MLAKVCacheSpecTest, DenseFp8RejectsUnalignedLoraRankForAllFp8Dtypes) {
    EXPECT_THROW(makeMlaSpec(DataType::TYPE_FP8_E4M3, false, 100), rtp_llm::RTPException);
    EXPECT_THROW(makeMlaSpec(DataType::TYPE_FP8_E8M0, false, 100), rtp_llm::RTPException);
}

}  // namespace
}  // namespace rtp_llm::test