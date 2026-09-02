#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"

namespace rtp_llm::test {
namespace {

#if USING_ROCM

std::shared_ptr<MLAKVCacheSpec>
makeSparseMlaSpec(DataType dtype, size_t kv_lora_rank, size_t rope_head_dim, size_t seq_size_per_block) {
    AttentionConfigs attn{};
    attn.kv_lora_rank  = static_cast<int>(kv_lora_rank);
    attn.rope_head_dim = static_cast<int>(rope_head_dim);
    attn.is_sparse     = true;

    KVCacheSpecDesc desc;
    desc.tag        = "sparse_mla";
    desc.cache_type = KVCacheSpecType::MultiHeadLatentAttention;
    desc.dtype      = dtype;

    SpecBuildContext ctx;
    ctx.dtype              = dtype;
    ctx.seq_size_per_block = seq_size_per_block;
    ctx.attn_config        = &attn;

    return std::dynamic_pointer_cast<MLAKVCacheSpec>(MLAKVCacheSpec::build(desc, ctx));
}

CacheConfig makeSparseMlaBlockPoolConfig(KVCacheSpecPtr spec, int layer_num, int block_num) {
    CacheConfig config;
    config.dtype                     = spec->memoryLayoutDType();
    config.layer_num                 = static_cast<uint32_t>(layer_num);
    config.layer_all_num             = static_cast<uint32_t>(layer_num);
    config.block_num                 = static_cast<uint32_t>(block_num);
    config.seq_size_per_block        = spec->seq_size_per_block;
    config.kernel_seq_size_per_block = spec->seq_size_per_block;
    config.use_mla                   = true;
    config.is_sparse                 = true;

    std::vector<int> layer_ids(static_cast<size_t>(layer_num));
    std::iota(layer_ids.begin(), layer_ids.end(), 0);
    config.fromGroupedSpecs({spec}, {layer_ids}, {CacheGroupType::FULL}, {spec->tag});

    config.kv_block_stride_bytes = spec->block_size_bytes();
    config.kv_block_size_bytes   = static_cast<size_t>(layer_num) * config.kv_block_stride_bytes;
    config.kv_scale_stride_bytes = spec->scale_block_size_bytes();
    config.kv_scale_size_bytes   = static_cast<size_t>(layer_num) * config.kv_scale_stride_bytes;
    config.block_size_bytes      = config.kv_block_size_bytes + config.kv_scale_size_bytes;

    const size_t layer_stride_bytes = config.kv_block_stride_bytes + config.kv_scale_stride_bytes;
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(layer_num), static_cast<int>(layer_stride_bytes));
    return config;
}

void writeSparseMlaBlock(uint8_t* base,
                         size_t   kv_lora_rank,
                         size_t   rope_head_dim,
                         size_t   seq_size_per_block,
                         int      layer_id,
                         int      block_id,
                         DataType dtype) {
    auto* k_base = base;
    auto* v_base = base + kv_lora_rank * seq_size_per_block;
    for (size_t token = 0; token < seq_size_per_block; ++token) {
        for (size_t offset = 0; offset < kv_lora_rank; ++offset) {
            k_base[token * kv_lora_rank + offset] =
                static_cast<uint8_t>((layer_id * 61 + block_id * 17 + token * 7 + offset + static_cast<int>(dtype))
                                     % 251);
        }
        for (size_t offset = 0; offset < rope_head_dim; ++offset) {
            v_base[token * rope_head_dim + offset] =
                static_cast<uint8_t>((layer_id * 43 + block_id * 19 + token * 11 + offset + 97
                                      + static_cast<int>(dtype))
                                     % 251);
        }
    }
}

TEST(MLAKVCacheBlockPoolTest, RocmSparseFp8CompactSpecAllowsUnalignedRankForAllFp8Dtypes) {
    constexpr size_t kv_lora_rank       = 160;
    constexpr size_t rope_head_dim      = 32;
    constexpr size_t seq_size_per_block = 2;

    for (auto dtype : {DataType::TYPE_FP8_E4M3, DataType::TYPE_FP8_E8M0}) {
        SCOPED_TRACE(static_cast<int>(dtype));

        auto spec = makeSparseMlaSpec(dtype, kv_lora_rank, rope_head_dim, seq_size_per_block);
        ASSERT_NE(spec, nullptr);
        ASSERT_EQ(getTypeSize(dtype), 1u);
        EXPECT_EQ(spec->k_block_size(), kv_lora_rank * seq_size_per_block);
        EXPECT_EQ(spec->v_block_size(), rope_head_dim * seq_size_per_block);
        EXPECT_EQ(spec->block_size(), (kv_lora_rank + rope_head_dim) * seq_size_per_block);
    }
}

TEST(MLAKVCacheBlockPoolTest, RocmSparseFp8CompactLayoutReadsWritesAcrossTokensAndBlocks) {
    constexpr DataType dtype              = DataType::TYPE_FP8_E4M3;
    constexpr size_t   kv_lora_rank       = 160;
    constexpr size_t   rope_head_dim      = 32;
    constexpr size_t   seq_size_per_block = 2;
    constexpr int      layer_num          = 2;
    constexpr int      block_num          = 4;

    auto spec = makeSparseMlaSpec(dtype, kv_lora_rank, rope_head_dim, seq_size_per_block);
    ASSERT_NE(spec, nullptr);
    ASSERT_EQ(getTypeSize(dtype), 1u);
    EXPECT_EQ(spec->k_block_size(), kv_lora_rank * seq_size_per_block);
    EXPECT_EQ(spec->v_block_size(), rope_head_dim * seq_size_per_block);
    EXPECT_EQ(spec->block_size(), (kv_lora_rank + rope_head_dim) * seq_size_per_block);

    auto config      = makeSparseMlaBlockPoolConfig(spec, layer_num, block_num);
    auto pool_config = BlockPoolConfigHelper::createConfig(config);
    BlockPool block_pool(pool_config, AllocationType::HOST);
    ASSERT_TRUE(block_pool.init());

    const size_t block_bytes = spec->block_size_bytes();
    for (int layer_id = 0; layer_id < layer_num; ++layer_id) {
        auto first_addr = block_pool.convertIndexToAddr(layer_id, 0);
        auto last_addr  = block_pool.convertIndexToAddr(layer_id, block_num - 1);
        ASSERT_NE(first_addr.kv_addr, nullptr);
        ASSERT_NE(last_addr.kv_addr, nullptr);
        EXPECT_EQ(static_cast<uint8_t*>(last_addr.kv_addr) - static_cast<uint8_t*>(first_addr.kv_addr),
                  static_cast<ptrdiff_t>((block_num - 1) * block_bytes));

        for (int block_id : {0, block_num - 1}) {
            auto addr  = block_pool.convertIndexToAddr(layer_id, block_id);
            auto* base = static_cast<uint8_t*>(addr.kv_addr);
            ASSERT_NE(base, nullptr);
            writeSparseMlaBlock(base, kv_lora_rank, rope_head_dim, seq_size_per_block, layer_id, block_id, dtype);

            std::vector<uint8_t> expected(block_bytes);
            writeSparseMlaBlock(
                expected.data(), kv_lora_rank, rope_head_dim, seq_size_per_block, layer_id, block_id, dtype);
            EXPECT_EQ(std::memcmp(base, expected.data(), block_bytes), 0)
                << "compact MLA cache mismatch at layer " << layer_id << ", block " << block_id;
        }
    }
}

#endif

}  // namespace
}  // namespace rtp_llm::test
