#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>

#include "rtp_llm/cpp/cache/BlockPool.h"
#include "rtp_llm/cpp/cache/BlockPoolConfigHelper.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MLAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/test/CacheConfigTestUtils.h"

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
    auto config      = makeSingleGroupCacheConfig(std::move(spec), CacheGroupType::FULL, layer_num, block_num);
    config.use_mla   = true;
    config.is_sparse = true;
    return config;
}

uint8_t tokenMajorValue(int layer_id, int block_id, size_t token, size_t elem, DataType dtype) {
    return static_cast<uint8_t>((layer_id * 61 + block_id * 17 + token * 7 + elem * 3 + static_cast<int>(dtype))
                                % 251);
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

    constexpr uint8_t sentinel    = 0xA5;
    const size_t      block_bytes = spec->block_size_bytes();
    const size_t      token_elems = kv_lora_rank + rope_head_dim;
    for (int layer_id = 0; layer_id < layer_num; ++layer_id) {
        for (int block_id = 0; block_id < block_num; ++block_id) {
            auto addr = block_pool.convertIndexToAddr(layer_id, block_id);
            ASSERT_NE(addr.kv_addr, nullptr);
            std::memset(addr.kv_addr, sentinel, block_bytes);
        }
    }

    auto layer_tensors = block_pool.allLayerCacheBase();
    ASSERT_EQ(layer_tensors.size(), static_cast<size_t>(layer_num));
    for (int layer_id = 0; layer_id < layer_num; ++layer_id) {
        auto& layer_tensor = layer_tensors[static_cast<size_t>(layer_id)];
        ASSERT_EQ(layer_tensor.dim(), 3);
        ASSERT_EQ(layer_tensor.size(0), block_num);
        ASSERT_EQ(layer_tensor.size(1), static_cast<int64_t>(seq_size_per_block));
        ASSERT_EQ(layer_tensor.size(2), static_cast<int64_t>(token_elems));
        ASSERT_EQ(layer_tensor.element_size(), 1);

        auto* layer_base   = static_cast<uint8_t*>(layer_tensor.data_ptr());
        auto  block_stride = layer_tensor.stride(0) * static_cast<int64_t>(layer_tensor.element_size());
        auto  token_stride = layer_tensor.stride(1) * static_cast<int64_t>(layer_tensor.element_size());
        auto  elem_stride  = layer_tensor.stride(2) * static_cast<int64_t>(layer_tensor.element_size());
        ASSERT_EQ(block_stride, static_cast<int64_t>(block_bytes));
        ASSERT_EQ(token_stride, static_cast<int64_t>(token_elems));
        ASSERT_EQ(elem_stride, 1);

        auto first_addr = block_pool.convertIndexToAddr(layer_id, 0);
        auto last_addr  = block_pool.convertIndexToAddr(layer_id, block_num - 1);
        ASSERT_NE(first_addr.kv_addr, nullptr);
        ASSERT_NE(last_addr.kv_addr, nullptr);
        EXPECT_EQ(static_cast<uint8_t*>(last_addr.kv_addr) - static_cast<uint8_t*>(first_addr.kv_addr),
                  static_cast<ptrdiff_t>((block_num - 1) * block_stride));

        for (int block_id : {1, 2}) {
            for (size_t token = 0; token < seq_size_per_block; ++token) {
                for (size_t elem = 0; elem < token_elems; ++elem) {
                    layer_base[block_id * block_stride + token * token_stride + elem * elem_stride] =
                        tokenMajorValue(layer_id, block_id, token, elem, dtype);
                }
            }
        }

        for (int block_id : {1, 2}) {
            auto addr  = block_pool.convertIndexToAddr(layer_id, block_id);
            auto* base = static_cast<uint8_t*>(addr.kv_addr);
            ASSERT_NE(base, nullptr);
            for (size_t token = 0; token < seq_size_per_block; ++token) {
                for (size_t elem = 0; elem < token_elems; ++elem) {
                    const size_t offset = token * token_elems + elem;
                    EXPECT_EQ(base[offset], tokenMajorValue(layer_id, block_id, token, elem, dtype))
                        << "compact MLA cache mismatch at layer " << layer_id << ", block " << block_id
                        << ", token " << token << ", elem " << elem;
                }
            }
        }

        for (int block_id : {0, block_num - 1}) {
            auto addr  = block_pool.convertIndexToAddr(layer_id, block_id);
            auto* base = static_cast<uint8_t*>(addr.kv_addr);
            ASSERT_NE(base, nullptr);
            for (size_t offset = 0; offset < block_bytes; ++offset) {
                EXPECT_EQ(base[offset], sentinel)
                    << "sentinel changed at layer " << layer_id << ", block " << block_id << ", offset " << offset;
            }
        }
    }
}

#endif

}  // namespace
}  // namespace rtp_llm::test
