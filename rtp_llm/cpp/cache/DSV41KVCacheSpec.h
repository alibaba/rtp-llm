#pragma once

#include <algorithm>
#include <numeric>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"

namespace rtp_llm {

enum class DSV41KVEncoding {
    SWA_FP8_UE8M0,
    GLOBAL_FP4_E4M3,
    INDEX_FP4_UE8M0,
    PAIR_FP32,
};

// Byte-addressed V4.1 rows contain payload followed by per-group scales.
// Pair snapshots contain FP32 partial KV/scores, int64 position and int32 valid.
struct DSV41KVCacheSpec: public KVCacheSpec {
    static constexpr uint32_t kLayoutVersion      = 1;
    static constexpr uint32_t kPairKvOffset       = 0;
    static constexpr uint32_t kPairScoreOffset    = 512 * sizeof(float);
    static constexpr uint32_t kPairPositionOffset = 2 * 512 * sizeof(float);
    static constexpr uint32_t kPairValidOffset    = kPairPositionOffset + sizeof(int64_t);
    static constexpr uint32_t kPairSnapshotBytes  = 4112;

    KVCacheRegionName region;
    DSV41KVEncoding   encoding;
    uint32_t          ratio;
    uint32_t          entry_bytes;
    uint32_t          entries_per_block;
    uint32_t          quant_group_size;
    uint32_t          cp_size;
    bool              prefill_byte_slice;

    DSV41KVCacheSpec(KVCacheRegionName region_name,
                     uint32_t          layers,
                     uint32_t          compress_ratio,
                     uint32_t          entries,
                     uint32_t          token_block_size,
                     uint32_t          context_parallel_size,
                     bool              byte_slice):
        region(region_name),
        ratio(compress_ratio),
        entries_per_block(entries),
        cp_size(context_parallel_size),
        prefill_byte_slice(byte_slice) {
        RTP_LLM_CHECK_WITH_INFO(entries > 0 && (cp_size == 1 || cp_size == 8), "invalid V4.1 page geometry");
        layer_num          = layers;
        local_head_num_kv  = 1;
        seq_size_per_block = token_block_size;
        type               = KVCacheSpecType::MultiHeadAttention;
        dtype              = DataType::TYPE_UINT8;
        switch (region) {
            case KVCacheRegionName::SWA_KV:
                encoding         = DSV41KVEncoding::SWA_FP8_UE8M0;
                entry_bytes      = 528;
                quant_group_size = 32;
                RTP_LLM_CHECK_WITH_INFO(ratio == 0 && entries >= 128, "invalid V4.1 SWA ring");
                break;
            case KVCacheRegionName::DSV41_GLOBAL_KV:
                encoding         = DSV41KVEncoding::GLOBAL_FP4_E4M3;
                entry_bytes      = 288;
                quant_group_size = 16;
                RTP_LLM_CHECK_WITH_INFO((ratio == 1 || ratio == 2) && !byte_slice, "invalid V4.1 global page");
                break;
            case KVCacheRegionName::DSV41_INDEX_KV:
                encoding         = DSV41KVEncoding::INDEX_FP4_UE8M0;
                entry_bytes      = 68;
                quant_group_size = 32;
                RTP_LLM_CHECK_WITH_INFO((ratio == 1 || ratio == 2) && !byte_slice, "invalid V4.1 index page");
                break;
            case KVCacheRegionName::DSV41_PAIR_STATE:
                encoding         = DSV41KVEncoding::PAIR_FP32;
                entry_bytes      = kPairSnapshotBytes;
                quant_group_size = 0;
                RTP_LLM_CHECK_WITH_INFO(ratio == 2, "V4.1 pair state requires ratio2");
                break;
            default:
                RTP_LLM_FAIL("unsupported V4.1 cache region %d", static_cast<int>(region));
        }
    }

    size_t full_block_size_bytes() const {
        constexpr size_t alignment = 512;
        const size_t     natural   = static_cast<size_t>(entry_bytes) * entries_per_block;
        return ((natural + alignment - 1) / alignment) * alignment;
    }

    size_t block_size_bytes() const override {
        return full_block_size_bytes() / (prefill_byte_slice ? cp_size : 1);
    }
    size_t block_size() const override {
        return block_size_bytes();
    }
    size_t k_block_size() const override {
        return block_size_bytes() / 2;
    }
    size_t v_block_size() const override {
        return block_size_bytes() / 2;
    }
    size_t k_block_size_bytes() const override {
        return k_block_size();
    }
    size_t v_block_size_bytes() const override {
        return v_block_size();
    }

    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "DSV41KVCacheSpec{version=1, region=" << static_cast<int>(region)
           << ", encoding=" << static_cast<int>(encoding) << ", ratio=" << ratio << ", entry_bytes=" << entry_bytes
           << ", entries=" << entries_per_block << ", quant_group=" << quant_group_size << ", cp_size=" << cp_size
           << ", prefill_byte_slice=" << prefill_byte_slice << ", physical_stride=" << block_size_bytes() << "}\n";
        return os.str();
    }
};

}  // namespace rtp_llm
