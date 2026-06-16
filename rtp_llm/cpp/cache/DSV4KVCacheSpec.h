#pragma once

#include <algorithm>
#include <numeric>

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

// FP8 KV slot byte sizes mirror the canonical fp8_model1_mla layout written
// by Python (rtp_llm/models_py/modules/dsv4/fp8/_compressor_vllm_triton.py).
// 584B = 448 fp8 NoPE + 64 bf16 RoPE + 8 UE8M0 scales; 132B = 128 fp8 + 4 fp32
// scale. FlashMLA SM100 sparse_attn (head64 instantiations) hard-requires
// ``k_cache.stride(0) % TMA_K_STRIDE == 0``; TMA_K_STRIDE for the FP8
// path is 576 bytes (= 9 cachelines x 64B). Natural 256 * 584 = 149504 fails
// (149504 % 576 = 320), so block_size_bytes() returns the physical padded stride.
inline constexpr uint32_t DSV4_FP8_KV_ENTRY_BYTES            = 584;
inline constexpr uint32_t DSV4_FP8_INDEXER_ENTRY_BYTES       = 132;
inline constexpr size_t   DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES = 576;
// SWA window in tokens. A full SWA_KV ring has >= this many entries; a CP slice
// has fewer. Used to tell them apart in DSV4StateSpec::block_size_bytes().
inline constexpr uint32_t DSV4_SWA_WINDOW_ENTRIES = 128;

inline size_t alignDsv4Fp8KvBlockBytes(size_t natural, size_t extra_multiple = 1) {
    const size_t align = std::lcm(DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES, std::max<size_t>(extra_multiple, 1));
    return ((natural + align - 1) / align) * align;
}

// KVCacheSpec for compressed paged KV pools. These are variable-length paged
// pools storing KV entries as uint8 (byte-addressed).
// BF16 mode: head_dim * 2 bytes per entry (1024 KV / 256 Indexer).
// FP8 mode: 584B per KV entry, 132B per Indexer entry — see constants above.
// Each entry contains the full KV for one compressed token (one KV head).
struct DSV4KVSpec: public KVCacheSpec {
    KVCacheRegionName cache_type = KVCacheRegionName::DEFAULT;
    uint32_t          entry_elems;        // bytes per entry (1024/584 KV, 256/132 Indexer)
    uint32_t          entries_per_block;  // entries per block (64 or 2)
    uint32_t          compression_ratio           = 1;
    DataType          store_dtype                 = DataType::TYPE_INVALID;
    size_t            block_size_bytes_alignment = 0;

    DSV4KVSpec() {
        type      = KVCacheSpecType::CompressedKV;
        lifecycle = CacheGroupType::FULL;
    }

    DSV4KVSpec(KVCacheRegionName cache_region,
               uint32_t          entry_elements,
               uint32_t          block_entries,
               DataType          storage_dtype,
               uint32_t          seq_size_per_blk,
               uint32_t          cache_compression_ratio = 1,
               size_t            block_size_alignment    = 0)
        : DSV4KVSpec() {
        cache_type        = cache_region;
        entry_elems       = entry_elements;
        entries_per_block = block_entries;
        compression_ratio = cache_compression_ratio;
        store_dtype       = storage_dtype;
        block_size_bytes_alignment = block_size_alignment;

        // KVCacheSpec base fields
        local_head_num_kv  = 1;  // DSV4 is MQA (1 KV head)
        seq_size_per_block = seq_size_per_blk;
        dtype              = store_dtype;
    }

    size_t block_size() const override {
        return static_cast<size_t>(entries_per_block) * entry_elems;
    }

    // K/V symmetric split — each entry has interleaved K and V components
    size_t k_block_size() const override {
        return block_size() / 2;
    }
    size_t v_block_size() const override {
        return block_size() / 2;
    }

    size_t natural_block_size_bytes() const {
        return static_cast<size_t>(entries_per_block) * entry_elems * getTypeSize(store_dtype);
    }

    // Public block size is the physical per-block stride. FP8 KV pools need
    // padding for FlashMLA TMA; BF16 and indexer pools keep their natural size.
    size_t block_size_bytes() const override {
        const size_t natural = natural_block_size_bytes();
        if (block_size_bytes_alignment > 0) {
            return ((natural + block_size_bytes_alignment - 1) / block_size_bytes_alignment)
                   * block_size_bytes_alignment;
        }
        return natural;
    }

    size_t k_block_size_bytes() const override {
        return block_size_bytes() / 2;
    }
    size_t v_block_size_bytes() const override {
        return block_size_bytes() / 2;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<DSV4KVSpec>(*this);
    }

protected:
    std::string fingerprintExtra() const override {
        std::ostringstream os;
        os << ";dsv4kv.cache_type=" << static_cast<int>(cache_type) << ";dsv4kv.entry_elems=" << entry_elems
           << ";dsv4kv.compression_ratio=" << compression_ratio
           << ";dsv4kv.store_dtype=" << static_cast<int>(store_dtype)
           << ";dsv4kv.block_size_bytes_alignment=" << block_size_bytes_alignment;
        return os.str();
    }

public:
    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "DSV4KVSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent + 2, ' ') << "cache_type=" << static_cast<int>(cache_type) << "\n";
        os << std::string(indent + 2, ' ') << "entry_elems=" << entry_elems << "\n";
        os << std::string(indent + 2, ' ') << "entries_per_block=" << entries_per_block << "\n";
        os << std::string(indent + 2, ' ') << "compression_ratio=" << compression_ratio << "\n";
        os << std::string(indent + 2, ' ') << "block_size_bytes_alignment=" << block_size_bytes_alignment << "\n";
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

// KVCacheSpec for fixed-allocation state/SWA payload pools. State pools store
// compressor/indexer state as float32; SWA_KV stores byte-addressed KV entries.
// They use SWA tail allocation, and non-null tail blocks can participate in
// prefix cache. The K/V split is a placeholder for state pools because state is
// an opaque blob.
struct DSV4StateSpec: public KVCacheSpec {
    KVCacheRegionName cache_type = KVCacheRegionName::DEFAULT;
    uint32_t          state_dim;          // state dimension (entry_elems in pool_spec)
    uint32_t          entries_per_block;  // 4 or 8
    DataType          store_dtype                      = DataType::TYPE_INVALID;
    size_t            block_size_bytes_override         = 0;
    size_t            block_size_bytes_alignment        = 0;
    uint32_t          block_size_alignment_min_entries = 0;

    DSV4StateSpec() {
        type      = KVCacheSpecType::FixedState;
        lifecycle = CacheGroupType::SWA;
    }

    DSV4StateSpec(KVCacheRegionName cache_region,
                  uint32_t          state_elements,
                  uint32_t          block_entries,
                  DataType          storage_dtype,
                  uint32_t          seq_size_per_blk,
                  size_t            block_size_bytes_override_value = 0,
                  size_t            block_size_alignment            = 0,
                  uint32_t          block_alignment_min_entries     = 0,
                  bool              state_cache                    = true,
                  bool              skip_reuse                     = false)
        : DSV4StateSpec() {
        cache_type        = cache_region;
        state_dim         = state_elements;
        entries_per_block = block_entries;
        store_dtype               = storage_dtype;
        block_size_bytes_override = block_size_bytes_override_value;
        block_size_bytes_alignment        = block_size_alignment;
        block_size_alignment_min_entries = block_alignment_min_entries;

        // KVCacheSpec base fields
        local_head_num_kv  = 1;
        seq_size_per_block = seq_size_per_blk;
        dtype              = store_dtype;
        is_state_cache     = state_cache;
        skip_prefix_reuse  = skip_reuse;
    }

    size_t block_size() const override {
        return static_cast<size_t>(entries_per_block) * state_dim;
    }

    // Placeholder K/V split — state is opaque, no real K/V semantic
    size_t k_block_size() const override {
        return block_size() / 2;
    }
    size_t v_block_size() const override {
        return block_size() / 2;
    }

    size_t natural_block_size_bytes() const {
        return static_cast<size_t>(entries_per_block) * state_dim * getTypeSize(store_dtype);
    }

    // Public block size is the physical per-block stride. The full (unsliced)
    // SWA_KV ring stores the FP8 KV layout and needs TMA padding; a prefill
    // CP-sliced sub-block (< window) keeps its natural size.
    size_t block_size_bytes() const override {
        if (block_size_bytes_override > 0) {
            return block_size_bytes_override;
        }
        const size_t natural = natural_block_size_bytes();
        if (block_size_bytes_alignment > 0 && entries_per_block >= block_size_alignment_min_entries) {
            return ((natural + block_size_bytes_alignment - 1) / block_size_bytes_alignment)
                   * block_size_bytes_alignment;
        }
        return natural;
    }

    size_t k_block_size_bytes() const override {
        return block_size_bytes() / 2;
    }
    size_t v_block_size_bytes() const override {
        return block_size_bytes() / 2;
    }

    KVCacheSpecPtr clone() const override {
        return std::make_shared<DSV4StateSpec>(*this);
    }

protected:
    std::string fingerprintExtra() const override {
        std::ostringstream os;
        os << ";dsv4state.cache_type=" << static_cast<int>(cache_type) << ";dsv4state.state_dim=" << state_dim
           << ";dsv4state.store_dtype=" << static_cast<int>(store_dtype)
           << ";dsv4state.block_size_bytes_override=" << block_size_bytes_override
           << ";dsv4state.block_size_bytes_alignment=" << block_size_bytes_alignment
           << ";dsv4state.block_size_alignment_min_entries=" << block_size_alignment_min_entries;
        return os.str();
    }

public:
    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "DSV4StateSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent + 2, ' ') << "cache_type=" << static_cast<int>(cache_type) << "\n";
        os << std::string(indent + 2, ' ') << "state_dim=" << state_dim << "\n";
        os << std::string(indent + 2, ' ') << "entries_per_block=" << entries_per_block << "\n";
        os << std::string(indent + 2, ' ') << "block_size_bytes_override=" << block_size_bytes_override << "\n";
        os << std::string(indent + 2, ' ') << "block_size_bytes_alignment=" << block_size_bytes_alignment << "\n";
        os << std::string(indent + 2, ' ')
           << "block_size_alignment_min_entries=" << block_size_alignment_min_entries << "\n";
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

}  // namespace rtp_llm
