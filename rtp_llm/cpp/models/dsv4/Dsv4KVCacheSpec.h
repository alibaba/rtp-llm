#pragma once

#include <algorithm>
#include <numeric>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"

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
struct DSV4KVSpec: public OpaqueKVCacheSpec {
    KVCacheRegionName cache_type = KVCacheRegionName::DEFAULT;
    uint32_t          compression_ratio           = 1;

    DSV4KVSpec() {
        type      = KVCacheSpecType::OpaqueKV;
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

    KVCacheSpecPtr clone() const override {
        return std::make_shared<DSV4KVSpec>(*this);
    }

    KVCacheRegionName regionName() const override {
        return cache_type;
    }

protected:
    std::string fingerprintExtra() const override {
        std::ostringstream os;
        os << ";dsv4kv.cache_type=" << static_cast<int>(cache_type)
           << ";dsv4kv.compression_ratio=" << compression_ratio
           << opaqueFingerprintExtra("dsv4kv");
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
struct DSV4StateSpec: public OpaqueKVCacheSpec {
    KVCacheRegionName cache_type = KVCacheRegionName::DEFAULT;
    uint32_t&         state_dim;

    DSV4StateSpec(): state_dim(entry_elems) {
        type      = KVCacheSpecType::OpaqueState;
        lifecycle = CacheGroupType::SWA;
    }

    DSV4StateSpec(const DSV4StateSpec& other): OpaqueKVCacheSpec(other), cache_type(other.cache_type), state_dim(entry_elems) {}

    DSV4StateSpec& operator=(const DSV4StateSpec& other) {
        if (this != &other) {
            OpaqueKVCacheSpec::operator=(other);
            cache_type = other.cache_type;
        }
        return *this;
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

    KVCacheSpecPtr clone() const override {
        return std::make_shared<DSV4StateSpec>(*this);
    }

    KVCacheRegionName regionName() const override {
        return cache_type;
    }

protected:
    std::string fingerprintExtra() const override {
        std::ostringstream os;
        os << ";dsv4state.cache_type=" << static_cast<int>(cache_type) << opaqueFingerprintExtra("dsv4state");
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
