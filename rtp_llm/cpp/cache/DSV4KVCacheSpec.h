#pragma once

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

// FP8 KV slot byte sizes mirror the canonical fp8_model1_mla layout written
// by Python (rtp_llm/models_py/modules/dsv4/fp8/_compressor_vllm_triton.py).
// 584B = 448 fp8 NoPE + 64 bf16 RoPE + 8 UE8M0 scales; 132B = 128 fp8 + 4 fp32
// scale. FlashMLA SM100 sparse_attn (head64 instantiations) hard-requires
// ``k_cache.stride(0) % TMA_K_STRIDE == 0``; TMA_K_STRIDE for the FP8
// path is 576 bytes (= 9 cachelines × 64B). Natural 256·584 = 149504 fails
// (149504 % 576 = 320), so block_size_bytes() rounds up to 576-aligned.
inline constexpr uint32_t DSV4_FP8_KV_ENTRY_BYTES            = 584;
inline constexpr uint32_t DSV4_FP8_INDEXER_ENTRY_BYTES       = 132;
inline constexpr size_t   DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES = 576;

// KVCacheSpec for DSV4 paged KV pools (Pool 0/1/2/6: CSA_KV, HCA_KV, INDEXER_KV, SWA_KV).
// These are variable-length paged pools storing KV entries as uint8 (byte-addressed).
// BF16 mode: head_dim * 2 bytes per entry (1024 KV / 256 Indexer).
// FP8 mode: 584B per KV entry, 132B per Indexer entry — see constants above.
// Each entry contains the full KV for one compressed token (one KV head).
struct DSV4KVSpec: public KVCacheSpec {
    KVCacheRegionName cache_type = KVCacheRegionName::DEFAULT;
    uint32_t          entry_elems;        // bytes per entry (1024/584 KV, 256/132 Indexer)
    uint32_t          entries_per_block;  // entries per block (64 or 2)
    DataType          store_dtype;        // TYPE_UINT8

    DSV4KVSpec() = default;

    DSV4KVSpec(KVCacheRegionName cache_region,
               uint32_t          layer_count,
               uint32_t          entry_elements,
               uint32_t          block_entries,
               DataType          storage_dtype,
               uint32_t          seq_size_per_blk) {
        cache_type        = cache_region;
        entry_elems       = entry_elements;
        entries_per_block = block_entries;
        store_dtype       = storage_dtype;

        // KVCacheSpec base fields
        layer_num          = layer_count;
        local_head_num_kv  = 1;  // DSV4 is MQA (1 KV head)
        seq_size_per_block = seq_size_per_blk;
        type               = KVCacheSpecType::MultiHeadAttention;
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

    size_t block_size_bytes() const override {
        return static_cast<size_t>(entries_per_block) * entry_elems * getTypeSize(store_dtype);
    }

    // FlashMLA SM100 sparse_attn requires per-block byte alignment for
    // FP8 KV pools. BF16 KV pools (1024B / 256B per entry) are already
    // naturally aligned so this collapses to block_size_bytes().
    size_t padded_block_size_bytes() const {
        const size_t natural = block_size_bytes();
        if (entry_elems == DSV4_FP8_KV_ENTRY_BYTES) {
            constexpr size_t align = DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES;
            return ((natural + align - 1) / align) * align;
        }
        return natural;
    }

    size_t k_block_size_bytes() const override {
        return block_size_bytes() / 2;
    }
    size_t v_block_size_bytes() const override {
        return block_size_bytes() / 2;
    }

    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "DSV4KVSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent + 2, ' ') << "cache_type=" << static_cast<int>(cache_type) << "\n";
        os << std::string(indent + 2, ' ') << "entry_elems=" << entry_elems << "\n";
        os << std::string(indent + 2, ' ') << "entries_per_block=" << entries_per_block << "\n";
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

// KVCacheSpec for DSV4 fixed-allocation state pools (Pool 3/4/5: INDEXER_STATE, CSA_STATE, HCA_STATE).
// These are per-request fixed-size allocations storing compressor/indexer state as float32.
// They do NOT participate in prefix cache (CacheGroupType::FIXED).
// The K/V split is a placeholder — state is an opaque blob with no K/V semantic.
struct DSV4StateSpec: public KVCacheSpec {
    KVCacheRegionName cache_type = KVCacheRegionName::DEFAULT;
    uint32_t          state_dim;             // state dimension (entry_elems in pool_spec)
    uint32_t          entries_per_block;     // 4 or 8
    uint32_t          fixed_blocks_per_req;  // blocks per request (2 or 16)
    DataType          store_dtype;           // TYPE_FP32

    DSV4StateSpec() = default;

    DSV4StateSpec(KVCacheRegionName cache_region,
                  uint32_t          layer_count,
                  uint32_t          state_elements,
                  uint32_t          block_entries,
                  uint32_t          fixed_blocks,
                  DataType          storage_dtype,
                  uint32_t          seq_size_per_blk) {
        cache_type           = cache_region;
        state_dim            = state_elements;
        entries_per_block    = block_entries;
        fixed_blocks_per_req = fixed_blocks;
        store_dtype          = storage_dtype;

        // KVCacheSpec base fields
        layer_num          = layer_count;
        local_head_num_kv  = 1;
        seq_size_per_block = seq_size_per_blk;
        type               = KVCacheSpecType::MultiHeadAttention;
        dtype              = store_dtype;
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

    size_t block_size_bytes() const override {
        return static_cast<size_t>(entries_per_block) * state_dim * getTypeSize(store_dtype);
    }

    // SWA_KV uses DSV4StateSpec for its fixed/ring allocation strategy but
    // stores the same FP8 ``fp8_model1_mla`` 584B layout as the paged FP8 KV
    // pools. FlashMLA SM100 sparse-decode/prefill kernels read SWA blocks via
    // TMA and assert ``k_cache.stride(0) % TMA_K_STRIDE == 0`` (576B), so
    // pad SWA_KV per-block bytes the same way DSV4KVSpec does. Other state
    // pools (INDEXER_STATE / CSA_STATE / HCA_STATE) keep state_dim != 584B
    // per slot — they fall through to natural block_size_bytes().
    size_t padded_block_size_bytes() const {
        const size_t natural = block_size_bytes();
        if (state_dim == DSV4_FP8_KV_ENTRY_BYTES) {
            constexpr size_t align = DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES;
            return ((natural + align - 1) / align) * align;
        }
        return natural;
    }

    size_t k_block_size_bytes() const override {
        return block_size_bytes() / 2;
    }
    size_t v_block_size_bytes() const override {
        return block_size_bytes() / 2;
    }

    std::string debugString(size_t indent = 0) const override {
        std::ostringstream os;
        os << std::string(indent, ' ') << "DSV4StateSpec{\n";
        os << commonDebugString(indent);
        os << std::string(indent + 2, ' ') << "cache_type=" << static_cast<int>(cache_type) << "\n";
        os << std::string(indent + 2, ' ') << "state_dim=" << state_dim << "\n";
        os << std::string(indent + 2, ' ') << "entries_per_block=" << entries_per_block << "\n";
        os << std::string(indent + 2, ' ') << "fixed_blocks_per_req=" << fixed_blocks_per_req << "\n";
        os << std::string(indent, ' ') << "}\n";
        return os.str();
    }
};

}  // namespace rtp_llm
