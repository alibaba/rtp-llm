#pragma once

#include "rtp_llm/cpp/cache/KVCacheSpecBase.h"
#include "rtp_llm/cpp/cache/DSV4CacheConfig.h"

namespace rtp_llm {

// KVCacheSpec for DSV4 paged KV pools (Pool 0/1/2/6: CSA_KV, HCA_KV, INDEXER_KV, SWA_KV).
// These are variable-length paged pools storing KV entries as uint8 (byte-addressed).
// Current model (BF16-only): each entry is head_dim * 2 bytes (bf16).
// Future mixed-precision: FP8 NoPE + BF16 RoPE + UE8M0 scales (smaller entries).
// Each entry contains the full KV for one compressed token (one KV head).
struct DSV4KVSpec: public KVCacheSpec {
    DSV4CacheType cache_type;
    uint32_t      entry_elems;        // bytes per entry (584 for KV, 132 for Indexer)
    uint32_t      entries_per_block;  // entries per block (64 or 2)
    DataType      store_dtype;        // TYPE_UINT8

    DSV4KVSpec() = default;

    DSV4KVSpec(const DSV4PoolSpec& pool_spec, uint32_t seq_size_per_blk) {
        cache_type        = pool_spec.type;
        entry_elems       = pool_spec.entry_elems;
        entries_per_block = pool_spec.entries_per_block;
        store_dtype       = pool_spec.store_dtype;

        // KVCacheSpec base fields
        layer_num          = pool_spec.layer_num;
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
    DSV4CacheType cache_type;
    uint32_t      state_dim;             // state dimension (entry_elems in pool_spec)
    uint32_t      entries_per_block;     // 4 or 8
    uint32_t      fixed_blocks_per_req;  // blocks per request (2 or 16)
    DataType      store_dtype;           // TYPE_FP32

    DSV4StateSpec() = default;

    DSV4StateSpec(const DSV4PoolSpec& pool_spec, uint32_t seq_size_per_blk) {
        cache_type           = pool_spec.type;
        state_dim            = pool_spec.entry_elems;
        entries_per_block    = pool_spec.entries_per_block;
        fixed_blocks_per_req = pool_spec.fixed_blocks_per_req;
        store_dtype          = pool_spec.store_dtype;

        // KVCacheSpec base fields
        layer_num          = pool_spec.layer_num;
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
