#pragma once

#include "rtp_llm/models_py/bindings/core/Types.h"
#include <array>
#include <cstdint>
#include <vector>

namespace rtp_llm {

enum class DSV4CacheType : int {
    CSA_KV        = 0,  // Pool 0: paged, compressed KV for CSA layers (ratio=4)
    HCA_KV        = 1,  // Pool 1: paged, compressed KV for HCA layers (ratio=128)
    INDEXER_KV    = 2,  // Pool 2: paged, indexer KV for CSA layers (ratio=4)
    INDEXER_STATE = 3,  // Pool 3: fixed, indexer state per request
    CSA_STATE     = 4,  // Pool 4: fixed, CSA compressor state per request
    HCA_STATE     = 5,  // Pool 5: fixed, HCA compressor state per request
    SWA_KV        = 6,  // Pool 6: paged, sliding window KV for all layers
};

constexpr int DSV4_NUM_POOLS = 7;

struct DSV4PoolSpec {
    DSV4CacheType type;
    uint32_t      layer_num;             // number of layers this pool covers
    uint32_t      entry_elems;           // elements per entry
    uint32_t      entries_per_block;     // entries per block
    DataType      store_dtype;           // TYPE_UINT8 (KV) or TYPE_FP32 (State)
    bool          is_paged;              // true=variable-length paged, false=fixed per-request
    uint32_t      fixed_blocks_per_req;  // blocks per request when is_paged=false

    size_t block_size_bytes() const {
        return static_cast<size_t>(entries_per_block) * entry_elems * getTypeSize(store_dtype);
    }

    // Total bytes per layer per block
    size_t layer_block_bytes() const {
        return block_size_bytes();
    }

    // Total bytes for all layers per block
    size_t total_block_bytes() const {
        return layer_block_bytes() * layer_num;
    }
};

struct DSV4CacheConfig {
    std::array<DSV4PoolSpec, DSV4_NUM_POOLS> pool_specs;

    // Layer classification
    std::vector<int> csa_layer_ids;       // layers with compress_ratio=4
    std::vector<int> hca_layer_ids;       // layers with compress_ratio=128
    std::vector<int> swa_only_layer_ids;  // layers with compress_ratio=0 (Flash only)
    std::vector<int> all_layer_ids;       // all transformer layers (for SWA pool)

    // Block counts
    uint32_t variable_num_blocks = 0;  // Pool 0/1/2 shared block count

    // All groups use the same tokens_per_block = 256
    static constexpr uint32_t TOKENS_PER_BLOCK = 256;
    static constexpr uint32_t SLIDING_WINDOW   = 128;

    // KV entry layout — must match the model's actual storage format.
    // Current model (BF16-only path): head_dim=512, bf16 → 512 * 2 = 1024 bytes.
    // Future mixed-precision target: 448 FP8 NoPE + 128 BF16 RoPE + 7 UE8M0 scale + 1 pad = 584 bytes.
    // Update this when the model switches to FP8/BF16 mixed KV storage.
    static constexpr uint32_t KV_ENTRY_BYTES = 1024;
    // Indexer entry — must match the model's actual storage format.
    // Current model (BF16-only path): index_head_dim=128, bf16 → 128 * 2 = 256 bytes.
    // Future mixed-precision target: 128 FP8 + 4 FP32 scale = 132 bytes.
    // Update this when the model switches to FP8 indexer KV storage.
    static constexpr uint32_t INDEXER_ENTRY_BYTES = 256;

    uint32_t num_csa_layers() const {
        return csa_layer_ids.size();
    }
    uint32_t num_hca_layers() const {
        return hca_layer_ids.size();
    }
    uint32_t num_swa_only_layers() const {
        return swa_only_layer_ids.size();
    }
    uint32_t num_all_layers() const {
        return all_layer_ids.size();
    }
};

}  // namespace rtp_llm
