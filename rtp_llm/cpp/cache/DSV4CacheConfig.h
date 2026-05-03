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
    // Logical dtype of the entry payload - independent of `store_dtype` (which
    // is always TYPE_UINT8 for KV pools because `entry_elems` is in bytes).
    // Set to TYPE_FP8_E4M3 when the model runs with kv_cache_dtype=FP8 so
    // downstream kernels can detect the FP8 layout from the spec without
    // re-reading the model config. Defaults to TYPE_BF16 (BF16 KV path).
    // Always TYPE_FP32 for state pools.
    DataType logical_dtype = DataType::TYPE_BF16;

    size_t block_size_bytes() const {
        return static_cast<size_t>(entries_per_block) * entry_elems * getTypeSize(store_dtype);
    }

    // FlashMLA TMA path (MODEL1, V4-Flash) requires the per-block byte stride
    // for the FP8 KV pools (CSA_KV / HCA_KV / SWA_KV, entry_elems == 584) to
    // be a multiple of TMA_K_STRIDE = D_NOPE + 2 * D_ROPE = 448 + 128 = 576;
    // the SM100 sparse decode/prefill kernel asserts on
    // `k_cache.stride(0) % TMA_K_STRIDE == 0`. Other pools (state pools,
    // INDEXER_KV, BF16 KV) don't go through that path and stay at natural
    // size.
    size_t padded_block_size_bytes() const;

    // Total bytes per layer per block (with TMA padding when applicable).
    size_t layer_block_bytes() const {
        return padded_block_size_bytes();
    }

    // Total bytes for all layers per block (with TMA padding when applicable).
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

    // KV entry layout for SWA_KV / CSA_KV / HCA_KV pools. All three store
    // a head_dim=512 K vector per entry, so they share the same per-entry
    // byte size. The actual size depends on attn_config.kv_cache_dtype:
    //   BF16: head_dim=512, bf16 -> 512 * 2 = 1024 bytes.
    //   FP8 : 448 fp8_e4m3 NoPE + 64 bf16 RoPE (=128 bytes) + 7 UE8M0 scale
    //         bytes + 1 pad = 584 bytes. Matches the ``fp8_model1_mla``
    //         layout produced by mla_quant_kernel.cu MODEL1 and consumed
    //         by FlashMLA's ``flash_mla_with_kvcache(is_fp8_kvcache=True)``.
    static constexpr uint32_t KV_ENTRY_BYTES_BF16 = 1024;
    static constexpr uint32_t KV_ENTRY_BYTES_FP8  = 584;

    // TMA_K_STRIDE for the FlashMLA MODEL1 (V4-Flash) FP8 path. The per-block
    // byte stride of the FP8 KV pools must be a multiple of this value.
    static constexpr uint32_t FP8_MLA_BLOCK_ALIGNMENT_BYTES = 576;

    // Indexer KV entry layout for INDEXER_KV pool - index_head_dim=128 per token.
    //   BF16: 128 * 2 = 256 bytes.
    //   FP8 : 128 fp8_e4m3 + 1 int32-packed UE8M0 scale (= 4 bytes) = 132 bytes.
    //         Matches sgl_per_token_group_quant_fp8(group=128, ue8m0=True) +
    //         the packing used by deep_gemm.fp8_paged_mqa_logits' kv_cache.
    static constexpr uint32_t INDEXER_ENTRY_BYTES_BF16 = 256;
    static constexpr uint32_t INDEXER_ENTRY_BYTES_FP8  = 132;

    // Back-compat aliases (= BF16 sizes). Prefer the helpers below in new
    // code so the call site explicitly reflects the FP8 vs BF16 choice.
    static constexpr uint32_t KV_ENTRY_BYTES      = KV_ENTRY_BYTES_BF16;
    static constexpr uint32_t INDEXER_ENTRY_BYTES = INDEXER_ENTRY_BYTES_BF16;

    static constexpr uint32_t kvEntryBytes(bool fp8) {
        return fp8 ? KV_ENTRY_BYTES_FP8 : KV_ENTRY_BYTES_BF16;
    }
    static constexpr uint32_t indexerEntryBytes(bool fp8) {
        return fp8 ? INDEXER_ENTRY_BYTES_FP8 : INDEXER_ENTRY_BYTES_BF16;
    }

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

inline size_t DSV4PoolSpec::padded_block_size_bytes() const {
    const size_t natural = block_size_bytes();
    // Only the FP8 MLA KV pools (entry_elems == 584) feed FlashMLA's TMA path
    // and need 576-byte stride alignment. Everything else (BF16 KV, INDEXER_KV
    // FP8 132B, FP32 state pools) is left at natural size.
    if (entry_elems == DSV4CacheConfig::KV_ENTRY_BYTES_FP8) {
        constexpr size_t align = DSV4CacheConfig::FP8_MLA_BLOCK_ALIGNMENT_BYTES;
        return ((natural + align - 1) / align) * align;
    }
    return natural;
}

}  // namespace rtp_llm
