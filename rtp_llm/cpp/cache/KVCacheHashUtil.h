#pragma once

#include <cstdint>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

// ---------------------------------------------------------------------------
// Hash salt (M03 §3.1 / §4 — Hash Salt Strategy)
//
// PR-M03-1: introduce a per-rolling-hash salt that disambiguates configs
// whose unsalted token-only Jenkins hash would otherwise collide on the
// shared block-cache LRU. The salt is the "PD-payload-compatibility class"
// (F04 25-7 / 39-6): two configs that produce byte-equivalent PD payloads
// share the same salt; any incompatibility produces distinct salts.
//
// This PR is behaviour-preserving:
//   - existing call sites pass no salt and go through the wrapper that
//     supplies a default-initialised (all-zero) CacheKeySalt;
//   - XOR with all-zero is identity, so the resulting cache_key bytes are
//     bit-identical to the legacy unsalted Jenkins output.
//
// Production opt-in (per-field) is gated by F07 step 4; PD peers exchange
// (schema_version, nonzero_field_bitmap) over the M04 §4.3 handshake to
// fail-fast on mismatched salt schemas (Panel D §1.3, REQ-D1).
// ---------------------------------------------------------------------------

struct CacheKeySalt {
    uint64_t model_id = 0;  // bit0 — ModelConfig::model_id
    uint32_t dtype_id = 0;  // bit1 — kv_cache_dtype (FP8/BF16/...)
    uint32_t lora_id  = 0;  // bit2 — lora_config.id (0 = no LoRA)
    uint32_t K_state  = 0;  // bit3 — F01 phase-2 K_state semantic
    uint32_t Tlog     = 0;  // bit4 — super_block_layout step
};

// Salt schema version (M03 §4 REQ-D1). Bumped whenever the set or
// interpretation of CacheKeySalt fields changes. PR-M03-1 ships v1 with
// every field defaulting to zero -> legacy-identical hashes.
constexpr uint32_t kCacheKeySaltSchemaVersion = 1u;

// Per-field non-zero bitmap (M03 §4 REQ-D1). bit0=model_id, bit1=dtype_id,
// bit2=lora_id, bit3=K_state, bit4=Tlog. PD handshake carries
// (schema_version, bitmap) so peers detect mismatched salt schemas.
uint32_t nonzeroFieldBitmap(const CacheKeySalt& salt);

// Initial fill: build cache_keys for all blocks (including the final partial block).
// Also updates BatchKVCacheResource::last_block_aligned based on seq_len % seq_size_per_block.
//
// Salted overload: XORs CacheKeySalt fields into every rolling hash result.
void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block,
                   const CacheKeySalt&     salt);

// Legacy unsalted entry: thin wrapper that passes a zero-initialised salt,
// producing bit-identical hashes to today.
void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block);

// Subsequent fill: rebuild cache_keys only for fully-aligned blocks (ignores the tail partial block).
// Also updates BatchKVCacheResource::last_block_aligned based on current seq_len.
//
// Salted overload: XORs CacheKeySalt fields into every rolling hash result.
void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block,
                     const CacheKeySalt&     salt);

// Legacy unsalted entry: thin wrapper that passes a zero-initialised salt.
void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block);

// Drop the last block in cache_keys
void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource);

}  // namespace rtp_llm
