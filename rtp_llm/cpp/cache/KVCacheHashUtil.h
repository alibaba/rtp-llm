#pragma once

#include <cstdint>

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"

namespace rtp_llm {

struct CacheConfig;  // forward decl to avoid pulling CacheConfig.h into the hash-util header


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
//
// F01-PR2-followup (v2 layout): salt fields are packed into NON-OVERLAPPING
// byte windows of a uint64 before the XOR-fold, removing the v1 hazard
// where ``model_id`` (uint64) covered every other field's 32-bit XOR slot
// (R4-4 F7 trivial collision: ``{Tlog=1} ≡ {model_id=1<<32}``).
//
// On-wire layout of the assembled uint64 salt (little-endian byte index):
//     byte 0 : fold8(model_id)        — XOR-fold of all 8 bytes of model_id
//     byte 1 : dtype_id & 0xff        — kv_cache_dtype id (FP8/BF16/INT8...)
//     byte 2 : lora_id  & 0xff        — lora bucket (mod 256)
//     byte 3 : K_state  & 0xff        — F01 phase-2 K_state (<=256)
//     byte 4 : Tlog     & 0xff        — F02 super-block layout step
//     bytes 5..7 : reserved (must be 0)
//
// Pairwise non-overlap proof: each non-zero field writes to its own byte
// window; two distinct field combinations cannot land on identical salt
// bytes unless both are zero. fold8(model_id) ensures any non-zero bit
// of the 64-bit model_id surfaces in byte 0 (the v1 layout silently
// aliased model_id high words because uint64 << 0 dropped the high
// bytes off the dtype/lora/K_state/Tlog XOR slots).
// ---------------------------------------------------------------------------

struct CacheKeySalt {
    uint64_t model_id = 0;  // bit0 — ModelConfig::model_id (fold8 -> byte0)
    uint32_t dtype_id = 0;  // bit1 — kv_cache_dtype (FP8/BF16/...) -> byte1
    uint32_t lora_id  = 0;  // bit2 — lora_config.id (0 = no LoRA)  -> byte2
    uint32_t K_state  = 0;  // bit3 — F01 phase-2 K_state semantic  -> byte3
    uint32_t Tlog     = 0;  // bit4 — super_block_layout step       -> byte4
};

// Salt schema version (M03 §4 REQ-D1). Bumped whenever the set or
// interpretation of CacheKeySalt fields changes. v1 used overlapping
// XOR slots; v2 (F01-PR2-followup) uses the non-overlapping byte-window
// layout described above. PD peers exchange (version, bitmap) so a v1
// and v2 peer trip REQ-D1 fail-loud at handshake time.
constexpr uint32_t kCacheKeySaltSchemaVersion = 2u;

// Per-field non-zero bitmap (M03 §4 REQ-D1). bit0=model_id, bit1=dtype_id,
// bit2=lora_id, bit3=K_state, bit4=Tlog. PD handshake carries
// (schema_version, bitmap) so peers detect mismatched salt schemas.
uint32_t nonzeroFieldBitmap(const CacheKeySalt& salt);

// F01-PR2 producer: build the per-engine CacheKeySalt from a CacheConfig.
// Single source of truth for both (a) the PD-handshake (version, bitmap)
// emitted by KVCacheConnectorCoordinator::init and (b) the cache_key XOR
// applied by KVCacheManager via initCacheKeys/updateCacheKeys.
//
// Currently populates only bit3 (K_state) from
// ``CacheConfig::state_entries_per_block_constant`` (set by F01-PR1's
// DSV4CacheConfigHelper::applyConfig hook). Default config →
// state_entries_per_block_constant == 0 → all-zero salt → byte-identical
// to today's unsalted cache_key bytes (XOR with zero = identity).
//
// Future PRs MUST extend this helper as additional CacheKeySalt fields
// come online (model_id, dtype_id, lora_id, Tlog) — keeping the producer
// in one place avoids the silent-divergence hazard the PR2 round-2
// reviews flagged when the handshake metadata and the cache_key XOR
// drift out of sync.
CacheKeySalt makeCacheKeySalt(const CacheConfig& cache_config);

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
