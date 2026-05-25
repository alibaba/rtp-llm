#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"

#include <algorithm>
#include <cstdint>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/HashUtil.h"

namespace rtp_llm {

namespace {

// XOR-fold a 64-bit value down to a single byte. Ensures every bit of
// the input contributes to the output byte so high-word-only inputs
// (e.g. model_id = 1<<32) cannot silently land on byte 0 == 0.
inline uint64_t fold8(uint64_t v) {
    v ^= v >> 32;
    v ^= v >> 16;
    v ^= v >> 8;
    return v & 0xffull;
}

// Assemble the v2 salt layout: each field occupies its own byte window
// of the resulting uint64. See KVCacheHashUtil.h for the on-wire diagram.
inline uint64_t packSaltBytes(const CacheKeySalt& s) {
    uint64_t bytes = 0;
    bytes |= fold8(s.model_id)                                                << (0 * 8);
    bytes |= (static_cast<uint64_t>(s.dtype_id) & 0xffull)                    << (1 * 8);
    bytes |= (static_cast<uint64_t>(s.lora_id)  & 0xffull)                    << (2 * 8);
    bytes |= (static_cast<uint64_t>(s.K_state)  & 0xffull)                    << (3 * 8);
    bytes |= (static_cast<uint64_t>(s.Tlog)     & 0xffull)                    << (4 * 8);
    // bytes 5..7 reserved — must stay zero so a future field bump is
    // detectable by callers that inspect the raw assembled salt bytes.
    //
    // FIX-B HIGH-4 (DEFEND-4 #3): invariant CHECK that reserved bytes 5..7
    // really are 0 before passing to Jenkins-64.  Catches a future field
    // that writes ``<<(5*8)`` (or higher) without bumping
    // ``kCacheKeySaltSchemaVersion`` — v2 peers would otherwise produce
    // byte-divergent cache_keys yet matching handshake bitmap/version,
    // which the PD handshake REQ-D1 cannot detect.  Use
    // RTP_LLM_CHECK_WITH_INFO (release-active) rather than bare DCHECK
    // so the schema-drift hazard fires in prod, not only in debug builds.
    RTP_LLM_CHECK_WITH_INFO((bytes & 0xffffff0000000000ULL) == 0ULL,
                            "CacheKeySalt reserved bytes 5..7 are non-zero "
                            "(packed=0x%lx); bump kCacheKeySaltSchemaVersion before adding a field "
                            "in that byte window",
                            static_cast<unsigned long>(bytes));
    return bytes;
}

// Apply salt fields on top of the unsalted Jenkins roll (M03 §3.1).
// A default-initialised (all-zero) salt produces packSaltBytes() == 0
// → XOR with zero is identity → today's legacy hash bytes are preserved
// exactly for K_state=0 / bitmap=0.
inline int64_t applySalt(int64_t h, const CacheKeySalt& s) {
    return static_cast<int64_t>(static_cast<uint64_t>(h) ^ packSaltBytes(s));
}

// Rolling hash with salt: Jenkins-64 over the token chunk, then XOR-fold
// the salt fields. This is the single source of truth for cache_key bytes.
inline int64_t rollHash(int64_t prev, int32_t* begin, int32_t* end, const CacheKeySalt& s) {
    int64_t h = rtp_llm::hashInt64Array(prev, begin, end);
    return applySalt(h, s);
}

}  // namespace

uint32_t nonzeroFieldBitmap(const CacheKeySalt& salt) {
    uint32_t bitmap = 0;
    if (salt.model_id != 0) bitmap |= 1u << 0;
    if (salt.dtype_id != 0) bitmap |= 1u << 1;
    if (salt.lora_id  != 0) bitmap |= 1u << 2;
    if (salt.K_state  != 0) bitmap |= 1u << 3;
    if (salt.Tlog     != 0) bitmap |= 1u << 4;
    return bitmap;
}

CacheKeySalt makeCacheKeySalt(const CacheConfig& cache_config) {
    // F01-PR2 part A: only the K_state bit (bit3) is populated by this
    // producer.  K_state == 0 → all-zero salt → byte-identical legacy
    // cache_keys + handshake (version 0, bitmap 0).  K_state > 0 (set when
    // ``--dsv4_state_entries_per_block`` flips on under
    // DSV4CacheConfigHelper::applyConfig) → salt.K_state mirrors the
    // override so PD peers with different K_state values produce
    // distinct cache_keys (reuse-miss instead of silent corruption,
    // Risk 9.6) AND advertise a non-zero (version, bitmap) over the
    // handshake so validatePeerHandshake can refuse mixed-mode pairs
    // (REQ-D1).  Other CacheKeySalt fields (model_id, dtype_id, lora_id,
    // Tlog) are reserved for follow-up PRs and stay zero here.
    CacheKeySalt salt{};
    const int k_state = cache_config.state_entries_per_block_constant;
    // FIX-B HIGH-1 (DEFEND-4 #1): bounds CHECK at salt producer.  DEV-2
    // normalizes K_state=256 to OFF (state_entries_per_block_constant=0
    // in that branch), so the producer must never observe a K_state > 128.
    // Any value > 128 here means an upstream regression bypassed the
    // DSV4CacheConfigHelper::applyConfig normalisation; a value > 255
    // would silently alias byte 3 of the salt (256 & 0xff == 0 vs
    // bitmap bit3 = 1 → silent reuse-miss across DSV4 streams).
    // RTP_LLM_CHECK_WITH_INFO (release-active) so the misconfig fires in
    // prod, not only in debug builds (Risk 9.6 silent reuse-miss path).
    RTP_LLM_CHECK_WITH_INFO(k_state >= 0 && k_state <= 128,
                            "makeCacheKeySalt: state_entries_per_block_constant=%d outside [0,128]; "
                            "DSV4CacheConfigHelper must normalize 256 to 0 (kernel identity) and "
                            "reject larger values before the salt producer runs",
                            k_state);
    if (k_state > 0) {
        salt.K_state = static_cast<uint32_t>(k_state);
    }
    return salt;
}

void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block,
                   const CacheKeySalt&     salt) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    // Initial fill: compute cache_keys for all blocks, including the final partial block.
    const int desired_blocks = (seq_len + seq_size_per_block - 1) / seq_size_per_block;  // ceil

    for (int i = 0; i < batch_size; ++i) {
        batch_kv_cache_resource->clearCacheKeys(i);

        int64_t rolling_hash = 0;
        auto*   token_ids    = complete_token_ids->data(i);
        for (int index = 0; index < desired_blocks; ++index) {
            const int pos       = index * seq_size_per_block;
            const int block_len = std::min(seq_size_per_block, seq_len - pos);
            rolling_hash        = rollHash(rolling_hash, token_ids + pos, token_ids + pos + block_len, salt);
            batch_kv_cache_resource->pushBackCacheKey(i, rolling_hash);
        }
    }

    batch_kv_cache_resource->setLastBlockAligned(seq_len % seq_size_per_block == 0);
}

void initCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                   CompleteTokenIdsPtr     complete_token_ids,
                   int                     seq_size_per_block) {
    // Legacy unsalted entry: zero-initialised salt -> XOR with 0 -> identity.
    static const CacheKeySalt kZeroSalt{};
    initCacheKeys(batch_kv_cache_resource, complete_token_ids, seq_size_per_block, kZeroSalt);
}

void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block,
                     const CacheKeySalt&     salt) {
    const int batch_size = batch_kv_cache_resource->batchSize();
    const int seq_len    = complete_token_ids->seqLength();

    for (int i = 0; i < batch_size; ++i) {
        const auto& keys         = batch_kv_cache_resource->cacheKeys(i);
        const int   total_blocks = seq_len / seq_size_per_block;  // floor, only full blocks

        // If last_block_aligned was false previously, the last cache key corresponds to a partial block.
        // Drop it before we append new full-block cache keys.
        if (!batch_kv_cache_resource->lastBlockAligned() && !keys.empty()) {
            batch_kv_cache_resource->popBackCacheKey(i);
        }

        auto*   token_ids = complete_token_ids->data(i);
        int64_t hash      = keys.empty() ? 0 : keys.back();
        int     start_idx = static_cast<int>(keys.size());

        for (int index = start_idx; index < total_blocks; ++index) {
            const int pos = index * seq_size_per_block;
            hash          = rollHash(hash, token_ids + pos, token_ids + pos + (int)seq_size_per_block, salt);
            batch_kv_cache_resource->pushBackCacheKey(i, hash);
        }
    }

    // After incremental update we guarantee all existing keys are for full blocks.
    batch_kv_cache_resource->setLastBlockAligned(true);
}

void updateCacheKeys(BatchKVCacheResourcePtr batch_kv_cache_resource,
                     CompleteTokenIdsPtr     complete_token_ids,
                     int                     seq_size_per_block) {
    // Legacy unsalted entry: zero-initialised salt -> XOR with 0 -> identity.
    static const CacheKeySalt kZeroSalt{};
    updateCacheKeys(batch_kv_cache_resource, complete_token_ids, seq_size_per_block, kZeroSalt);
}

void dropLastPartialBlock(BatchKVCacheResourcePtr batch_kv_cache_resource) {
    if (batch_kv_cache_resource->lastBlockAligned()) {
        return;
    }
    batch_kv_cache_resource->popBackAllBatchCacheKeys();
    batch_kv_cache_resource->setLastBlockAligned(true);
}

}  // namespace rtp_llm
