#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

namespace rtp_llm {

// M04 PR-3 — KV cache PD pair startup handshake.
//
// Two RTP-LLM instances paired across a prefill/decode wire must agree on the
// structural cache layout (M01 SuperBlockLayout, M02 pool descriptors, M03
// hash salt schema, M04 protocol magic) before exchanging any
// BlockBufferInfo.  Without the handshake the magic-byte negotiation alone
// cannot catch the "silent reuse-miss" failure (cache_key strings opaque to
// RDMA layer, §4.2) or the cross-version pool-shape drift (Risk 9.6).
//
// The struct below carries the four handshake fields enumerated in M04 §4.3:
//
//   * protocol_magic            — receiver capability set ({0,1}); 0 = legacy,
//                                  1 = unified-aware (bps=1 schema).  Mixed
//                                  pairs downgrade to the common subset.
//   * pool_descriptor_hash      — uint64 over the PINNED input set (REQ-D2);
//                                  see ``poolDescriptorHashInputsFor`` below.
//                                  Mismatch ⇒ abort.
//   * hash_salt_version         — packed (schema_version : uint16,
//                                  nonzero_bitmap : uint16) for the M03 hash
//                                  salt; REQ-D1.  Mismatch ⇒ abort.
//   * hash_salt_nonzero_bitmap  — sender's per-field salt-domain bitmap so the
//                                  receiver can detect "both zero" vs
//                                  "schema-zero on one side only".
//
// Default-constructed instance corresponds to the legacy peer (protocol_magic
// = 0, hash = 0, salt zeros).  Default-on legacy ↔ legacy pairs therefore
// hand-shake trivially.  When the engine is built without the unified path
// engaged (cache_config.super_block_layout.enabled == false) the handshake is
// computed but the validator no-ops, preserving the legacy-only behaviour.
struct HandshakeInfo {
    uint32_t protocol_magic{0};
    uint64_t pool_descriptor_hash{0};
    uint32_t hash_salt_version{0};
    uint32_t hash_salt_nonzero_bitmap{0};

    bool operator==(const HandshakeInfo& other) const {
        return protocol_magic == other.protocol_magic && pool_descriptor_hash == other.pool_descriptor_hash
               && hash_salt_version == other.hash_salt_version
               && hash_salt_nonzero_bitmap == other.hash_salt_nonzero_bitmap;
    }
    bool operator!=(const HandshakeInfo& other) const {
        return !(*this == other);
    }

    std::string toString() const;
};

// Pinned pool-descriptor-hash inputs (REQ-D2).  Fields here change only
// across protocol-magic bumps, so additive future PRs (e.g. PR-4 last_partial
// flip, PR-5 aligned_flags) can land WITHOUT forcing a fleet-wide lockstep
// deploy.  Explicitly EXCLUDED (NOT in the hash):
//   * last_partial             — per-pool, flips at runtime per request
//   * aligned_flags/has_scale  — per-block, on the wire
//   * future protocol_magic≥2 fields — bump cache_layout_version instead
struct PoolDescriptorHashInput {
    uint32_t pool_id;
    uint32_t group_type;             // CacheGroupType enum value
    uint32_t region_name;            // KVCacheRegionName enum value
    uint64_t group_block_size_bytes; // per-pool slab size
    uint32_t layer_count;            // CacheConfig::layer_all_num
    uint32_t tma_alignment;          // DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES (576 on SM100)
};

// Extract the PINNED hash inputs from a CacheConfig.  Order is by pool_id
// ascending so the hash is deterministic across boots.
std::vector<PoolDescriptorHashInput> poolDescriptorHashInputsFor(const CacheConfig& cache_config);

// Compute the FNV-1a 64-bit hash over the pinned input set.  Stable across
// process restarts; identical between sender and receiver when the underlying
// CacheConfig agrees on the pinned fields.
uint64_t hashPoolDescriptorInputs(const std::vector<PoolDescriptorHashInput>& inputs);

// Build the local HandshakeInfo for the given config.  When the unified path
// is not engaged (super_block_layout.enabled == false) the result has
// protocol_magic=0 and is byte-equivalent to a legacy peer's handshake (so
// legacy↔legacy pairs hand-shake trivially).
HandshakeInfo computeLocalHandshakeInfo(const CacheConfig& cache_config,
                                        uint32_t           hash_salt_version        = 0,
                                        uint32_t           hash_salt_nonzero_bitmap = 0);

// Validate a peer handshake against the local handshake.  Returns true if the
// pair may proceed (legacy↔legacy OR same-version unified pair); on any
// mismatch returns false and fills ``error_message`` with a diagnostic.
//
// Behaviour:
//   * legacy↔legacy (both magic=0)    — accept; salt/hash MUST be zero on both
//                                       sides (legacy never populates them)
//   * legacy↔unified (one side magic=0)
//                                     — REFUSE.  Mixed-mode silent corruption
//                                       (Risk 9.6) prevented at startup
//                                       rather than via opaque cache_key
//                                       string races.  Receiver-side magic-
//                                       byte negotiation handles the
//                                       *per-request* fall-back; this stage
//                                       enforces *config-level* invariants.
//   * unified↔unified                 — pool_descriptor_hash MUST match;
//                                       hash_salt_version MUST match;
//                                       hash_salt_nonzero_bitmap MUST match
//                                       (REQ-D1).  Any mismatch ⇒ REFUSE.
bool validateHandshake(const HandshakeInfo& local, const HandshakeInfo& peer, std::string* error_message);

}  // namespace rtp_llm
