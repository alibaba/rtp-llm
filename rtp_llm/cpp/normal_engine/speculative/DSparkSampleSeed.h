#pragma once

#include <cstdint>

namespace rtp_llm {
namespace speculative {

// Counter-based sampling seeds for DSpARK.
//
// A seeded request must produce the same token for the same absolute position
// no matter how the engine got there. A stateful generator cannot promise that:
// bounded verification decides per round how many draft positions to verify, so
// the same generator draw lands on a different absolute position whenever the
// budget (or the accepted prefix) changes. A counter seed removes the coupling
// by making the draw a pure function of (sequence seed, absolute position),
// which is the construction DeepInfer uses for its per-sequence seeds.
//
// Draft proposals and target acceptance share one sequence seed, so each domain
// mixes in its own namespace constant first. Without that, a draft draw at
// position p would reuse the target acceptance draw at the same position.
constexpr uint64_t kDSparkTargetSeedNamespace = 0;
// Matches DeepInfer's _MTP_SEED_OFFSET so the two engines document the same
// namespace split.
constexpr uint64_t kDSparkDraftSeedNamespace = 1000000007ULL;
// A residual token becomes part of the next round's history. Its draw must
// therefore never reuse a future acceptance draw at the same token position.
constexpr uint64_t kDSparkResidualSeedNamespace = 0x726573696475616cULL;

inline uint64_t dsparkSplitMix64(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// Absolute positions are non-negative token offsets from the start of the
// sequence, so a request keeps the same draw after preemption or re-batching.
inline uint64_t dsparkSeedAt(uint64_t sequence_seed, uint64_t seed_namespace, int64_t position) {
    return dsparkSplitMix64(sequence_seed + seed_namespace + static_cast<uint64_t>(position) * 0x9E3779B97F4A7C15ULL);
}

// Uniform in [0, 1) with 24 random bits, matching float mantissa precision so
// the value survives the FP32 uniform buffer without rounding surprises.
inline float dsparkUniformAt(uint64_t sequence_seed, uint64_t seed_namespace, int64_t position) {
    constexpr float kScale = 1.0f / 16777216.0f;
    return static_cast<float>(dsparkSeedAt(sequence_seed, seed_namespace, position) >> 40) * kScale;
}

// Bounded rejection reserves the last of gamma+1 uniforms for its one possible
// residual sample. Acceptance uses only slots [0, gamma); addressing those by
// absolute position preserves the existing acceptance stream across budgets.
// Each round advances the committed prefix, so its base position uniquely
// addresses the residual stream without coupling it to a future accept draw.
inline float
dsparkRejectionUniformAt(uint64_t sequence_seed, int64_t round_base_position, int64_t slot, int64_t proposal_width) {
    return slot == proposal_width ?
               dsparkUniformAt(sequence_seed, kDSparkResidualSeedNamespace, round_base_position) :
               dsparkUniformAt(sequence_seed, kDSparkTargetSeedNamespace, round_base_position + slot);
}

}  // namespace speculative
}  // namespace rtp_llm
