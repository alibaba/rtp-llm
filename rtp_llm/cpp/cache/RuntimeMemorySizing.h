#pragma once

#include <cstddef>
#include <cstdint>

namespace rtp_llm {

// Pure sizing math, intentionally dependency-free (no logger, no device access).
//
// safety_ratio and no_warmup_floor_bytes deliberately do NOT default to the product
// values (those live in KVCacheConfig, ConfigModules.h): the initializers below are
// neutral -- 0.0 adds no headroom, 0 floors nothing -- so a caller that forgets one
// under-reserves silently. MemoryEvaluationHelper assigns every field by name for that
// reason.
//
// Scope: this is a SINGLE-RANK budget derived from that rank's own measurement. Per-rank peaks
// can differ; cluster agreement comes from KVCacheManager::allocateAndSync reducing block_num
// with std::min across the world.
//
// Behaviour change: the warmup path does NOT apply no_warmup_floor_bytes (a measurement
// replaces the guesswork floors), so a small model on a small GPU can reserve less than it
// did before this feature. See RuntimeMemorySizingResult::warmup_below_no_warmup_floor.
//
// safety_ratio intentionally has two compatibility-preserving meanings: additive headroom on the
// trusted warmup path, but a max() floor on the no-warmup path. A mixed-role cluster therefore
// cannot tune those meanings independently with this knob. Use configured_reserve_bytes
// (--reserver_runtime_mem_mb at the service entrypoint) when an individual role needs additional
// absolute reserve without changing the shared ratio.
struct RuntimeMemorySizingInput {
    bool   has_warmup               = false;
    size_t configured_reserve_bytes = 0;
    size_t warmup_required_bytes    = 0;
    size_t sampler_required_bytes   = 0;
    size_t total_gpu_bytes          = 0;
    double safety_ratio             = 0.0;
    size_t no_warmup_floor_bytes    = 0;
};

struct RuntimeMemorySizingResult {
    // total_gpu_bytes * safety_ratio. Added on top of the base requirement when a
    // warmup measurement exists; otherwise it is one more floor inside the max()
    // (the pre-warmup-feature "at least 5% of total GPU" minimum).
    size_t safety_ratio_bytes     = 0;
    size_t runtime_required_bytes = 0;
    // True when the warmup path produced a reserve below no_warmup_floor_bytes, i.e.
    // below what the same deployment would have reserved before this feature. Reported
    // as a flag rather than logged here because this layer stays dependency-free (no
    // logger); MemoryEvaluationHelper turns it into a WARNING naming
    // --reserver_runtime_mem_mb. Always false on the no-warmup path, where the floor is
    // one of the max() terms and cannot be undercut.
    bool warmup_below_no_warmup_floor = false;
};

RuntimeMemorySizingResult calculateRuntimeMemorySizing(const RuntimeMemorySizingInput& input);

}  // namespace rtp_llm
