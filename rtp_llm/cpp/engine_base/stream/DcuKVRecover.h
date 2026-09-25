#pragma once

#include <cstdlib>
#include <string>

namespace rtp_llm {

// DCU-only KV allocation recovery: pause-wait + deadlock eviction + requeue recompute.
//
// Background (2026-09-19/20, see 断流问题.md): on the DCU (galaxyhip) backend,
// terminating a running stream mid-decode on incremental KV malloc failure
// leaves GPU-side state that can hang the engine main loop forever
// (hipMemcpyWithStream waiting on hsaKmtWaitOnEvent_Ext for an event that
// never fires). Upstream keeps the "fail the sacrificed request" semantics on
// other backends; on DCU we instead:
//   1. pause the stream on retryable incremental KV shortage (C2a),
//   2. when every running stream is paused (zero progress), evict the newest
//      running stream (C2b) — requeue it for full recompute instead of
//      terminating it (C2c), capped per stream against starvation.
//
// Compile-time gated by USING_DCU (set by --config=dcu); runtime kill switch:
// RTP_DCU_KV_RECOVER=0 restores the upstream terminate-on-failure behavior.
inline bool dcuKVRecoverEnabled() {
#if USING_DCU
    static const bool enabled = [] {
        const char* env = std::getenv("RTP_DCU_KV_RECOVER");
        return env == nullptr || std::string(env) != "0";
    }();
    return enabled;
#else
    return false;
#endif
}

// Max times a single stream may be evicted + requeued before it is finished
// via the normal GenerateDone path (anti-starvation cap for C2c).
constexpr int kMaxKvRequeuePerStream = 3;

}  // namespace rtp_llm
