#pragma once

#include <ATen/record_function.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#if USING_CUDA && __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define RTP_LLM_HAS_NVTX_PROFILING 1
#else
#define RTP_LLM_HAS_NVTX_PROFILING 0
#endif

namespace rtp_llm {

class ProfilingScope {
public:
    // Named scopes can also be collected by Nsight, including on output workers
    // where thread-affine Torch profiler callbacks are intentionally disabled.
    explicit ProfilingScope(const char* name, bool named_scope = true): record_function_(at::RecordScope::FUNCTION) {
#if RTP_LLM_HAS_NVTX_PROFILING
        if (named_scope && nvtxEnabled()) {
            nvtxRangePushA(name);
            nvtx_active_ = true;
        }
#endif
        if (record_function_.isActive()) {
            record_function_.before(name);
            active_ = true;
        }
    }

    ~ProfilingScope() {
        if (active_) {
            record_function_.end();
        }
#if RTP_LLM_HAS_NVTX_PROFILING
        if (nvtx_active_) {
            nvtxRangePop();
        }
#endif
    }

    static bool nvtxEnabled() {
#if RTP_LLM_HAS_NVTX_PROFILING
        static const bool enabled = [] {
            const char* value = std::getenv("RTP_LLM_PROFILE_NVTX");
            return value != nullptr && value[0] == '1' && value[1] == '\0';
        }();
        return enabled;
#else
        return false;
#endif
    }

    ProfilingScope(const ProfilingScope&)            = delete;
    ProfilingScope& operator=(const ProfilingScope&) = delete;

private:
    at::RecordFunction record_function_;
    bool               active_ = false;
    bool               nvtx_active_ = false;
};

}  // namespace rtp_llm

// Token-pasting helpers for unique variable names
#define RTP_LLM_PROFILE_CONCAT_(a, b) a##b
#define RTP_LLM_PROFILE_CONCAT(a, b) RTP_LLM_PROFILE_CONCAT_(a, b)

// Profile a named scope (e.g. "rpc.enqueue_engine")
#define RTP_LLM_PROFILE_SCOPE(scope_name_literal)                                                                      \
    ::rtp_llm::ProfilingScope RTP_LLM_PROFILE_CONCAT(rtp_llm_ps_, __LINE__)(scope_name_literal)

// Profile a scope with a dynamic name (printf-style formatting).
// The snprintf is only performed when the profiler is active — zero cost when OFF.
// Usage:
//   RTP_LLM_PROFILE_SCOPE_DYNAMIC("engine.schedule(reserve_step=%d)", reserve_step);
#define RTP_LLM_PROFILE_SCOPE_DYNAMIC(fmt, ...)                                                                        \
    char RTP_LLM_PROFILE_CONCAT(rtp_llm_buf_, __LINE__)[256];                                                          \
    if (at::hasCallbacks() || ::rtp_llm::ProfilingScope::nvtxEnabled()) {                                               \
        snprintf(RTP_LLM_PROFILE_CONCAT(rtp_llm_buf_, __LINE__),                                                       \
                 sizeof(RTP_LLM_PROFILE_CONCAT(rtp_llm_buf_, __LINE__)),                                               \
                 fmt,                                                                                                  \
                 __VA_ARGS__);                                                                                         \
    } else {                                                                                                           \
        RTP_LLM_PROFILE_CONCAT(rtp_llm_buf_, __LINE__)[0] = '\0';                                                      \
    }                                                                                                                  \
    ::rtp_llm::ProfilingScope RTP_LLM_PROFILE_CONCAT(rtp_llm_ps_,                                                      \
                                                     __LINE__)(RTP_LLM_PROFILE_CONCAT(rtp_llm_buf_, __LINE__))

// Profile the enclosing function — automatically uses __PRETTY_FUNCTION__
// Keep fine-grained function scopes out of NVTX to avoid tracing every block
// allocator operation. Use named scopes for the diagnostic phases of interest.
#define RTP_LLM_PROFILE_FUNCTION() ::rtp_llm::ProfilingScope rtp_llm_pf_(__PRETTY_FUNCTION__, false)
