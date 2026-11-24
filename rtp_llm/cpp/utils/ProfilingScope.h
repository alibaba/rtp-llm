#pragma once

#include <ATen/record_function.h>

#include <cstdio>
#include <cstdint>
#include <string>

#if USING_CUDA
#include <cuda.h>
#if CUDA_VERSION >= 12090
#include <nvtx3/nvToolsExt.h>
#else
#include <nvToolsExt.h>
#endif
#endif

namespace rtp_llm {

class PressureNvtxScope {
public:
    static bool enabled() {
#if USING_CUDA
        return session_enabled_;
#else
        return false;
#endif
    }

    // Applied only on the engine thread at explicit session boundaries.
    static void setSessionEnabled(bool enabled) {
        session_enabled_ = enabled;
    }

    explicit PressureNvtxScope(const char* name, int64_t rank = -1, bool active = true): active_(active && enabled()) {
#if USING_CUDA
        if (active_) {
            nvtxEventAttributes_t attributes{};
            attributes.version       = NVTX_VERSION;
            attributes.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attributes.messageType   = NVTX_MESSAGE_TYPE_ASCII;
            attributes.message.ascii = name;
            if (rank >= 0) {
                attributes.payloadType      = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
                attributes.payload.ullValue = static_cast<uint64_t>(rank);
            }
            nvtxRangePushEx(&attributes);
        }
#endif
    }
    ~PressureNvtxScope() {
#if USING_CUDA
        if (active_) {
            nvtxRangePop();
        }
#endif
    }
    PressureNvtxScope(const PressureNvtxScope&)            = delete;
    PressureNvtxScope& operator=(const PressureNvtxScope&) = delete;

private:
    inline static thread_local bool session_enabled_{false};
    bool                            active_;
};

class ProfilingScope {
public:
    explicit ProfilingScope(const char* name, bool emit_nvtx = true):
        nvtx_scope_(name, -1, emit_nvtx), record_function_(at::RecordScope::FUNCTION) {
        if (record_function_.isActive()) {
            record_function_.before(name);
            active_ = true;
        }
    }

    ~ProfilingScope() {
        if (active_) {
            record_function_.end();
        }
    }

    ProfilingScope(const ProfilingScope&)            = delete;
    ProfilingScope& operator=(const ProfilingScope&) = delete;

private:
    PressureNvtxScope  nvtx_scope_;
    at::RecordFunction record_function_;
    bool               active_ = false;
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
    if (at::hasCallbacks() || ::rtp_llm::PressureNvtxScope::enabled()) {                                               \
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
#define RTP_LLM_PROFILE_FUNCTION() ::rtp_llm::ProfilingScope rtp_llm_pf_(__PRETTY_FUNCTION__)
