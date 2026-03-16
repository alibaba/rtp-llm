#pragma once

#include <ATen/record_function.h>

#include <memory>
#include <string>
#include <unordered_map>

namespace rtp_llm {

using ProfileArgs = std::unordered_map<std::string, c10::IValue>;

class ProfilingScope {
public:
    explicit ProfilingScope(const char* name): record_function_(at::RecordScope::FUNCTION) {
        if (record_function_.isActive()) {
            record_function_.before(name);
            active_ = true;
        }
    }

    ProfilingScope(const char* name, const ProfileArgs& kwargs): record_function_(at::RecordScope::FUNCTION) {
        if (record_function_.isActive()) {
            record_function_.before(name, &kwargs);
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

// Profile a named scope with key-value args visible in the trace.
// Usage:
//   rtp_llm::ProfileArgs args{{"bytes", (int64_t)4096}, {"block_id", (int64_t)42}};
//   RTP_LLM_PROFILE_SCOPE_WITH_ARGS("cache.read", args);
#define RTP_LLM_PROFILE_SCOPE_WITH_ARGS(scope_name_literal, args_map)                                                  \
    ::rtp_llm::ProfilingScope RTP_LLM_PROFILE_CONCAT(rtp_llm_ps_, __LINE__)(scope_name_literal, args_map)

// Profile the enclosing function — automatically uses __PRETTY_FUNCTION__
#define RTP_LLM_PROFILE_FUNCTION() ::rtp_llm::ProfilingScope rtp_llm_pf_(__PRETTY_FUNCTION__)
