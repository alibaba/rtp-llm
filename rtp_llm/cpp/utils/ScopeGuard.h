#pragma once

#include <utility>

namespace rtp_llm {

template<typename F>
class ScopeGuard {
    F f;
    ScopeGuard()                  = delete;
    ScopeGuard(const ScopeGuard&) = delete;
    ScopeGuard(ScopeGuard&&)      = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ScopeGuard& operator=(ScopeGuard&&) = delete;

public:
    ScopeGuard(F&& f): f(std::move(f)) {}
    ScopeGuard(const F& f): f(f) {}
    ~ScopeGuard() {
        f();
    }
};

#define RTP_LLM_SCOPE_GUARD_NAME_IMPL(line) _rtp_llm_scope_guard__##line
#define RTP_LLM_SCOPE_GUARD_NAME(line) RTP_LLM_SCOPE_GUARD_NAME_IMPL(line)
#define RTP_LLM_SCOPE_GUARD(...) ::rtp_llm::ScopeGuard RTP_LLM_SCOPE_GUARD_NAME(__LINE__)(__VA_ARGS__);

}  // namespace rtp_llm
