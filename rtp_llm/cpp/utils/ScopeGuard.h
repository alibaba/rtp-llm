#pragma once

#include <utility>

namespace rtp_llm {

template<typename F>
class ScopeGuard {
    F f;
    ScopeGuard()                             = delete;
    ScopeGuard(const ScopeGuard&)            = delete;
    ScopeGuard(ScopeGuard&&)                 = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ScopeGuard& operator=(ScopeGuard&&)      = delete;

public:
    ScopeGuard(F&& f): f(std::move(f)) {}
    ScopeGuard(const F& f): f(f) {}
    ~ScopeGuard() {
        f();
    }
};

#define FT_SCOPE_GUARD_NAME_IMPL(line) _ft_scope_guard__##line
#define FT_SCOPE_GUARD_NAME(line) FT_SCOPE_GUARD_NAME_IMPL(line)
#define FT_SCOPE_GUARD(...) ::rtp_llm::ScopeGuard FT_SCOPE_GUARD_NAME(__LINE__)(__VA_ARGS__);

}  // namespace rtp_llm
