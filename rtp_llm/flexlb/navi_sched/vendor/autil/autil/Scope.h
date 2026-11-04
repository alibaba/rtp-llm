#pragma once

#include <functional>
#include <utility>

namespace autil {

class ScopeGuard {
public:
    template <typename F>
    ScopeGuard(F &&f) : _exitFn(std::forward<F>(f)) {}

    ~ScopeGuard() {
        if (_exitFn) {
            _exitFn();
        }
    }

    void release() { _exitFn = std::function<void()>(); }

private:
    std::function<void()> _exitFn;
};

} // namespace autil
