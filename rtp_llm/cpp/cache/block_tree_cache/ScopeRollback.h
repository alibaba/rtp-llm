#pragma once

#include <utility>

namespace rtp_llm {
namespace block_tree_cache_detail {

template<typename Cleanup>
class ScopeRollback {
public:
    explicit ScopeRollback(Cleanup cleanup): cleanup_(std::move(cleanup)) {}

    ~ScopeRollback() {
        run();
    }

    ScopeRollback(const ScopeRollback&)            = delete;
    ScopeRollback& operator=(const ScopeRollback&) = delete;
    ScopeRollback(ScopeRollback&&)                 = delete;
    ScopeRollback& operator=(ScopeRollback&&)      = delete;

    void run() {
        if (!active_) {
            return;
        }
        active_ = false;
        cleanup_();
    }

    void dismiss() noexcept {
        active_ = false;
    }

private:
    Cleanup cleanup_;
    bool    active_{true};
};

}  // namespace block_tree_cache_detail
}  // namespace rtp_llm
