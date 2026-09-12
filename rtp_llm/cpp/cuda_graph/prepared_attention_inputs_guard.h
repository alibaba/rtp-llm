#pragma once

#include <atomic>

namespace rtp_llm {

// Treat attention-input preparation as a transaction. A new attempt first
// invalidates any previously published state; only a fully successful attempt
// may publish readiness for the following forward.
class PreparedAttentionInputsGuard {
public:
    explicit PreparedAttentionInputsGuard(std::atomic<bool>& prepared): prepared_(prepared) {
        prepared_.store(false, std::memory_order_release);
    }

    PreparedAttentionInputsGuard(const PreparedAttentionInputsGuard&)            = delete;
    PreparedAttentionInputsGuard& operator=(const PreparedAttentionInputsGuard&) = delete;

    ~PreparedAttentionInputsGuard() {
        if (!committed_) {
            prepared_.store(false, std::memory_order_release);
        }
    }

    void commit() noexcept {
        prepared_.store(true, std::memory_order_release);
        committed_ = true;
    }

private:
    std::atomic<bool>& prepared_;
    bool               committed_{false};
};

}  // namespace rtp_llm
