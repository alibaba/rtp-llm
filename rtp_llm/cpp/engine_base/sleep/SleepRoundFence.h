#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>

namespace rtp_llm {

// One producer: the engine loop. Control threads may freeze/setTarget/wait/resume.
// Tickets count admitted executor calls, NOT completed calls or generated tokens.
// Freeze and admission share a single atomic word: no call can be admitted after
// the snapshot without an explicit catch-up allowance. No collective or GIL here.
class SleepRoundFence {
public:
    enum class Action {
        RUN,
        QUIESCE,
        STOP
    };
    struct Permit {
        Action   action;
        uint64_t generation;
        uint64_t round;
    };

    Permit next() {
        auto word = admitted_.load(std::memory_order_relaxed);
        while (!(word & kFrozen) && (word & kRoundMask) < kRoundMask) {
            if (admitted_.compare_exchange_weak(word, word + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) {
                return {Action::RUN, 0, word + 1};
            }
        }
        return nextSlow();
    }

    // Does not wait for the engine, GPU, or Python. Idempotent within a freeze.
    uint64_t freeze();
    bool     setTarget(uint64_t round);
    bool     wait(std::chrono::milliseconds timeout);
    // Called by the engine thread after draining ALL async runners and streams.
    void finishQuiesce(uint64_t generation, const std::string& error = "");
    void fail(const std::string& error);
    // Refuse a false rollback if a device drain is still stuck or has failed.
    bool resume();
    void stop();

private:
    Permit nextSlow();

    static constexpr uint64_t kFrozen    = uint64_t{1} << 63;
    static constexpr uint64_t kRoundMask = kFrozen - 1;
    std::atomic<uint64_t>     admitted_{0};
    std::mutex                mutex_;
    std::condition_variable   cv_;
    uint64_t                  generation_{0};
    uint64_t                  frozen_round_{0};
    std::optional<uint64_t>   target_;
    bool                      quiescing_{false};
    bool                      quiesced_{false};
    bool                      stopped_{false};
    std::string               failure_;
};

}  // namespace rtp_llm
