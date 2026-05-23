#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

// Encode a list of bools (slot_count length) into a packed bytes string of
// length ceil(slot_count / 8). Bit i within byte j corresponds to slot
// (j * 8 + i). Bit 1 = valid (has gpu block), bit 0 = NULL slot.
inline std::string packBitmap(const std::vector<bool>& bits) {
    const size_t n     = bits.size();
    const size_t bytes = (n + 7) / 8;
    std::string  out(bytes, '\0');
    for (size_t i = 0; i < n; ++i) {
        if (bits[i]) {
            out[i / 8] = static_cast<char>(static_cast<unsigned char>(out[i / 8]) | (1u << (i % 8)));
        }
    }
    return out;
}

// Decode a packed bitmap back into bools of `slot_count` length. Returns empty
// vector if the input length is inconsistent with slot_count.
inline std::vector<bool> unpackBitmap(const std::string& packed, size_t slot_count) {
    const size_t expected = (slot_count + 7) / 8;
    if (packed.size() != expected) {
        return {};
    }
    std::vector<bool> out(slot_count, false);
    for (size_t i = 0; i < slot_count; ++i) {
        if ((static_cast<unsigned char>(packed[i / 8]) >> (i % 8)) & 1u) {
            out[i] = true;
        }
    }
    return out;
}

// Per-(master_id, peer_id) op_sequence FIFO checker. Workers maintain one
// instance per master and enforce strict monotonic increase. Out-of-order /
// duplicate / skipped messages are rejected with protocol_violation.
//
// Master gets a single instance and uses next() to bump.
//
// First call: any sequence accepted (initial sync). Subsequent calls accepted
// only when seq == last_accepted + 1.
class OpSequenceTracker {
public:
    enum class CheckResult : uint8_t {
        OK              = 0,
        DUPLICATE       = 1,
        OUT_OF_ORDER    = 2,
        SKIPPED         = 3,
    };

    // Caller-facing: register a received seq. Returns OK if accepted (and
    // last_accepted is bumped). Otherwise the violation type.
    CheckResult checkReceived(uint64_t seq) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialized_) {
            initialized_   = true;
            last_accepted_ = seq;
            return CheckResult::OK;
        }
        if (seq == last_accepted_) {
            return CheckResult::DUPLICATE;
        }
        if (seq < last_accepted_) {
            return CheckResult::OUT_OF_ORDER;
        }
        if (seq > last_accepted_ + 1) {
            return CheckResult::SKIPPED;
        }
        last_accepted_ = seq;
        return CheckResult::OK;
    }

    // Master-facing: next outgoing seq.
    uint64_t next() {
        return next_outgoing_.fetch_add(1) + 1;
    }

    uint64_t lastAccepted() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return last_accepted_;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        initialized_   = false;
        last_accepted_ = 0;
        next_outgoing_.store(0);
    }

private:
    mutable std::mutex    mutex_;
    bool                  initialized_{false};
    uint64_t              last_accepted_{0};
    std::atomic<uint64_t> next_outgoing_{0};
};

}  // namespace rtp_llm
