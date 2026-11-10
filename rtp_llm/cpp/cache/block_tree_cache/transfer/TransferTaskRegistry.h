#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

// Local completion evidence for worker executions. Entries, including safe ones, survive until destruction:
// deleting an entry would otherwise allow a delayed duplicate to execute again.
// Ordinary transfer reuses its numeric trace ID; KVCM can reuse its existing string
// identity in a separate registry instance. No per-task identity is generated here.
// The worker group shares one lifetime; records are not reused across instances.
class TransferTaskRegistry {
public:
    enum class State { UNKNOWN, RUNNING, BUFFER_SAFE };

    TransferTaskRegistry(): epoch_(makeEpoch()) {}

    const std::string& epoch() const noexcept {
        return epoch_;
    }

    // Includes both pending and completed executions; records are never evicted.
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return records_.size();
    }

    // The caller must obtain admission before starting any buffer access.
    bool begin(const std::string& trace_id) {
        if (trace_id.empty()) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        return records_.emplace(trace_id, State::RUNNING).second;
    }

    bool begin(uint64_t trace_id) {
        return trace_id != 0 && begin(std::to_string(trace_id));
    }

    // buffer_safe is an explicit I/O completion proof, not business success.
    // False preserves the last known state. A true proof is monotonic and can
    // describe a safely drained failure. The return value only reports ID presence.
    bool finish(const std::string& trace_id, bool buffer_safe) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto                  it = records_.find(trace_id);
        if (it == records_.end()) {
            return false;
        }
        if (buffer_safe) {
            it->second = State::BUFFER_SAFE;
        }
        return true;
    }

    bool finish(uint64_t trace_id, bool buffer_safe) {
        return trace_id != 0 && finish(std::to_string(trace_id), buffer_safe);
    }

    // The worker group has one shared lifetime. Read-only queries of this
    // instance preserve request order and never create missing records.
    std::vector<State> queryLocal(const std::vector<std::string>& trace_ids) const {
        std::vector<State> states(trace_ids.size(), State::UNKNOWN);
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t i = 0; i < trace_ids.size(); ++i) {
            const auto it = records_.find(trace_ids[i]);
            if (it != records_.end()) {
                states[i] = it->second;
            }
        }
        return states;
    }

    std::vector<State> queryLocal(const std::vector<uint64_t>& trace_ids) const {
        std::vector<std::string> keys;
        keys.reserve(trace_ids.size());
        for (const auto trace_id : trace_ids) {
            // Numeric zero is never an identity, even if a string caller has
            // registered the literal "0". Empty string records cannot exist.
            keys.push_back(trace_id == 0 ? std::string{} : std::to_string(trace_id));
        }
        return queryLocal(keys);
    }

    // Retain the explicit epoch check for callers that already hold one.
    std::vector<State> query(const std::string& expected_epoch, const std::vector<std::string>& trace_ids) const {
        return expected_epoch == epoch_ ? queryLocal(trace_ids) : std::vector<State>(trace_ids.size(), State::UNKNOWN);
    }

    std::vector<State> query(const std::string& expected_epoch, const std::vector<uint64_t>& trace_ids) const {
        return expected_epoch == epoch_ ? queryLocal(trace_ids) : std::vector<State>(trace_ids.size(), State::UNKNOWN);
    }

private:
    static std::string makeEpoch() {
        std::random_device random;
        std::string        epoch = "transfer-registry-";
        for (int i = 0; i < 4; ++i) {
            epoch += std::to_string(random()) + "-";
        }
        return epoch;
    }

    const std::string                     epoch_;
    mutable std::mutex                    mutex_;
    std::unordered_map<std::string, State> records_;
};

}  // namespace rtp_llm
