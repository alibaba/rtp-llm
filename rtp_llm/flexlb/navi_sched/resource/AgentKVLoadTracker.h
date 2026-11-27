#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace navi_sched {
// Agent-local lifecycle only. EngineCore owns connector completion aggregation;
// this adapter neither enumerates workers nor reconstructs PP/TP/PCP completion.
class AgentKVLoadTracker {
public:
    struct Key {
        uint64_t agentSession = 0;
        uint64_t requestId = 0;
        uint64_t generation = 0;
        bool operator==(const Key &other) const {
            return agentSession == other.agentSession && requestId == other.requestId &&
                   generation == other.generation;
        }
        bool valid() const { return agentSession && requestId && generation; }
    };
    enum class State { PREPARING, READY, FAILED, CANCELED };
    enum class Update { INVALID, STALE, DUPLICATE, READY, FAILED, CONFLICT };
    struct Snapshot {
        Key key;
        State state = State::PREPARING;
        int64_t registeredNs = 0;
        // Agent-local receipt of EngineCore's unified completion, not a worker
        // DMA timestamp. Never subtract monotonic clocks across hosts.
        int64_t readyObservedNs = 0;
        double preparationElapsedMs(int64_t nowAgentMonoNs) const {
            if (registeredNs <= 0) return -1.0;
            const auto end = state == State::READY ? readyObservedNs :
                (state == State::PREPARING ? nowAgentMonoNs : int64_t(0));
            return end >= registeredNs ? (end - registeredNs) / 1e6 : -1.0;
        }
    };
    struct LatencyWindowStats {
        uint64_t readyCount = 0;
        uint64_t failedCount = 0;
        long double totalPreparationNs = 0;
        int64_t maxPreparationNs = 0;
    };
    struct Result {
        Update update = Update::INVALID;
        std::optional<Snapshot> snapshot;
        bool terminalChanged = false;
    };
    explicit AgentKVLoadTracker(uint64_t session) : _session(session) {}

    std::optional<Snapshot> begin(Key key, int64_t registeredNs) {
        if (!key.valid() || registeredNs <= 0) return std::nullopt;
        std::lock_guard<std::mutex> lock(_mutex);
        if (key.agentSession != _session) return std::nullopt;
        auto at = _loads.find(key.requestId);
        if (at != _loads.end()) {
            if (at->second.key == key) return at->second;
            // Superseding a pending Native load requires explicit cancellation.
            if (at->second.state == State::PREPARING ||
                at->second.key.generation >= key.generation) return std::nullopt;
        }
        if (key.generation <= _lastRegisteredGeneration) return std::nullopt;
        Snapshot load;
        load.key = key; load.registeredNs = registeredNs;
        _loads.insert_or_assign(key.requestId, load);
        _lastRegisteredGeneration = key.generation;
        return load;
    }

    Result complete(Key key, bool success, int64_t observedNs) {
        if (!key.valid() || observedNs <= 0) return {};
        std::lock_guard<std::mutex> lock(_mutex);
        auto at = _loads.find(key.requestId);
        if (key.agentSession != _session || at == _loads.end() ||
            !(at->second.key == key)) return {Update::STALE, {}, false};
        auto &load = at->second;
        if (load.state == State::CANCELED) return {Update::STALE, load, false};
        if (observedNs < load.registeredNs) return {Update::INVALID, load, false};
        const auto desired = success ? State::READY : State::FAILED;
        if (load.state == desired) return {Update::DUPLICATE, load, false};
        if (load.state != State::PREPARING) {
            // Contradictory completion cannot recover an already failed load.
            const bool changed = load.state != State::FAILED;
            load.state = State::FAILED; load.readyObservedNs = 0;
            if (changed) ++_latencyWindow.failedCount;
            return {Update::CONFLICT, load, changed};
        }
        load.state = desired;
        if (!success) {
            ++_latencyWindow.failedCount;
            return {Update::FAILED, load, true};
        }
        load.readyObservedNs = observedNs;
        const auto elapsed = observedNs - load.registeredNs;
        ++_latencyWindow.readyCount;
        _latencyWindow.totalPreparationNs += elapsed;
        _latencyWindow.maxPreparationNs = std::max(_latencyWindow.maxPreparationNs, elapsed);
        return {Update::READY, load, true};
    }
    std::optional<Snapshot> snapshot(Key key) const {
        std::lock_guard<std::mutex> lock(_mutex);
        auto at = _loads.find(key.requestId);
        return at != _loads.end() && at->second.key == key
                   ? std::optional<Snapshot>(at->second) : std::nullopt;
    }
    bool cancel(Key key) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto at = _loads.find(key.requestId);
        if (at == _loads.end() || !(at->second.key == key)) return false;
        at->second.state = State::CANCELED; at->second.readyObservedNs = 0;
        return true;
    }
    bool retire(Key key) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto at = _loads.find(key.requestId);
        if (at == _loads.end() || !(at->second.key == key)) return false;
        _loads.erase(at); return true;
    }
    std::vector<std::optional<Snapshot>> snapshots(const std::vector<Key> &keys) const {
        std::vector<std::optional<Snapshot>> result; result.reserve(keys.size());
        std::lock_guard<std::mutex> lock(_mutex);
        for (const auto &key : keys) {
            auto at = _loads.find(key.requestId);
            if (at != _loads.end() && at->second.key == key) result.emplace_back(at->second);
            else result.emplace_back(std::nullopt);
        }
        return result;
    }
    LatencyWindowStats takeLatencyWindowStats() {
        std::lock_guard<std::mutex> lock(_mutex);
        auto result = _latencyWindow; _latencyWindow = {}; return result;
    }
    size_t activeCount() const {
        std::lock_guard<std::mutex> lock(_mutex); return _loads.size();
    }
private:
    const uint64_t _session;
    uint64_t _lastRegisteredGeneration = 0;
    LatencyWindowStats _latencyWindow;
    mutable std::mutex _mutex;
    std::unordered_map<uint64_t, Snapshot> _loads;
};
} // namespace navi_sched
