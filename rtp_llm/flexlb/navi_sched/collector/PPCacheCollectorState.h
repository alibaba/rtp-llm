#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace navi_sched {

struct PPRequestAttempt {
    uint64_t requestId = 0;
    uint64_t attemptId = 0;
    bool operator==(const PPRequestAttempt &other) const
    {
        return requestId == other.requestId && attemptId == other.attemptId;
    }
};

struct PPRequestAttemptHash {
    size_t operator()(const PPRequestAttempt &value) const
    {
        return std::hash<uint64_t>()(value.requestId) ^
               (std::hash<uint64_t>()(value.attemptId) << 1);
    }
};

enum class PPShardRequestStatus : uint8_t {
    PENDING = 0,
    SUCCEEDED = 1,
    FAILED = 2,
    CANCELLED = 3,
    DISCONNECTED = 4,
};

struct PPShardRequestTerminal {
    PPRequestAttempt request;
    PPShardRequestStatus status = PPShardRequestStatus::FAILED;
};

// Cache changes describe observed KV availability, independently of whether
// the request succeeded. A failed request can leave older valid KV intact.
struct PPCacheShardUpdate {
    uint32_t shardId = 0;
    uint64_t streamGeneration = 0;
    uint64_t sequence = 0;
    bool fullSnapshot = false;
    // -1 leaves readiness unchanged; 0/1 report the native peer read state.
    int8_t peerCacheReadAvailable = -1;
    std::vector<uint64_t> addedFull;
    std::vector<uint64_t> removedFull;
    std::vector<uint64_t> addedTail;
    std::vector<uint64_t> removedTail;
    std::vector<PPShardRequestTerminal> terminals;
};

struct PPRequestShardConnection {
    uint32_t shardId = 0;
    uint64_t generation = 0;
};

struct PPRequestCompletion {
    PPRequestAttempt request;
    bool succeeded = false;
    std::vector<PPShardRequestStatus> shardStatuses;
};

struct PPCacheCollectorChanges {
    int8_t peerCacheReadAvailable = -1;
    bool accepted = false;
    bool needsFullSnapshot = false;
    std::vector<uint64_t> addedFull;
    std::vector<uint64_t> removedFull;
    std::vector<uint64_t> addedTail;
    std::vector<uint64_t> removedTail;
    std::vector<PPRequestCompletion> completedRequests;
};

class PPCacheCollectorState {
public:
    explicit PPCacheCollectorState(uint32_t shardCount);

    // Generations are assigned by the collector's stream manager. Replacing
    // a connection clears that shard's cache and completes its old waits.
    PPCacheCollectorChanges connect(uint32_t shardId, uint64_t generation);
    PPCacheCollectorChanges disconnect(uint32_t shardId, uint64_t generation);
    PPCacheCollectorChanges apply(const PPCacheShardUpdate &update);
    PPCacheCollectorChanges beginRequest(const PPRequestAttempt &request);
    std::vector<PPRequestShardConnection> pendingRequestShards(
        const PPRequestAttempt &request) const;
    size_t pendingRequestCount() const;
    bool fullReady(uint64_t hash) const;
    bool tailReady(uint64_t hash) const;

private:
    using HashSet = std::unordered_set<uint64_t>;
    using Counts = std::unordered_map<uint64_t, uint32_t>;
    struct Shard {
        uint64_t generation = 0;
        uint64_t sequence = 0;
        bool connected = false;
        bool synced = false;
        bool peerCacheReadAvailable = false;
        HashSet full;
        HashSet tail;
    };
    struct Request {
        std::vector<uint64_t> generations;
        std::vector<PPShardRequestStatus> statuses;
        uint32_t remaining = 0;
    };

    void checkShard(uint32_t shardId) const;
    void remove(uint64_t hash, HashSet &local, Counts &counts,
                std::vector<uint64_t> &removed);
    void add(uint64_t hash, HashSet &local, Counts &counts, std::vector<uint64_t> &added);
    void clearCache(Shard &shard, PPCacheCollectorChanges &changes);
    void setPeerCacheReadAvailable(Shard &shard, bool available,
                                   PPCacheCollectorChanges &changes);
    void failShardRequests(uint32_t shardId, uint64_t generation,
                           PPCacheCollectorChanges &changes);
    void completeRequestIfReady(const PPRequestAttempt &key,
                                PPCacheCollectorChanges &changes);

    const uint32_t _shardCount;
    mutable std::mutex _mutex;
    std::vector<Shard> _shards;
    uint32_t _peerReadableShards = 0;
    Counts _fullCounts;
    Counts _tailCounts;
    std::unordered_map<PPRequestAttempt, Request, PPRequestAttemptHash> _requests;
};

}
