#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace rtp_llm {

struct CpuBroadcastSharedState;

// CPU-only UDS broadcaster for intra-node TP metadata sync.
// Callers must handle unsupported topologies before using this interface.
// Python initializes this during distributed bootstrap, while the C++ engine
// thread performs request-time broadcasts. Calls may cross threads, but
// broadcastCPU is still serialized: no concurrent or re-entrant broadcasts, and
// reset must not race with an in-flight broadcast. Any transport failure makes
// the instance unusable until reset() rebuilds the stream state.
class CpuBroadcaster {
public:
    static CpuBroadcaster& instance();

    // Bootstrap UDS endpoints. Call once per process. Subsequent calls with
    // matching (rank, world_size, base_path) are no-ops; mismatched re-init
    // throws.
    void initialize(int rank, int world_size, const std::string& base_path);

    // Blocking broadcast of `nbytes` bytes from `buf` originating at root.
    // After every peer acknowledges the payload, root atomically publishes one
    // shared commit decision; a pre-commit failure publishes a shared abort.
    // Throws on transport error, frame-size mismatch, or shared abort.
    void broadcast(void* buf, std::size_t nbytes, int root);

    // Close all sockets and return the singleton to its initial state so a
    // later distributed init can bootstrap a fresh base_path.
    void reset();

    bool isInitialized() const {
        return initialized_.load(std::memory_order_acquire);
    }

private:
    CpuBroadcaster() = default;
    ~CpuBroadcaster();
    CpuBroadcaster(const CpuBroadcaster&) = delete;
    CpuBroadcaster& operator=(const CpuBroadcaster&) = delete;

    void cleanupStateLocked();
    void markBroadcastFailedLocked();

    std::mutex        mu_;
    std::atomic<bool> initialized_{false};
    bool              broadcast_in_progress_ = false;
    bool              failed_                = false;
    int               rank_                  = 0;
    int               world_size_            = 1;
    int               broadcast_timeout_ms_  = 0;
    uint32_t          broadcast_generation_  = 0;
    std::string       base_path_;
    int               listen_fd_ = -1;  // rank 0 only
    // peer_fds_[k] = fd connecting this rank to rank k. peer_fds_[rank_] = -1.
    std::vector<int>         peer_fds_;
    std::string              my_uds_path_;  // path to unlink at shutdown (rank 0 only)
    CpuBroadcastSharedState* shared_state_ = nullptr;
    std::string              shared_state_path_;  // unlinked after all ranks map it
};

}  // namespace rtp_llm
