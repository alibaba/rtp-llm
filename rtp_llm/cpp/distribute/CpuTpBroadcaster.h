#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace rtp_llm {

// CPU-only UDS broadcaster for intra-node TP metadata sync.
// Callers must handle unsupported topologies before using this interface.
// Python initializes this during distributed bootstrap, while the C++ engine
// thread performs request-time broadcasts. Calls may cross threads, but
// broadcastCPU is still serialized: no concurrent or re-entrant broadcasts, and
// reset must not race with an in-flight broadcast.
class CpuTpBroadcaster {
public:
    static CpuTpBroadcaster& instance();

    // Bootstrap UDS endpoints. Call once per process. Subsequent calls with
    // matching (tp_rank, tp_size, base_path) are no-ops; mismatched re-init
    // throws.
    void initialize(int tp_rank, int tp_size, const std::string& base_path);

    // Blocking broadcast of `nbytes` bytes from `buf` originating at root.
    // root rank writes to all peer sockets; non-root reads from its peer-0 fd.
    // Throws on transport error.
    void broadcast(void* buf, std::size_t nbytes, int root);

    // Close all sockets and return the singleton to its initial state so a
    // later distributed init can bootstrap a fresh base_path.
    void reset();

    bool isInitialized() const {
        return initialized_.load(std::memory_order_acquire);
    }

private:
    CpuTpBroadcaster() = default;
    ~CpuTpBroadcaster();
    CpuTpBroadcaster(const CpuTpBroadcaster&)            = delete;
    CpuTpBroadcaster& operator=(const CpuTpBroadcaster&) = delete;

    void cleanupStateLocked();

    std::mutex        mu_;
    std::atomic<bool> initialized_{false};
    bool              broadcast_in_progress_ = false;
    int               tp_rank_               = 0;
    int               tp_size_               = 1;
    std::string       base_path_;
    int               listen_fd_ = -1;  // rank 0 only
    // peer_fds_[k] = fd connecting this rank to rank k. peer_fds_[tp_rank_] = -1.
    std::vector<int> peer_fds_;
    std::string      my_uds_path_;  // path to unlink at shutdown (rank 0 only)
};

}  // namespace rtp_llm
