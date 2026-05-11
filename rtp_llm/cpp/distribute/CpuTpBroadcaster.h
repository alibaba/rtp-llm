#pragma once

#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <vector>

namespace rtp_llm {

// Lightweight CPU-only broadcaster used by tpSyncModelInputs to avoid the
// NCCL-induced cudaDeviceSynchronize stall (see m2.md / mtp_stream_async).
//
// Topology: star, root = rank 0.
//   - rank 0 binds + listens on a Unix Domain Socket and accepts (tp_size-1)
//     peers.
//   - rank K > 0 connects to rank 0's socket.
//
// Only intra-node TP groups should call initialize(); cross-node TP must keep
// using the NCCL path (callers detect via local_world_size). When not
// initialized, callers fall back to the original NCCL broadcast.
//
// Thread safety: initialize() is synchronized. broadcast() is NOT — the design
// assumes only the engine main thread (the same thread that runs
// tpSyncModelInputs) ever calls it. If multiple threads need to broadcast,
// add external synchronization or call out per-mode broadcasters.
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

    std::mutex        mu_;
    std::atomic<bool> initialized_{false};
    int               tp_rank_ = 0;
    int               tp_size_ = 1;
    std::string       base_path_;
    int               listen_fd_ = -1;  // rank 0 only
    // peer_fds_[k] = fd connecting this rank to rank k. peer_fds_[tp_rank_] = -1.
    std::vector<int> peer_fds_;
    std::string      my_uds_path_;  // path to unlink at shutdown (rank 0 only)
};

}  // namespace rtp_llm
