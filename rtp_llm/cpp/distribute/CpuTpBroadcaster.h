#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

namespace rtp_llm {

// Only a fully validated, in-sequence terminal frame produces this exception.
// Transport EOF and malformed frames remain errors, including during shutdown.
class CpuTpShutdown final: public std::exception {
public:
    const char* what() const noexcept override {
        return "CpuTpBroadcaster received an ordered shutdown frame";
    }
};

// CPU-only framed UDS broadcaster for intra-node TP input sync.
// It avoids NCCL cudaDeviceSynchronize for small CPU tensors; cross-node or
// uninitialized callers fall back to NCCL. Each frame carries its sequence and
// payload size so a rank-local input-layout mismatch fails immediately instead
// of consuming part of the next frame and hanging the TP group.
class CpuTpBroadcaster {
public:
    static CpuTpBroadcaster& instance();

    // Bootstrap UDS endpoints. Call once per process. Subsequent calls with
    // matching (tp_rank, tp_size, base_path) are no-ops; mismatched re-init
    // throws.
    void initialize(int tp_rank, int tp_size, const std::string& base_path);

    // Blocking broadcast of one framed payload from `buf` originating at root.
    // Calls are serialized per process. Throws on transport or frame-contract
    // mismatch before reading a differently sized payload.
    void broadcast(void* buf, std::size_t nbytes, int root);

    // Claim the one engine loop allowed to consume this process-global session.
    // Warmup executors do not claim it. A new engine requires reset/reinitialize.
    void claimEngineSession(int tp_rank, int tp_size);

    // Rank 0 sends exactly one terminal frame after its last complete engine step.
    // Followers observe CpuTpShutdown from their next broadcast. No payload is sent.
    // This does not close sockets or acknowledge GPU/executor destruction.
    void sendShutdown();

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
    std::mutex        broadcast_mu_;
    std::atomic<bool> initialized_{false};
    int               tp_rank_ = 0;
    int               tp_size_ = 1;
    std::string       base_path_;
    int               listen_fd_ = -1;  // rank 0 only
    // peer_fds_[k] = fd connecting this rank to rank k. peer_fds_[tp_rank_] = -1.
    std::vector<int>  peer_fds_;
    std::string       my_uds_path_;  // path to unlink at shutdown (rank 0 only)
    uint64_t          next_sequence_ = 0;
    std::atomic<bool> shutdown_{false};
    bool              engine_session_claimed_ = false;  // protected by mu_
};

}  // namespace rtp_llm
