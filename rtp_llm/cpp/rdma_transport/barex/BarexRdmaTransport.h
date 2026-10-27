#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"
#include "rtp_llm/cpp/rdma_transport/DeferredReadQueue.h"

namespace arpc {
class RdmaMempool;
}

namespace accl {
namespace barex {
class XClientTransport;
class XConnection;
class XServerTransport;
class XSimpleMempool;
}  // namespace barex
}  // namespace accl

namespace rtp_llm {
namespace rdma_transport {

struct BarexRdmaConnectionState;

class BarexRdmaRuntime final {
public:
    enum class Mode {
        EXPORT,
        READ
    };

    BarexRdmaRuntime(const RdmaConfig& config, Mode mode, int device_id);
    ~BarexRdmaRuntime();

    bool init();

    bool create(const std::vector<torch::Tensor>& tensors, RdmaDescriptor* descriptor);
    void release(const std::vector<std::string>& lease_ids);
    RdmaReadResult read(const std::vector<RdmaDescriptor>& descriptors,
                        int64_t timeout_ms = 0,
                        std::function<void()> release_after_completion = {},
                        bool* release_deferred = nullptr);

private:
    struct Slot {
        void* addr{nullptr};
        uint64_t allocated_bytes{0};
    };

    std::string genLeaseId();
    void releaseSlot(const Slot& slot);
    void gcLoop();  // Reclaims confirmed late reads; exported slots never expire by time.

    std::vector<std::shared_ptr<::accl::barex::XConnection>>
    getConnections(const std::string& ip, uint32_t port, int64_t timeout_ms);
    void invalidateConnection(const std::string& ip,
                              uint32_t port,
                              const std::shared_ptr<::accl::barex::XConnection>& connection);
    bool readOne(const RdmaDescriptor& descriptor,
                 std::vector<torch::Tensor>* tensors,
                 int64_t timeout_ms,
                 std::function<void()> release_after_completion,
                 bool* release_deferred);

private:
    RdmaConfig config_;
    Mode mode_;
    int device_id_;

    bool factory_acquired_{false};
    std::shared_ptr<::arpc::RdmaMempool> mempool_;
    std::shared_ptr<::accl::barex::XSimpleMempool> x_mempool_;

    // Export mode only.
    std::shared_ptr<::accl::barex::XServerTransport> server_;
    std::string rdma_ip_;
    uint32_t rdma_port_{0};

    // Read mode only.
    std::shared_ptr<::accl::barex::XClientTransport> client_;
    std::mutex connections_mutex_;
    std::map<std::string, std::shared_ptr<BarexRdmaConnectionState>> connections_;

    // Exported lease table and timeout reclamation.
    RdmaDeferredReadQueue deferred_reads_;
    std::mutex slots_mutex_;
    uint64_t inflight_bytes_{0};
    std::map<std::string, Slot> slots_;
    std::atomic<uint64_t> lease_seq_{0};
    std::atomic<bool> running_{false};
    std::thread gc_thread_;
};

}  // namespace rdma_transport
}  // namespace rtp_llm
