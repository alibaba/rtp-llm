#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/model_rpc/BroadcastManager.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

// Cross-node CPU TP broadcaster over RpcService. Root rank fanouts bytes to TP
// peers; non-root ranks wait on a local inbox filled by the gRPC server thread.
// The logical API intentionally matches CpuTpBroadcaster so execBroadcastCpu can
// choose this path without changing tpSyncModelInputs' packing/unpacking logic.
class RpcCpuTpBroadcaster {
public:
    static RpcCpuTpBroadcaster& instance();

    void initialize(int                      tp_rank,
                    int                      tp_size,
                    int                      dp_rank,
                    int                      world_size,
                    const std::vector<std::string>& worker_grpc_addrs,
                    int                      timeout_ms);

    void reset();

    bool isInitialized() const {
        return initialized_.load(std::memory_order_acquire);
    }

    void broadcast(void* buf, std::size_t nbytes, int root);

    bool handleBroadcastRequest(const CpuTpBroadcastRequestPB& request, CpuTpBroadcastResponsePB* response);

private:
    struct InboxKey {
        std::string group_key;
        uint64_t    seq = 0;
        int         dst_tp_rank = 0;

        bool operator==(const InboxKey& other) const {
            return group_key == other.group_key && seq == other.seq && dst_tp_rank == other.dst_tp_rank;
        }
    };

    struct InboxKeyHash {
        std::size_t operator()(const InboxKey& key) const;
    };

    RpcCpuTpBroadcaster() = default;
    ~RpcCpuTpBroadcaster() = default;
    RpcCpuTpBroadcaster(const RpcCpuTpBroadcaster&)            = delete;
    RpcCpuTpBroadcaster& operator=(const RpcCpuTpBroadcaster&) = delete;

    uint64_t nextSeq();
    std::string makeGroupKey(int dp_rank, int tp_size, int world_size) const;

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> initialized_{false};
    std::atomic<uint64_t> seq_{0};

    int         tp_rank_ = 0;
    int         tp_size_ = 1;
    int         dp_rank_ = 0;
    int         world_size_ = 1;
    int         timeout_ms_ = 3000;
    std::string group_key_;

    std::vector<std::string> peer_addrs_;
    std::vector<int>         peer_tp_ranks_;
    std::shared_ptr<BroadcastManager> broadcast_manager_;

    std::unordered_map<InboxKey, std::string, InboxKeyHash> inbox_;
};

}  // namespace rtp_llm
