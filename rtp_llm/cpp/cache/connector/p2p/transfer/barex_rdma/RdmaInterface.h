#pragma once

// RDMA interface for barex_rdma. The backend implementation links the closed
// accl-barex library and only builds with --define=use_rdma=true (internal
// overlay).
#include <torch/extension.h>
#include "rtp_llm/cpp/cache/BlockInfo.h"
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <map>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

struct RemoteBuffer {
    int64_t                                       addr;
    uint32_t                                      len;
    std::shared_ptr<std::map<uint32_t, uint32_t>> nic_rkeys;
    RemoteBuffer(int64_t addr, uint32_t len, const std::shared_ptr<std::map<uint32_t, uint32_t>>& nic_rkeys):
        addr(addr), len(len), nic_rkeys(nic_rkeys) {}
};

class IRdmaMemoryManager {
public:
    virtual ~IRdmaMemoryManager() = default;
    virtual bool regUserMr(const BlockInfo& block_info, uint64_t aligned_size = 0) = 0;
    virtual bool deregUserMr(const BlockInfo& block_info) = 0;
    virtual std::shared_ptr<RemoteBuffer> findMemoryMr(const BlockInfo& block_info) = 0;
};

class IRdmaConnection {
public:
    virtual ~IRdmaConnection() = default;
    virtual void read(const std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>>& local_remote_buffers,
                      std::function<void(bool)>                                                  callback,
                      uint64_t                                                                   deadline_ms) = 0;
};

class IRdmaClient {
public:
    virtual ~IRdmaClient() = default;
    virtual std::shared_ptr<IRdmaConnection> getConnection(const std::string& ip, uint32_t port) = 0;
};

class IRdmaServer {
public:
    virtual ~IRdmaServer() = default;
    virtual uint32_t getListenPort() const = 0;
};

std::shared_ptr<IRdmaMemoryManager> createRdmaMemoryManager();
std::shared_ptr<IRdmaClient> createRdmaClient(const std::shared_ptr<IRdmaMemoryManager>& memory_manager,
                                              int                                        io_thread_count,
                                              int                                        worker_thread_count       = 0,
                                              uint32_t                                   rdma_connections_per_host = 8);
std::shared_ptr<IRdmaServer> createRdmaServer(const std::shared_ptr<IRdmaMemoryManager>& memory_manager,
                                              uint32_t                                   listen_port,
                                              int                                        io_thread_count,
                                              int                                        worker_thread_count = 0);

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
