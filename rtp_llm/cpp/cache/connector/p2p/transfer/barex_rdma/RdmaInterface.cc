// Real RDMA factory implementations for barex RDMA-enabled builds.
// In open-source / no-RDMA builds, transfer/RdmaInterface.cc provides stubs instead.
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaMemoryManager.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaServer.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

std::shared_ptr<IRdmaMemoryManager> createRdmaMemoryManager() {
    auto manager = std::make_shared<RdmaMemoryManager>();
    return manager;
}

std::shared_ptr<IRdmaClient> createRdmaClient(const std::shared_ptr<IRdmaMemoryManager>& memory_manager,
                                              int                                        io_thread_count,
                                              int                                        worker_thread_count,
                                              uint32_t                                   rdma_connections_per_host) {
    if (!memory_manager) {
        RTP_LLM_LOG_ERROR("createRdmaClient: memory_manager is nullptr");
        return nullptr;
    }
    auto client = std::make_shared<RdmaClient>(memory_manager, rdma_connections_per_host);
    if (!client->init(io_thread_count, worker_thread_count)) {
        RTP_LLM_LOG_ERROR("createRdmaClient: client init failed");
        return nullptr;
    }
    return client;
}

std::shared_ptr<IRdmaServer> createRdmaServer(const std::shared_ptr<IRdmaMemoryManager>& memory_manager,
                                              uint32_t                                   listen_port,
                                              int                                        io_thread_count,
                                              int                                        worker_thread_count) {
    if (!memory_manager) {
        RTP_LLM_LOG_ERROR("createRdmaServer: memory_manager is nullptr");
        return nullptr;
    }
    auto server = std::make_shared<RdmaServer>(memory_manager);
    if (!server->init(listen_port, io_thread_count, worker_thread_count)) {
        RTP_LLM_LOG_ERROR("createRdmaServer: server init failed on port %u", listen_port);
        return nullptr;
    }
    return server;
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
