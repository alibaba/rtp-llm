#include "rtp_llm/cpp/cache/connector/p2p/transfer/RdmaInterface.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::shared_ptr<IRdmaMemoryManager> createRdmaMemoryManager() {
    RTP_LLM_LOG_WARNING("create rdma memory manager not supported");
    return nullptr;
}

std::shared_ptr<IRdmaClient> createRdmaClient(const std::shared_ptr<IRdmaMemoryManager>& memory_manager,
                                              int                                        io_thread_count,
                                              int                                        worker_thread_count,
                                              uint32_t                                   rdma_connections_per_host,
                                              int                                        connect_timeout_ms) {
    if (!memory_manager) {
        RTP_LLM_LOG_ERROR("create rdma client failed, memory manager is nullptr");
        return nullptr;
    }
    RTP_LLM_LOG_WARNING("rdma client not supported");
    return nullptr;
}

std::shared_ptr<IRdmaServer> createRdmaServer(const std::shared_ptr<IRdmaMemoryManager>& memory_manager,
                                              uint32_t                                   listen_port,
                                              int                                        io_thread_count,
                                              int                                        worker_thread_count) {
    if (!memory_manager) {
        RTP_LLM_LOG_ERROR("create rdma server failed, memory manager is nullptr");
        return nullptr;
    }
    RTP_LLM_LOG_WARNING("rdma server not supported");
    return nullptr;
}

}  // namespace rtp_llm
