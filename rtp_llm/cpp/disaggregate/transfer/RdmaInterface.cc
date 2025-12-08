#include "rtp_llm/cpp/disaggregate/transfer/RdmaInterface.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

std::shared_ptr<IRdmaMemoryManager> createRdmaMemoryManager() {
    RTP_LLM_LOG_WARNING("create rdma memory manager not supported");
    return nullptr;
}

std::shared_ptr<IRdmaClient> createRdmaClient(uint32_t rdma_connections_per_host, int connect_timeout_ms) {
    RTP_LLM_LOG_WARNING("rdma client not supported");
    return nullptr;
}

std::shared_ptr<IRdmaServer> createRdmaServer() {
    RTP_LLM_LOG_WARNING("rdma server not supported");
    return nullptr;
}

}  // namespace rtp_llm
