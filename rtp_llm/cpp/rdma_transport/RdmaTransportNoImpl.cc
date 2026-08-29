#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm::rdma_transport {

bool hasRdmaImplementation() {
    return false;
}

std::shared_ptr<RdmaExport> createRdmaExport(const RdmaConfig&) {
    RTP_LLM_LOG_WARNING("no Tensor RDMA implementation is linked");
    return nullptr;
}

std::shared_ptr<RdmaRead> createRdmaRead(const RdmaConfig&) {
    RTP_LLM_LOG_WARNING("no Tensor RDMA implementation is linked");
    return nullptr;
}

}  // namespace rtp_llm::rdma_transport
