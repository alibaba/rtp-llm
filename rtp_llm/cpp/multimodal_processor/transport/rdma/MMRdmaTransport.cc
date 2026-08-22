#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaTransport.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
MMRdmaTransportCreator g_mm_rdma_transport_creator = nullptr;
}  // namespace

MMRdmaTransportCreator registerMMRdmaTransportCreator(MMRdmaTransportCreator creator) {
    auto previous = g_mm_rdma_transport_creator;
    g_mm_rdma_transport_creator = creator;
    return previous;
}

std::shared_ptr<MMRdmaTransport> createMMRdmaTransport(const MMRdmaConfig& config, MMRdmaRole role) {
    if (g_mm_rdma_transport_creator == nullptr) {
        RTP_LLM_LOG_WARNING(
            "no MMRdmaTransport implementation is linked (open-source build?), fall back to inline-bytes path");
        return nullptr;
    }
    auto transport = g_mm_rdma_transport_creator(config, role);
    if (transport == nullptr) {
        RTP_LLM_LOG_WARNING("create mm rdma transport failed, fall back to inline-bytes path");
    }
    return transport;
}

}  // namespace rtp_llm
