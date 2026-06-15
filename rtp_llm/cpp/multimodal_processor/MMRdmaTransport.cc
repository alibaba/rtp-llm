#include "rtp_llm/cpp/multimodal_processor/MMRdmaTransport.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {
MMRdmaTransportCreator g_mm_rdma_transport_creator = nullptr;
}  // namespace

void registerMMRdmaTransportCreator(MMRdmaTransportCreator creator) {
    g_mm_rdma_transport_creator = creator;
}

std::shared_ptr<MMRdmaTransport> createMMRdmaTransport(const VitConfig& vit_config, MMRdmaRole role) {
    if (!vit_config.mm_rdma_enable) {
        RTP_LLM_LOG_INFO("mm rdma disabled, fall back to inline-bytes multimodal embedding transport");
        return nullptr;
    }
    if (g_mm_rdma_transport_creator == nullptr) {
        RTP_LLM_LOG_WARNING(
            "mm_rdma_enable=true but no MMRdmaTransport implementation is linked (open-source build?), "
            "fall back to inline-bytes path");
        return nullptr;
    }
    auto transport = g_mm_rdma_transport_creator(vit_config, role);
    if (transport == nullptr) {
        RTP_LLM_LOG_WARNING("create mm rdma transport failed, fall back to inline-bytes path");
    }
    return transport;
}

}  // namespace rtp_llm
