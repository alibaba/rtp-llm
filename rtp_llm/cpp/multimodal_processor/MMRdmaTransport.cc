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
    if (vit_config.mm_transport_mode == "grpc") {
        RTP_LLM_LOG_INFO("mm transport mode is grpc, skip rdma initialization");
        return nullptr;
    }
    if (vit_config.mm_transport_mode != "auto") {
        RTP_LLM_LOG_WARNING("unknown mm transport mode '%s', fall back to grpc", vit_config.mm_transport_mode.c_str());
        return nullptr;
    }
    if (g_mm_rdma_transport_creator == nullptr) {
        RTP_LLM_LOG_WARNING(
            "mm transport mode is auto but no MMRdmaTransport implementation is linked (open-source build?), "
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
