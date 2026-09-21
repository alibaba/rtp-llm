#include "rtp_llm/cpp/multimodal_processor/MMRdmaTransport.h"

#include <limits>
#include <stdexcept>

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
    if (vit_config.mm_transport_mode != "auto" && vit_config.mm_transport_mode != "rdma") {
        throw std::invalid_argument("mm_transport_mode must be grpc, auto, or rdma");
    }
    if (vit_config.mm_rdma_port < 0 || vit_config.mm_rdma_port > 65535 || vit_config.mm_rdma_connect_timeout_ms <= 0
        || vit_config.mm_rdma_read_timeout_ms <= 0 || vit_config.mm_rdma_release_timeout_ms <= 0
        || vit_config.mm_rdma_max_inflight_bytes <= 0 || vit_config.mm_rdma_max_slot_bytes <= 0
        || vit_config.mm_rdma_max_slot_bytes > vit_config.mm_rdma_max_inflight_bytes) {
        throw std::invalid_argument("Invalid ViT RDMA port, timeout, or pool capacity");
    }
    if (g_mm_rdma_transport_creator == nullptr) {
        RTP_LLM_LOG_WARNING(
            "mm transport mode is auto but no MMRdmaTransport implementation is linked (open-source build?), "
            "fall back to inline-bytes path");
        if (vit_config.mm_transport_mode == "rdma") {
            throw std::runtime_error("RDMA transport requested but no provider is linked");
        }
        return nullptr;
    }
    std::shared_ptr<MMRdmaTransport> transport;
    try {
        transport = g_mm_rdma_transport_creator(vit_config, role);
    } catch (const std::exception& error) {
        if (vit_config.mm_transport_mode == "rdma") {
            throw;
        }
        RTP_LLM_LOG_WARNING("mm rdma initialization failed: %s", error.what());
    }
    if (transport == nullptr) {
        if (vit_config.mm_transport_mode == "rdma") {
            throw std::runtime_error("RDMA transport initialization failed");
        }
        RTP_LLM_LOG_WARNING("create mm rdma transport failed, fall back to inline-bytes path");
    }
    return transport;
}

bool validateMMRdmaDescriptor(const MMRdmaDescPB& desc) {
    if (desc.addr() == 0 || desc.handle().empty() || desc.rdma_ip().empty() || desc.rdma_port() == 0
        || desc.rdma_port() > 65535 || desc.nic_rkeys_size() == 0 || desc.tensors_size() != 1 || desc.nbytes() == 0
        || desc.nbytes() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())
        || desc.addr() > std::numeric_limits<uint64_t>::max() - desc.nbytes()) {
        return false;
    }
    for (const auto& tensor : desc.tensors()) {
        if (tensor.role() != MMRdmaTensorPB::EMBEDDING || tensor.shape_size() != 2) {
            return false;
        }
        uint64_t element_bytes;
        switch (tensor.data_type()) {
            case TensorPB::FP32:
            case TensorPB::INT32:
                element_bytes = 4;
                break;
            case TensorPB::FP16:
            case TensorPB::BF16:
                element_bytes = 2;
                break;
            default:
                return false;
        }
        if (tensor.shape_size() == 0 || tensor.offset() % element_bytes != 0 || tensor.offset() > desc.nbytes()
            || tensor.nbytes() > desc.nbytes() - tensor.offset()) {
            return false;
        }
        uint64_t bytes = element_bytes;
        for (int64_t dim : tensor.shape()) {
            if (dim <= 0 || bytes > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(dim)) {
                return false;
            }
            bytes *= static_cast<uint64_t>(dim);
        }
        if (bytes != tensor.nbytes()) {
            return false;
        }
    }
    return true;
}

}  // namespace rtp_llm
