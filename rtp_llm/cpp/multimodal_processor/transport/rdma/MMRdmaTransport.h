#pragma once

#include <memory>
#include <string>
#include <vector>
#include <torch/all.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

inline constexpr uint64_t kMMRdmaSlotAlign = 256;

enum class MMRdmaRole {
    ENCODER_SERVER,
    LLM_CLIENT,
};

// GPUDirect RDMA data-plane interface between the ViT encoder and LLM. The concrete implementation is
// registered by internal_source; builds without it fall back to inline gRPC bytes.
class MMRdmaTransport {
public:
    virtual ~MMRdmaTransport() = default;

    // Pack tensors into one registered slot. The manifest must match roles, shapes, offsets and
    // byte lengths. Thread-safe; success means source tensors may be reused.
    virtual bool exportEmbedding(const std::vector<torch::Tensor>&         tensors,
                                 const std::vector<MMRdmaTensorPB::Role>&  roles,
                                 MMRdmaDescPB*                             desc) = 0;

    // Return slots to the free pool. Best-effort, idempotent and thread-safe.
    virtual void releaseEmbedding(const std::vector<std::string>& handles) = 0;

    // Read one slot and return owned tensors in manifest order. Blocks until completion or timeout;
    // thread-safe and visible to the caller's current CUDA stream.
    virtual bool readEmbedding(const MMRdmaDescPB& desc,
                               std::vector<torch::Tensor>* out,
                               int64_t                    timeout_ms = 0) = 0;
};

// Registered by the internal implementation at static initialization.
using MMRdmaTransportCreator = std::shared_ptr<MMRdmaTransport> (*)(const MMRdmaConfig&, MMRdmaRole);
// Returns the previous creator so tests can restore it.
MMRdmaTransportCreator registerMMRdmaTransportCreator(MMRdmaTransportCreator creator);

// Returns nullptr when no RDMA implementation is linked or initialization fails.
std::shared_ptr<MMRdmaTransport> createMMRdmaTransport(const MMRdmaConfig& config, MMRdmaRole role);

}  // namespace rtp_llm
