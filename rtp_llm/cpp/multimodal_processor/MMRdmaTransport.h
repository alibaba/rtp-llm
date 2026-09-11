#pragma once

#include <memory>
#include <string>
#include <vector>
#include <torch/all.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

// Role the transport plays in the encoder/LLM split.
enum class MMRdmaRole {
    ENCODER_SERVER,  // ViT side: registers + exports embeddings, serves one-sided READs
    LLM_CLIENT,      // LLM side: pulls embeddings via one-sided RDMA READ
};

enum class MMRdmaReadStatus {
    SUCCESS,
    RETRYABLE_ERROR,
    POOL_EXHAUSTED,
    INVALID_DESCRIPTOR,
    // The NIC may still be reading. The caller must not release the remote
    // slot; it remains charged to the encoder's bounded pool until restart.
    IN_FLIGHT,
};

// Abstract data-plane for moving multimodal embeddings between the (separated) ViT
// encoder's registered GPU memory and the LLM's pinned CPU receive pool.
//
// The implementation lives in internal_source and uses the existing Barex
// transports and RdmaMempoolFactory. Open-source builds have no provider;
// auto mode uses inline bytes while explicit rdma mode fails clearly.
class MMRdmaTransport {
public:
    virtual ~MMRdmaTransport() = default;

    // ---- Encoder (ENCODER_SERVER) side ----
    // Pack `tensors` (each a contiguous GPU tensor) contiguously into ONE registered slot
    // and fill `desc` with the slot info {addr, nic_rkeys, handle, rdma_ip, rdma_port, nbytes}
    // plus a per-tensor manifest (`desc.tensors`, parallel to `roles`: role/shape/dtype/offset/
    // nbytes). The slot stays alive until releaseEmbedding(handle). There is no
    // timed reclamation: a timeout alone cannot prove a remote READ finished.
    // `tensors` and `roles` must have equal, non-zero size. Returns false on any failure;
    // the caller must then fall back to inline bytes.
    virtual bool exportEmbedding(const std::vector<torch::Tensor>&        tensors,
                                 const std::vector<MMRdmaTensorPB::Role>& roles,
                                 MMRdmaDescPB*                            desc) = 0;

    // Return the slots backing `handles` to the free pool (MR kept registered). Best-effort.
    virtual void releaseEmbedding(const std::vector<std::string>& handles) = 0;

    // ---- LLM (LLM_CLIENT) side ----
    // Issue a single one-sided RDMA READ directly into a pooled pinned-CPU region, then slice
    // it into one tensor per `desc.tensors()` entry (shape/dtype/offset from the manifest),
    // returned via `out` in the same order. The views retain the pool lease; the region is
    // reusable only after the last view dies. Failed READs return an error; the
    // caller does not restart the already-consumed embedding RPC.
    virtual MMRdmaReadStatus
    readEmbedding(const MMRdmaDescPB& desc, std::vector<torch::Tensor>* out, int64_t timeout_ms) = 0;
};

// Creator registered by the internal implementation at static-init time (alwayslink).
using MMRdmaTransportCreator = std::shared_ptr<MMRdmaTransport> (*)(const VitConfig&, MMRdmaRole);
void registerMMRdmaTransportCreator(MMRdmaTransportCreator creator);

// Returns nullptr in grpc mode, when RDMA is unavailable (open-source build / no NIC),
// or when initialization fails. Auto mode callers then use inline gRPC bytes.
std::shared_ptr<MMRdmaTransport> createMMRdmaTransport(const VitConfig& vit_config, MMRdmaRole role);

// Validate all ranges before issuing any remote READ or allocating a receive buffer.
bool validateMMRdmaDescriptor(const MMRdmaDescPB& desc);

}  // namespace rtp_llm
