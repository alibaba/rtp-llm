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

// Abstract data-plane for moving multimodal embeddings between the (separated) ViT
// encoder and the LLM over GPUDirect RDMA.
//
// The concrete implementation lives in internal_source and reuses the cache_store
// RDMA stack (RdmaMemoryUtilImpl / RdmaServer / RdmaClient / RdmaConnection). The
// open-source build links no implementation, so createMMRdmaTransport() returns
// nullptr and every caller silently falls back to the inline-bytes path.
class MMRdmaTransport {
public:
    virtual ~MMRdmaTransport() = default;

    // ---- Encoder (ENCODER_SERVER) side ----
    // Pack `tensors` (each a contiguous GPU tensor) contiguously into ONE registered slot
    // and fill `desc` with the slot info {addr, nic_rkeys, handle, rdma_ip, rdma_port, nbytes}
    // plus a per-tensor manifest (`desc.tensors`, parallel to `roles`: role/shape/dtype/offset/
    // nbytes). The slot stays alive until releaseEmbedding(handle) or the GC timeout fires.
    // `tensors` and `roles` must have equal, non-zero size. Returns false on any failure;
    // the caller must then fall back to inline bytes.
    virtual bool exportEmbedding(const std::vector<torch::Tensor>&         tensors,
                                 const std::vector<MMRdmaTensorPB::Role>&  roles,
                                 MMRdmaDescPB*                             desc) = 0;

    // Return the slots backing `handles` to the free pool (MR kept registered). Best-effort.
    virtual void releaseEmbedding(const std::vector<std::string>& handles) = 0;

    // ---- LLM (LLM_CLIENT) side ----
    // Issue a single one-sided RDMA READ pulling the whole slot described by `desc`, then
    // slice it into one tensor per `desc.tensors()` entry (shape/dtype/offset from the
    // manifest), returned via `out` in the same order. Blocks until completion or timeout.
    // Returns false on any failure; the caller falls back to bytes if present.
    virtual bool readEmbedding(const MMRdmaDescPB& desc, std::vector<torch::Tensor>* out) = 0;
};

// Creator registered by the internal implementation at static-init time (alwayslink).
using MMRdmaTransportCreator = std::shared_ptr<MMRdmaTransport> (*)(const VitConfig&, MMRdmaRole);
void registerMMRdmaTransportCreator(MMRdmaTransportCreator creator);

// Returns nullptr in grpc mode, when RDMA is unavailable (open-source build / no NIC),
// or when initialization fails. Auto mode callers then use inline gRPC bytes.
std::shared_ptr<MMRdmaTransport> createMMRdmaTransport(const VitConfig& vit_config, MMRdmaRole role);

}  // namespace rtp_llm
