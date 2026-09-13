#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <torch/python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/multimodal_processor/MMRdmaTransport.h"

namespace py = pybind11;

namespace rtp_llm {

// Python-facing handle for the encoder (ViT) side of the output RDMA fast path.
// Constructed by the separated ViT gRPC server; exportEmbedding() packs a request's
// embedding tensors and optional outputs into one or more RDMA slots and returns
// serialized descriptors for forwarding to the LLM.
class MMRdmaEncoderOp {
public:
    // Takes the Python VitConfig object directly; mm-rdma fields are pulled out via the
    // shared extractMMRdmaVitConfig mapping, so there is no separate Python-side copy.
    explicit MMRdmaEncoderOp(const py::object& vit_config);

    // True when a real RDMA transport was created (flag on + impl linked + init ok).
    bool enabled() const {
        return transport_ != nullptr;
    }

    // Pack one request's embedding tensors, optional concat-ed position ids, and per-image
    // extra_input tensors, in that order, into one or more RDMA slots. Return one serialized
    // MMRdmaDescPB per slot. The embedding tensors stay separate until copied into the slots.
    // A slot is capped by mm_rdma_max_slot_bytes (1 GiB by default); oversized embeddings are
    // row-split, and the LLM concatenates EMBEDDING pieces in descriptor order. Return one
    // descriptor when it fits, N when chunked, or an empty list on failure so the caller falls
    // back to the inline-bytes path.
    std::vector<py::bytes> exportEmbedding(const std::vector<torch::Tensor>&   embeddings,
                                           const std::optional<torch::Tensor>& pos_id,
                                           const std::vector<torch::Tensor>&   extra_inputs);

    // Return slots backing the given handles to the pool. Best-effort.
    void release(const std::vector<std::string>& handles);

private:
    std::shared_ptr<MMRdmaTransport> transport_;
    // Upper bound (bytes) on ONE RDMA slot; outputs larger than this are split across slots.
    // 0 => unbounded (single slot, legacy behavior).
    int64_t max_slot_bytes_ = 0;
};

void registerMMRdmaEncoderOp(py::module& m);

}  // namespace rtp_llm
