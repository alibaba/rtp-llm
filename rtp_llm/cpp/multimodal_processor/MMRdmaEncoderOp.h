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
// Constructed by the separated ViT gRPC server; exportEmbedding() packs a request's whole
// output into one RDMA slot and returns a serialized MMRdmaDescPB the server forwards to the LLM.
class MMRdmaEncoderOp {
public:
    // Takes the Python VitConfig object directly; mm-rdma fields are pulled out via the
    // shared extractMMRdmaVitConfig mapping, so there is no separate Python-side copy.
    explicit MMRdmaEncoderOp(const py::object& vit_config);

    // True when a real RDMA transport was created (flag on + impl linked + init ok).
    bool enabled() const {
        return transport_ != nullptr;
    }

    // Pack the whole output of one request into a single RDMA slot: the (concat-ed) embedding,
    // the optional (concat-ed) position ids, and the per-image extra_input tensors, in that
    // order. Returns a serialized MMRdmaDescPB (slot info + per-tensor manifest); an empty
    // string => failure, in which case the caller falls back to the inline-bytes path.
    py::bytes exportEmbedding(torch::Tensor                embedding,
                              std::optional<torch::Tensor> pos_id,
                              std::vector<torch::Tensor>   extra_inputs);

    // Return slots backing the given handles to the pool. Best-effort.
    void release(const std::vector<std::string>& handles);

private:
    std::shared_ptr<MMRdmaTransport> transport_;
};

void registerMMRdmaEncoderOp(py::module& m);

}  // namespace rtp_llm
