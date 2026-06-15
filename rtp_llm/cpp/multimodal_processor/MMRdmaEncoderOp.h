#pragma once

#include <memory>
#include <string>
#include <vector>
#include <torch/python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/multimodal_processor/MMRdmaTransport.h"

namespace py = pybind11;

namespace rtp_llm {

// Python-facing handle for the encoder (ViT) side of the embedding RDMA fast path.
// Constructed by the separated ViT gRPC server; exportEmbedding() registers a GPU
// embedding and returns a serialized MMRdmaDescPB the server forwards to the LLM.
class MMRdmaEncoderOp {
public:
    // Takes the Python VitConfig object directly; mm-rdma fields are pulled out via the
    // shared extractMMRdmaVitConfig mapping, so there is no separate Python-side copy.
    explicit MMRdmaEncoderOp(const py::object& vit_config);

    // True when a real RDMA transport was created (flag on + impl linked + init ok).
    bool enabled() const {
        return transport_ != nullptr;
    }

    // Returns a serialized MMRdmaDescPB; empty string => failure (caller falls back to bytes).
    py::bytes exportEmbedding(torch::Tensor embedding);

    // Return slots backing the given handles to the pool. Best-effort.
    void release(const std::vector<std::string>& handles);

private:
    std::shared_ptr<MMRdmaTransport> transport_;
};

void registerMMRdmaEncoderOp(py::module& m);

}  // namespace rtp_llm
