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
class MMRdmaEncoderOp {
public:
    // Takes the Python VitConfig object directly; mm-rdma fields are pulled out via the
    // shared extractMMRdmaVitConfig mapping, so there is no separate Python-side copy.
    explicit MMRdmaEncoderOp(const py::object& vit_config);

    // True when a real RDMA transport was created (flag on + impl linked + init ok).
    bool enabled() const {
        return transport_ != nullptr;
    }

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
