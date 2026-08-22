#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <torch/python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaTransport.h"

namespace py = pybind11;

namespace rtp_llm {

// Python-facing ViT-side RDMA encoder.
class MMRdmaEncoderOp {
public:
    explicit MMRdmaEncoderOp(const py::object& rdma_config);
    explicit MMRdmaEncoderOp(const MMRdmaConfig& rdma_config);

    bool enabled() const {
        return transport_ != nullptr;
    }

    // Returns one serialized descriptor per slot, or an empty list for inline fallback.
    // Oversized embeddings are row-split; position and extra tensors must fit one slot each.
    std::vector<py::bytes> exportEmbedding(torch::Tensor                embedding,
                                           std::optional<torch::Tensor> pos_id,
                                           std::vector<torch::Tensor>   extra_inputs);

    // Best-effort slot release.
    void release(const std::vector<std::string>& handles);

private:
    // Plans slots and rolls back partial exports on failure.
    bool exportSlots(const torch::Tensor&                embedding,
                     const std::optional<torch::Tensor>& pos_id,
                     const std::vector<torch::Tensor>&   extra_inputs,
                     std::vector<MMRdmaDescPB>*          descs);

    std::shared_ptr<MMRdmaTransport> transport_;
    // Zero means unbounded.
    int64_t max_slot_bytes_ = 0;
};

void registerMMRdmaEncoderOp(py::module& m);

}  // namespace rtp_llm
