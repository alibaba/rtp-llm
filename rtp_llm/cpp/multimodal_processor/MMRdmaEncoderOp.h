#pragma once

#include <memory>
#include <string>
#include <vector>
#include <torch/python.h>
#include <pybind11/stl.h>

#include "rtp_llm/cpp/multimodal_processor/MMRdmaTransport.h"

namespace rtp_llm {

// Exports one V1 per-image embedding. The existing unary response size limit
// remains in force; this adapter does not concatenate or split image outputs.
class MMRdmaEncoderOp {
public:
    explicit MMRdmaEncoderOp(const pybind11::object& vit_config);
    bool enabled() const {
        return transport_ != nullptr;
    }
    pybind11::bytes exportEmbedding(const torch::Tensor& embedding);
    void            release(const std::vector<std::string>& handles);

private:
    std::shared_ptr<MMRdmaTransport> transport_;
};

void registerMMRdmaEncoderOp(pybind11::module& m);

}  // namespace rtp_llm
