#pragma once

#include <memory>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "rtp_llm/cpp/config/MMKvcmConfig.h"
#include "rtp_llm/cpp/multimodal_processor/transport/kvcm/MMKvcmClient.h"

namespace py = pybind11;

namespace rtp_llm {

// Python-facing ViT-side exact-size object writer.
class MMKvcmWriter {
public:
    explicit MMKvcmWriter(const py::object& kvcm_config);
    explicit MMKvcmWriter(const MMKvcmConfig& kvcm_config);

    bool enabled() const {
        return client_ != nullptr;
    }

    void save(const std::vector<std::string>& keys, const std::vector<torch::Tensor>& tensors);
    void remove(const std::vector<std::string>& keys);

private:
    std::shared_ptr<MMKvcmClient> client_;
    uint64_t                      max_object_bytes_  = 0;
    uint64_t                      max_receipt_bytes_ = 0;
};

void registerMMKvcmWriter(py::module& module);

}  // namespace rtp_llm
