#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/config/RdmaConfig.h"
#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"

namespace py = pybind11;

namespace rtp_llm {

// Python-facing ViT-side RDMA output exporter.
class MMRdmaExporter {
public:
    explicit MMRdmaExporter(const py::object& rdma_config);
    MMRdmaExporter(const py::object& rdma_config, int device_id);
    explicit MMRdmaExporter(const RdmaConfig& rdma_config);
    MMRdmaExporter(const RdmaConfig& rdma_config, int device_id);

    bool enabled() const {
        return exporter_ != nullptr;
    }

    std::vector<py::bytes> exportEmbedding(torch::Tensor                embedding,
                                           std::optional<torch::Tensor> pos_id,
                                           std::vector<torch::Tensor>   extra_inputs);
    void release(const std::vector<std::string>& handles);

private:
    MMRdmaExporter(std::shared_ptr<rdma_transport::RdmaExport> exporter, int64_t max_slot_bytes):
        exporter_(std::move(exporter)), max_slot_bytes_(max_slot_bytes) {}

    bool exportSlots(const torch::Tensor&                embedding,
                     const std::optional<torch::Tensor>& pos_id,
                     const std::vector<torch::Tensor>&   extra_inputs,
                     std::vector<MMRdmaSlotPB>*          slots);

    std::shared_ptr<rdma_transport::RdmaExport> exporter_;
    int64_t                                    max_slot_bytes_ = 0;
    std::mutex                                 provider_mutex_;
};

void registerMMRdmaExporter(py::module& m);

}  // namespace rtp_llm
