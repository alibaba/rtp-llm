#pragma once

#include <chrono>
#include <memory>
#include <optional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <torch/python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransport.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"
namespace py = pybind11;
namespace rtp_llm {
// Python-facing ViT-side RDMA output exporter.
class MMRdmaOutputExporter {
public:
    explicit MMRdmaOutputExporter(const py::object& rdma_config);
    explicit MMRdmaOutputExporter(const RdmaConfig& rdma_config);

    bool enabled() const {
        return exporter_ != nullptr;
    }

    // Returns one serialized descriptor per slot, or an empty list for inline fallback.
    // Oversized embeddings are row-split; position and extra tensors must fit one slot each.
    std::vector<py::bytes> exportEmbedding(torch::Tensor                embedding,
                                           std::optional<torch::Tensor> pos_id,
                                           std::vector<torch::Tensor>   extra_inputs);

    // Best-effort slot release.
    void release(const std::vector<std::string>& handles);

private:
    MMRdmaOutputExporter(std::shared_ptr<rdma_transport::RdmaExport> exporter, int64_t max_slot_bytes):
        exporter_(std::move(exporter)), max_slot_bytes_(max_slot_bytes) {}

    // Plans slots and rolls back partial exports on failure.
    bool exportSlots(const torch::Tensor&                embedding,
                     const std::optional<torch::Tensor>& pos_id,
                     const std::vector<torch::Tensor>&   extra_inputs,
                     std::vector<MMRdmaSlotPB>*          slots);

    std::shared_ptr<rdma_transport::RdmaExport> exporter_;
    // Zero means unbounded.
    int64_t max_slot_bytes_ = 0;
};

void registerMMRdmaOutputExporter(py::module& m);

// Reassembles and validates tensors read from one or more RDMA slots.
bool assembleMMRdmaOutput(const std::vector<torch::Tensor>&        mm_tensors,
                          const std::vector<MMRdmaSlotPB::Role>&   roles,
                          const MultimodalOutputPB*                output_pb,
                          MultimodalOutput*                        mm_output);

class RdmaCircuitBreaker {
public:
    static constexpr int kFailuresToOpen = 3;
    static constexpr int kOpenSeconds    = 30;

    bool open(const std::string& endpoint) const;
    void recordFailure(const std::string& endpoint);
    void recordSuccess(const std::string& endpoint);

private:
    struct State {
        int                                   failures = 0;
        std::chrono::steady_clock::time_point open_until{};
    };

    static std::unordered_map<std::string, State>& table();
    static std::mutex&                             mutex();
};

class RdmaReceiptReader: public MMReceiptReader {
public:
    RdmaReceiptReader(std::shared_ptr<rdma_transport::RdmaRead> reader, MMTransportMetricsPtr metrics):
        reader_(std::move(reader)), metrics_(std::move(metrics)) {}
    RdmaReceiptReader(const RdmaConfig& rdma_config, MMTransportMetricsPtr metrics):
        metrics_(std::move(metrics)), rdma_config_(rdma_config) {}

    const char* name() const override {
        return "rdma";
    }

    bool          advertise(const std::string& endpoint, MultimodalInputsPB& request_pb) override;
    void          withdraw(MultimodalInputsPB& request_pb) override;
    bool          matches(const MultimodalOutputPB& receipt) const override;
    ConsumeResult consume(const MultimodalOutputPB& receipt, DeliveryContext& context) override;
    void          discard(const MultimodalOutputPB& receipt, DeliveryContext& context) override;

private:
    bool ensureReader();
    static std::vector<std::string> handlesOf(const MultimodalOutputPB& receipt);

    bool readAllSlots(const MultimodalOutputPB&          receipt,
                      DeliveryContext&                   context,
                      std::vector<torch::Tensor>*        mm_tensors,
                      std::vector<MMRdmaSlotPB::Role>*   roles);
    void noteFailure(const std::string& endpoint);

    std::shared_ptr<rdma_transport::RdmaRead> reader_;
    MMTransportMetricsPtr                      metrics_;
    std::optional<RdmaConfig>                  rdma_config_;
    RdmaCircuitBreaker                         circuit_;
};

// The MM adapter remains installed without a provider so it can reject and release
// incompatible RDMA receipts.
std::unique_ptr<MMReceiptReader> createMMRdmaReceiptReader(std::shared_ptr<rdma_transport::RdmaRead> reader,
                                                           MMTransportMetricsPtr                      metrics);
std::unique_ptr<MMReceiptReader> createLazyMMRdmaReceiptReader(const RdmaConfig& rdma_config,
                                                               MMTransportMetricsPtr metrics);


}  // namespace rtp_llm
