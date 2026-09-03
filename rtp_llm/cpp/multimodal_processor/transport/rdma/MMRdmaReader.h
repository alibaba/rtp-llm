#pragma once

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <torch/python.h>

#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransport.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/config/RdmaConfig.h"
#include "rtp_llm/cpp/rdma_transport/RdmaTransport.h"

namespace rtp_llm {

// Reassembles and validates tensors read from one or more RDMA slots.
bool assembleMMRdmaOutput(const std::vector<torch::Tensor>&        mm_tensors,
                          const std::vector<MMRdmaSlotPB::Role>&   roles,
                          const MultimodalOutputPB*                output_pb,
                          MultimodalOutput*                        mm_output);

class MMRdmaReader: public MMReceiptReader {
public:
    // Explicitly injected readers are trusted adapters; build-selected production readers
    // validate every descriptor before invoking the provider.
    explicit MMRdmaReader(std::shared_ptr<rdma_transport::RdmaRead> reader): reader_(std::move(reader)) {}
    MMRdmaReader(std::shared_ptr<rdma_transport::RdmaRead> reader, const RdmaConfig& rdma_config):
        reader_(std::move(reader)), rdma_config_(rdma_config), validate_descriptors_(true) {}

    const char* name() const override {
        return "rdma";
    }

    bool advertise(const std::string& endpoint, MultimodalInputsPB& request_pb) override;
    bool matches(const MultimodalOutputPB& receipt) const override;
    ConsumeResult consume(const MultimodalOutputPB& receipt, DeliveryContext& context) override;
    void          discard(const MultimodalOutputPB& receipt, DeliveryContext& context) override;

private:
    static std::vector<std::string> handlesOf(const MultimodalOutputPB& receipt);

    bool readAllSlots(const MultimodalOutputPB&          receipt,
                      DeliveryContext&                   context,
                      std::vector<torch::Tensor>*        mm_tensors,
                      std::vector<MMRdmaSlotPB::Role>*   roles,
                      bool*                              deadline_exhausted);

    std::shared_ptr<rdma_transport::RdmaRead> reader_;
    std::optional<RdmaConfig>                  rdma_config_;
    bool                                      validate_descriptors_ = false;
    std::timed_mutex                           provider_mutex_;
};

// The receipt reader remains installed without a provider so it can reject and release
// incompatible RDMA receipts.
std::unique_ptr<MMReceiptReader> createMMRdmaReader(std::shared_ptr<rdma_transport::RdmaRead> reader);

}  // namespace rtp_llm
