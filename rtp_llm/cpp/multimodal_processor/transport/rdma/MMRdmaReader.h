#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
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
bool assembleMMRdmaOutput(const std::vector<torch::Tensor>&      mm_tensors,
                          const std::vector<MMRdmaSlotPB::Role>& roles,
                          const MultimodalOutputPB*              output_pb,
                          MultimodalOutput*                      mm_output);

class MMRdmaReader: public MMReceiptReader {
public:
    // Explicitly injected readers are trusted adapters; build-selected production readers
    // validate every descriptor before invoking the provider. The single-provider
    // constructor keeps the legacy fully serialized behavior.
    explicit MMRdmaReader(std::shared_ptr<rdma_transport::RdmaRead> reader):
        providers_({std::move(reader)}), slot_busy_(1, false) {}
    // An injected adapter keeps its existing serialized-call contract. Pooling
    // requires explicitly supplying distinct providers through the vector overload.
    MMRdmaReader(std::shared_ptr<rdma_transport::RdmaRead> reader, const RdmaConfig& rdma_config):
        providers_({std::move(reader)}), rdma_config_(rdma_config), validate_descriptors_(true), slot_busy_(1, false) {}
    // Build-selected pool: one dedicated provider per slot. Each provider is lent to at
    // most one caller at a time, so per-provider calls remain serialized.
    MMRdmaReader(std::vector<std::shared_ptr<rdma_transport::RdmaRead>> providers, const RdmaConfig& rdma_config);

    const char* name() const override {
        return "rdma";
    }

    bool          advertise(const std::string& endpoint, MultimodalInputsPB& request_pb) override;
    bool          matches(const MultimodalOutputPB& receipt) const override;
    ConsumeResult consume(const MultimodalOutputPB& receipt, DeliveryContext& context) override;
    void          discard(const MultimodalOutputPB& receipt, DeliveryContext& context) override;

private:
    static std::vector<std::string> handlesOf(const MultimodalOutputPB& receipt);

    // Lends a free provider slot, waiting until deadline. Returns -1 on timeout and
    // reports the wait in *waited_ms either way.
    int  acquireProviderSlot(const std::chrono::steady_clock::time_point& deadline, int64_t* waited_ms);
    void releaseProviderSlot(int slot);
    // Approximate occupancy for diagnostics.
    size_t busySlotCount();

    bool hasProvider() const {
        return !providers_.empty() && providers_.front() != nullptr;
    }

    bool readAllSlots(const MultimodalOutputPB&        receipt,
                      DeliveryContext&                 context,
                      std::vector<torch::Tensor>*      mm_tensors,
                      std::vector<MMRdmaSlotPB::Role>* roles,
                      bool*                            deadline_exhausted,
                      bool*                            release_deferred);

    std::vector<std::shared_ptr<rdma_transport::RdmaRead>> providers_;
    std::optional<RdmaConfig>                              rdma_config_;
    bool                                                   validate_descriptors_ = false;

    // Provider pool state. A slot is lent to exactly one concurrent read; per-provider
    // call serialization is preserved without funneling the whole worker through one lock.
    // All three are guarded by pool_mutex_; slot_busy_ has one entry per provider.
    std::mutex              pool_mutex_;
    std::condition_variable pool_cv_;
    std::vector<bool>       slot_busy_;
    std::atomic<uint64_t>   round_robin_{0};
};

// The receipt reader remains installed without a provider so it can reject and release
// incompatible RDMA receipts.
std::unique_ptr<MMReceiptReader> createMMRdmaReader(std::shared_ptr<rdma_transport::RdmaRead> reader);

}  // namespace rtp_llm
