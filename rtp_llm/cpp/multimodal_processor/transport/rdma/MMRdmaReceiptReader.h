#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/multimodal_processor/transport/MMRemoteOutputTransport.h"
#include "rtp_llm/cpp/multimodal_processor/transport/rdma/MMRdmaTransport.h"

namespace rtp_llm {

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
    RdmaReceiptReader(std::shared_ptr<MMRdmaTransport> transport, MMTransportMetricsPtr metrics):
        transport_(std::move(transport)), metrics_(std::move(metrics)) {}

    const char* name() const override {
        return "rdma";
    }

    bool          advertise(const std::string& endpoint, MultimodalInputsPB& request_pb) override;
    void          withdraw(MultimodalInputsPB& request_pb) override;
    bool          matches(const MultimodalOutputPB& receipt) const override;
    ConsumeResult consume(const MultimodalOutputPB& receipt, DeliveryContext& context) override;
    void          discard(const MultimodalOutputPB& receipt, DeliveryContext& context) override;

private:
    static std::vector<std::string> handlesOf(const MultimodalOutputPB& receipt);

    bool readAllSlots(const std::vector<const MMRdmaDescPB*>& descs,
                      DeliveryContext&                        context,
                      std::vector<torch::Tensor>*             mm_tensors,
                      std::vector<MMRdmaTensorPB::Role>*      roles);
    void noteFailure(const std::string& endpoint);

    std::shared_ptr<MMRdmaTransport> transport_;
    MMTransportMetricsPtr            metrics_;
    RdmaCircuitBreaker               circuit_;
};

// The reader remains registered without an implementation so it can reject and release
// incompatible RDMA receipts.
std::shared_ptr<MMRdmaTransport> createMMRdmaClientTransport(const MMRdmaConfig& rdma_config);
std::unique_ptr<MMReceiptReader> createMMRdmaReceiptReader(std::shared_ptr<MMRdmaTransport> transport,
                                                           MMTransportMetricsPtr            metrics);

}  // namespace rtp_llm
