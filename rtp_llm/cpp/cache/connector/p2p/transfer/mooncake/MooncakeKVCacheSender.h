#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeRemoteDescriptor.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferEngineAdapter.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

class IMooncakeControlPlaneClient {
public:
    virtual ~IMooncakeControlPlaneClient() = default;

    virtual bool init(int io_thread_count) = 0;

    virtual bool prepare(const std::string& ip,
                         uint32_t port,
                         const std::string& unique_key,
                         int64_t deadline_ms,
                         MooncakeRemoteDescriptor* descriptor,
                         TransferErrorCode* error_code,
                         std::string* error_message) = 0;

    virtual bool finish(const std::string& ip,
                        uint32_t port,
                        const std::string& unique_key,
                        bool success,
                        TransferErrorCode error_code,
                        const std::string& error_message,
                        TransferErrorCode* response_error_code,
                        std::string* response_error_message) = 0;
};

using IMooncakeControlPlaneClientPtr = std::shared_ptr<IMooncakeControlPlaneClient>;

IMooncakeControlPlaneClientPtr createMooncakeControlPlaneClient();

class MooncakeKVCacheSender : public IKVCacheSender {
public:
    explicit MooncakeKVCacheSender(const IMooncakeTransferEngineAdapterPtr& adapter,
                                   const IMooncakeControlPlaneClientPtr& control_plane_client = nullptr,
                                   const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);

    bool init(const TransferBackendConfig& config);

    bool regMem(const BlockInfo& block_info, uint64_t aligned_size = 0) override;

    void send(const SendRequest& request,
              std::function<void(TransferErrorCode, const std::string&)> callback) override;

private:
    bool buildWriteRequests(const SendRequest& request,
                            const MooncakeRemoteDescriptor& descriptor,
                            std::vector<MooncakeWriteRequest>* write_requests,
                            TransferErrorCode* error_code,
                            std::string* error_message) const;
    bool waitTransferDone(uint64_t batch_id,
                          int64_t deadline_ms,
                          TransferErrorCode* error_code,
                          std::string* error_message) const;
    uint32_t resolveControlPlanePort(const SendRequest& request) const;

private:
    IMooncakeTransferEngineAdapterPtr adapter_;
    IMooncakeControlPlaneClientPtr    control_plane_client_;
    MooncakeBackendConfig             config_;
    kmonitor::MetricsReporterPtr      metrics_reporter_;
};

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm
