#pragma once

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpServer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeRemoteDescriptor.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferEngineAdapter.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferService.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

class MooncakeKVCacheReceiver : public IKVCacheReceiver, public IMooncakeControlPlaneHandler {
public:
    explicit MooncakeKVCacheReceiver(const IMooncakeTransferEngineAdapterPtr& adapter,
                                     const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);
    ~MooncakeKVCacheReceiver();

    bool init(const TransferBackendConfig& config);

    bool regMem(const BlockInfo& block_info, uint64_t aligned_size = 0) override;

    IKVCacheRecvTaskPtr recv(const RecvRequest& request) override;

    void stealTask(const std::string& unique_key) override;

    IKVCacheRecvTaskPtr getTask(const std::string& unique_key) override;

    bool prepareDescriptor(const std::string& unique_key,
                           int64_t deadline_ms,
                           MooncakeRemoteDescriptor* descriptor,
                           TransferErrorCode* error_code,
                           std::string* error_message) override;

    bool finishTransfer(const std::string& unique_key,
                        bool success,
                        TransferErrorCode error_code,
                        const std::string& error_message,
                        TransferErrorCode* response_error_code,
                        std::string* response_error_message) override;

    const std::shared_ptr<TransferTaskStore>& getTransferTaskStore() const {
        return task_store_;
    }

    const std::shared_ptr<MooncakeTransferService>& getTransferService() const {
        return transfer_service_;
    }

private:
    bool startControlPlaneServer(const TransferBackendConfig& config);
    MooncakeRemoteDescriptor buildDescriptor(const std::string& unique_key, const KeyBlockInfoMap& block_infos) const;
    void eraseDescriptor(const std::string& unique_key);

private:
    IMooncakeTransferEngineAdapterPtr               adapter_;
    MooncakeBackendConfig                           config_;
    std::shared_ptr<TransferTaskStore>              task_store_;
    std::shared_ptr<transfer::TcpServer>            tcp_server_;
    std::shared_ptr<MooncakeTransferService>        transfer_service_;
    mutable std::shared_mutex                       descriptor_mutex_;
    std::unordered_map<std::string, MooncakeRemoteDescriptor> descriptor_index_;
    kmonitor::MetricsReporterPtr                    metrics_reporter_;
};

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm
