#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorScheduler.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

P2PConnectorScheduler::P2PConnectorScheduler(P2PConnectorSchedulerConfig         config,
                                             const kmonitor::MetricsReporterPtr& metrics_reporter):
    config_(std::move(config)), metrics_reporter_(metrics_reporter) {}

P2PConnectorScheduler::~P2PConnectorScheduler() = default;

void P2PConnectorScheduler::stopChecker() {
    if (decode_) {
        decode_->stopChecker();
    }
}

bool P2PConnectorScheduler::init(const std::string& process_id) {
    RTP_LLM_LOG_INFO("init start");
    tp_broadcast_client_ =
        std::make_shared<P2PBroadcastClient>(config_.worker_grpc_addrs, config_.p2p_cancel_broadcast_timeout_ms);
    if (!tp_broadcast_client_->init()) {
        RTP_LLM_LOG_ERROR("init failed: tp_broadcast_client init failed");
        return false;
    }

    prefill_ = std::make_unique<P2PConnectorSchedulerPrefill>(config_, metrics_reporter_, tp_broadcast_client_);

    decode_ = std::make_unique<P2PConnectorSchedulerDecode>(config_, metrics_reporter_, tp_broadcast_client_);
    if (!decode_->init(process_id)) {
        RTP_LLM_LOG_ERROR("init failed: decode scheduler init failed");
        return false;
    }

    RTP_LLM_LOG_INFO("init success");
    return true;
}

P2PConnectorScheduler::AsyncReadResult P2PConnectorScheduler::asyncRead(const KVCacheResourcePtr&    resource,
                                                                        const std::shared_ptr<Meta>& meta,
                                                                        const std::pair<int, int>&   block_range,
                                                                        bool                         no_transfer) {
    return decode_->asyncRead(resource, meta, block_range, no_transfer);
}

void P2PConnectorScheduler::cancel(const std::shared_ptr<P2PConnectorAsyncReadContext>& context) {
    if (decode_) {
        decode_->cancel(context);
    }
}

ErrorInfo
P2PConnectorScheduler::sendKVCache(const KVCacheResourcePtr&                            resource,
                                   const std::string&                                   unique_key,
                                   int64_t                                              request_id,
                                   const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                                   int64_t                                              deadline_ms,
                                   std::function<bool()>                                is_cancelled,
                                   bool                                                 no_transfer,
                                   int64_t                                              request_deadline_ms) {
    return prefill_->sendKVCache(resource,
                                 unique_key,
                                 request_id,
                                 decode_transfer_servers,
                                 deadline_ms,
                                 std::move(is_cancelled),
                                 no_transfer,
                                 request_deadline_ms);
}

}  // namespace rtp_llm
