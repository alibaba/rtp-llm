#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.h"

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

P2PConnectorWorker::P2PConnectorWorker(P2PConnectorWorkerConfig                    config,
                                       const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                                       const kmonitor::MetricsReporterPtr&         metrics_reporter):
    config_(std::move(config)), layer_block_converter_(layer_block_converter), metrics_reporter_(metrics_reporter) {}

P2PConnectorWorker::~P2PConnectorWorker() = default;

bool P2PConnectorWorker::init(int64_t store_wait_timeout_ms) {
    RTP_LLM_LOG_INFO("init start, store_wait_timeout_ms: %ld", store_wait_timeout_ms);
    if (!layer_block_converter_) {
        RTP_LLM_LOG_ERROR("init failed: layer_block_converter is null");
        return false;
    }

    auto backend = config_.transfer_backend_config.cache_store_rdma_mode ? transfer::TransferBackend::kBarexRdma :
                                                                           transfer::TransferBackend::kTcp;
    auto [sender, receiver] =
        transfer::createTransferBackend(backend, config_.transfer_backend_config, metrics_reporter_);
    if (!sender || !receiver) {
        RTP_LLM_LOG_ERROR("init failed: createTransferBackend failed");
        return false;
    }

    auto buffers = layer_block_converter_->getAllBuffers();
    for (auto& [block_info, size] : buffers) {
        if (!sender->regMem(block_info, size)) {
            RTP_LLM_LOG_ERROR("init failed: sender regMem failed, addr: %p, size: %ld", block_info.addr, size);
            return false;
        }
        if (!receiver->regMem(block_info, size)) {
            RTP_LLM_LOG_WARNING(
                "receiver regMem failed, addr: %p, size: %ld (non-fatal for TCP mode)", block_info.addr, size);
        }
    }

    prefill_ = std::make_unique<P2PConnectorWorkerPrefill>(config_, layer_block_converter_, metrics_reporter_, sender);
    if (!prefill_->init(store_wait_timeout_ms)) {
        RTP_LLM_LOG_ERROR("init failed: prefill init failed");
        return false;
    }

    decode_ = std::make_unique<P2PConnectorWorkerDecode>(config_, layer_block_converter_, metrics_reporter_, receiver);

    RTP_LLM_LOG_INFO("init success");
    return true;
}

bool P2PConnectorWorker::writeByLayer(int                       layer_id,
                                      const KVCacheResourcePtr& resource,
                                      int64_t                   request_id,
                                      std::optional<c10::Event> event) {
    return prefill_->writeByLayer(layer_id, resource, request_id, std::move(event));
}

ErrorInfo
P2PConnectorWorker::sendKVCache(int64_t                                              request_id,
                                const std::string&                                   unique_key,
                                int64_t                                              deadline_ms,
                                const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers) {
    return prefill_->sendKVCache(request_id, unique_key, deadline_ms, decode_transfer_servers);
}

ErrorInfo P2PConnectorWorker::read(int64_t                                               request_id,
                                   const std::string&                                    unique_key,
                                   int64_t                                               deadline_ms,
                                   const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                   int                                                   remote_tp_size) {
    return decode_->read(request_id, unique_key, deadline_ms, layer_cache_buffers, remote_tp_size);
}

bool P2PConnectorWorker::cancelRead(const std::string& unique_key) {
    return decode_->cancelRead(unique_key);
}

bool P2PConnectorWorker::cancelSend(const std::string& unique_key) {
    return prefill_->cancelSend(unique_key);
}

std::shared_ptr<ComputedLayerCacheBufferStore> P2PConnectorWorker::getComputedBuffersStore() const {
    return prefill_->getComputedBuffersStore();
}

void P2PConnectorWorker::setStoreWaitTimeoutMs(int64_t store_wait_timeout_ms) {
    prefill_->setStoreWaitTimeoutMs(store_wait_timeout_ms);
}

}  // namespace rtp_llm
