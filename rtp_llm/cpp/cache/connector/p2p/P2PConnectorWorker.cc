#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorker.h"

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include <cstdlib>
#include <exception>

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

    const bool rdma_mode = config_.transfer_backend_config.cache_store_rdma_mode;
    auto       backend   = rdma_mode ? transfer::TransferBackend::kBarexRdma : transfer::TransferBackend::kTcp;
    // Emit selected backend at INFO so the next restart makes it obvious
    // which path the process took. Also dump CACHE_STORE_RDMA_MODE env so
    // we can disambiguate the three possible sources for the value:
    //   effective=0 + env="(unset)" -> env never reached the container
    //   effective=0 + env="1"       -> argparse cmdline override beat env
    //                                  (look for --cache_store_rdma_mode 0
    //                                   in the launch command)
    //   effective=1 + env="1"       -> normal RDMA path, expected
    // CACHE_STORE_RDMA_MODE env binds to CacheStoreConfig.cache_store_rdma_mode
    // via argparse (env_name in cache_store_group_args.py), which is then
    // synced into pd_sep_config in engine_config.setup_pd_sep_config.
    const char* env_raw = std::getenv("CACHE_STORE_RDMA_MODE");
    RTP_LLM_LOG_INFO(
        "P2PConnectorWorker init: effective_cache_store_rdma_mode=%d -> TransferBackend=%s, "
        "env CACHE_STORE_RDMA_MODE=%s",
        rdma_mode ? 1 : 0,
        rdma_mode ? "kBarexRdma" : "kTcp",
        env_raw ? env_raw : "(unset)");

    transfer::TransferBackendPair backend_pair;
    try {
        backend_pair = transfer::createTransferBackend(backend, config_.transfer_backend_config, metrics_reporter_);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("init failed: createTransferBackend threw for backend=%s, error=%s",
                          rdma_mode ? "kBarexRdma" : "kTcp",
                          e.what());
        return false;
    } catch (...) {
        RTP_LLM_LOG_ERROR("init failed: createTransferBackend threw unknown exception for backend=%s",
                          rdma_mode ? "kBarexRdma" : "kTcp");
        return false;
    }
    auto [sender, receiver] = std::move(backend_pair);
    if (!sender || !receiver) {
        RTP_LLM_LOG_ERROR("init failed: createTransferBackend failed for backend=%s",
                          rdma_mode ? "kBarexRdma" : "kTcp");
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

bool P2PConnectorWorker::writeByLayer(int                           layer_id,
                                      const KVCacheResourcePtr&     resource,
                                      int64_t                       request_id,
                                      std::shared_ptr<torch::Event> event,
                                      int64_t                       deadline_ms) {
    return prefill_->writeByLayer(layer_id, resource, request_id, std::move(event), deadline_ms);
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

bool P2PConnectorWorker::queryLeaseStatus(
    const std::string& unique_key, bool& sealed, int& started_ops, int& finished_ops, bool& stopped) {
    return decode_->queryLeaseStatus(unique_key, sealed, started_ops, finished_ops, stopped);
}

}  // namespace rtp_llm
