#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorBackend.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include <cstdlib>
#include <exception>

namespace rtp_llm::p2p_internal {

transfer::TransferBackendPair createAndRegisterTransferBackend(
    const P2PConnectorWorkerConfig&             config,
    const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
    const kmonitor::MetricsReporterPtr&         metrics_reporter) {
    if (!layer_block_converter) {
        RTP_LLM_LOG_ERROR("init failed: layer_block_converter is null");
        return {};
    }

    const bool rdma_mode = config.transfer_backend_config.cache_store_rdma_mode;
    auto       backend   = rdma_mode ? transfer::TransferBackend::kBarexRdma : transfer::TransferBackend::kTcp;
    const char* env_raw  = std::getenv("CACHE_STORE_RDMA_MODE");
    RTP_LLM_LOG_INFO(
        "P2PConnectorWorker init: effective_cache_store_rdma_mode=%d -> TransferBackend=%s, "
        "env CACHE_STORE_RDMA_MODE=%s",
        rdma_mode ? 1 : 0,
        rdma_mode ? "kBarexRdma" : "kTcp",
        env_raw ? env_raw : "(unset)");

    transfer::TransferBackendPair backend_pair;
    try {
        backend_pair = transfer::createTransferBackend(backend, config.transfer_backend_config, metrics_reporter);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("init failed: createTransferBackend threw for backend=%s, error=%s",
                          rdma_mode ? "kBarexRdma" : "kTcp",
                          e.what());
        return {};
    } catch (...) {
        RTP_LLM_LOG_ERROR("init failed: createTransferBackend threw unknown exception for backend=%s",
                          rdma_mode ? "kBarexRdma" : "kTcp");
        return {};
    }

    auto& [sender, receiver] = backend_pair;
    if (!sender || !receiver) {
        RTP_LLM_LOG_ERROR("init failed: createTransferBackend failed for backend=%s",
                          rdma_mode ? "kBarexRdma" : "kTcp");
        return {};
    }

    auto buffers = layer_block_converter->getAllBuffers();
    for (auto& [block_info, size] : buffers) {
        if (!sender->regMem(block_info, size)) {
            RTP_LLM_LOG_ERROR("init failed: sender regMem failed, addr: %p, size: %ld", block_info.addr, size);
            return {};
        }
        if (!receiver->regMem(block_info, size)) {
            if (rdma_mode) {
                RTP_LLM_LOG_ERROR("init failed: receiver regMem failed in RDMA mode, addr: %p, size: %ld",
                                  block_info.addr,
                                  size);
                return {};
            }
            RTP_LLM_LOG_WARNING(
                "receiver regMem failed, addr: %p, size: %ld (non-fatal for TCP mode)", block_info.addr, size);
        }
    }
    return backend_pair;
}

}  // namespace rtp_llm::p2p_internal
