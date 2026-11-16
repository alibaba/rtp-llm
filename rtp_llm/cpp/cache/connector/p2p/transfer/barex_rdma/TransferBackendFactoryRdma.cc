#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpBackendFactory.h"

#include <utility>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaKVCacheReceiver.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {

namespace {

TransferBackendPair createBarexRdmaBackend(const TransferBackendConfig&       config,
                                           TransferBackendRole                 role,
                                           const kmonitor::MetricsReporterPtr& metrics_reporter) {
    auto memory_manager = barex_rdma::createRdmaMemoryManager();

    TransferBackendPair result;
    if (role == TransferBackendRole::kSender) {
        auto sender = std::make_shared<barex_rdma::RdmaKVCacheSender>(memory_manager, metrics_reporter);
        if (!sender->init(config.messager_io_thread_count,
                          config.messager_io_thread_count,
                          config.messager_worker_thread_count)) {
            RTP_LLM_LOG_ERROR("createBarexRdmaBackend: RdmaKVCacheSender init failed");
            return {};
        }
        result.sender = std::move(sender);
    } else {
        auto receiver = std::make_shared<barex_rdma::RdmaKVCacheReceiver>(memory_manager, metrics_reporter);
        if (!receiver->init(config.cache_store_listen_port,
                            config.messager_io_thread_count,
                            config.messager_worker_thread_count,
                            config.messager_io_thread_count,
                            config.messager_worker_thread_count,
                            /*rdma_connections_per_host=*/8,
                            config.rdma_max_block_pairs_per_connection,
                            config.cache_store_tcp_anet_rpc_thread_num,
                            config.cache_store_tcp_anet_rpc_queue_num,
                            config.transfer_wait_check_interval_us,
                            config.cache_store_tcp_worker_queue_size,
                            config.rdma_transfer_worker_thread_count,
                            config.rdma_transfer_worker_queue_size,
                            config.p2p_rdma_enable_h2d_copy,
                            config.p2p_rdma_staging_total_bytes,
                            config.llm_kv_block_size_bytes,
                            config.rdma_disconnect_after_deadline_ms)) {
            RTP_LLM_LOG_ERROR("createBarexRdmaBackend: RdmaKVCacheReceiver init failed");
            return {};
        }
        result.receiver = std::move(receiver);
    }
    return result;
}

}  // anonymous namespace

TransferBackendPair createTransferBackend(TransferBackend                     backend,
                                          const TransferBackendConfig&         config,
                                          TransferBackendRole                 role,
                                          const kmonitor::MetricsReporterPtr& metrics_reporter) {
    switch (backend) {
        case TransferBackend::kTcp:
            return createTcpBackend(config, role, metrics_reporter);
        case TransferBackend::kBarexRdma:
            return createBarexRdmaBackend(config, role, metrics_reporter);
        default:
            RTP_LLM_LOG_ERROR("createTransferBackend: unknown backend");
            return {};
    }
}

}  // namespace transfer
}  // namespace rtp_llm
