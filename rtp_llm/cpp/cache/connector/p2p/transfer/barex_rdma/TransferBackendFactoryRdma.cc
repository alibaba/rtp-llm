#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h"

#include <chrono>
#include <cstdint>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaKVCacheReceiver.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {

namespace {

TransferBackendPair createTcpBackend(const TransferBackendConfig&       config,
                                     const kmonitor::MetricsReporterPtr& metrics_reporter) {
    auto sender = std::make_shared<tcp::TcpKVCacheSender>(metrics_reporter);
    const auto idle_ms = config.tcp_channel_idle_ttl_ms > 0 ? config.tcp_channel_idle_ttl_ms : int64_t{0};
    const auto sweep_n = config.tcp_channel_sweep_interval_calls > 0
                             ? static_cast<std::uint64_t>(config.tcp_channel_sweep_interval_calls)
                             : std::uint64_t{0};
    if (!sender->init(config.messager_io_thread_count,
                      std::chrono::milliseconds(idle_ms),
                      sweep_n)) {
        RTP_LLM_LOG_ERROR("createTcpBackend: TcpKVCacheSender init failed");
        return {};
    }

    auto receiver = std::make_shared<tcp::TcpKVCacheReceiver>(metrics_reporter);
    if (!receiver->init(config.cache_store_listen_port,
                        config.messager_io_thread_count,
                        config.messager_worker_thread_count,
                        static_cast<uint32_t>(config.cache_store_tcp_anet_rpc_thread_num),
                        static_cast<uint32_t>(config.cache_store_tcp_anet_rpc_queue_num),
                        config.transfer_wait_check_interval_us)) {
        RTP_LLM_LOG_ERROR("createTcpBackend: TcpKVCacheReceiver init failed");
        return {};
    }

    return {sender, receiver};
}

TransferBackendPair createBarexRdmaBackend(const TransferBackendConfig&       config,
                                           const kmonitor::MetricsReporterPtr& metrics_reporter) {
    auto memory_manager = barex_rdma::createRdmaMemoryManager();

    auto sender = std::make_shared<barex_rdma::RdmaKVCacheSender>(memory_manager, metrics_reporter);
    if (!sender->init(config.messager_io_thread_count,
                      config.messager_io_thread_count,
                      config.messager_worker_thread_count)) {
        RTP_LLM_LOG_ERROR("createBarexRdmaBackend: RdmaKVCacheSender init failed");
        return {};
    }

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
                        config.transfer_wait_check_interval_us)) {
        RTP_LLM_LOG_ERROR("createBarexRdmaBackend: RdmaKVCacheReceiver init failed");
        return {};
    }

    return {sender, receiver};
}

}  // anonymous namespace

TransferBackendPair createTransferBackend(TransferBackend                     backend,
                                          const TransferBackendConfig&         config,
                                          const kmonitor::MetricsReporterPtr& metrics_reporter) {
    switch (backend) {
        case TransferBackend::kTcp:
            return createTcpBackend(config, metrics_reporter);
        case TransferBackend::kBarexRdma:
            return createBarexRdmaBackend(config, metrics_reporter);
        default:
            RTP_LLM_LOG_ERROR("createTransferBackend: unknown backend");
            return {};
    }
}

}  // namespace transfer
}  // namespace rtp_llm
