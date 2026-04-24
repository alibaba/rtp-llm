#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendFactory.h"

#include <stdexcept>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/mooncake/MooncakeTransferEngineAdapterProvider.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheSender.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheReceiver.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {

namespace {

TransferBackendPair createTcpBackend(const TransferBackendConfig&        config,
                                     const kmonitor::MetricsReporterPtr& metrics_reporter) {
    auto sender = std::make_shared<tcp::TcpKVCacheSender>(metrics_reporter);
    if (!sender->init(config.messager_io_thread_count)) {
        RTP_LLM_LOG_ERROR("createTcpBackend: TcpKVCacheSender init failed");
        return {};
    }

    auto receiver = std::make_shared<tcp::TcpKVCacheReceiver>(metrics_reporter);
    if (!receiver->init(config.cache_store_listen_port,
                        config.messager_io_thread_count,
                        config.messager_worker_thread_count,
                        static_cast<uint32_t>(config.cache_store_tcp_anet_rpc_thread_num),
                        static_cast<uint32_t>(config.cache_store_tcp_anet_rpc_queue_num))) {
        RTP_LLM_LOG_ERROR("createTcpBackend: TcpKVCacheReceiver init failed");
        return {};
    }

    return {sender, receiver};
}

TransferBackendPair createMooncakeBackend(const TransferBackendConfig& config,
                                          const kmonitor::MetricsReporterPtr& metrics_reporter) {
    auto adapter = mooncake::createMooncakeTransferEngineAdapter();
    if (!adapter) {
        RTP_LLM_LOG_ERROR("createMooncakeBackend: Mooncake transfer engine is not enabled in this build");
        return {};
    }

    auto control_plane_client = mooncake::createMooncakeControlPlaneClient();

    auto sender = std::make_shared<mooncake::MooncakeKVCacheSender>(adapter, control_plane_client, metrics_reporter);
    if (!sender->init(config)) {
        RTP_LLM_LOG_ERROR("createMooncakeBackend: MooncakeKVCacheSender init failed");
        return {};
    }

    auto receiver = std::make_shared<mooncake::MooncakeKVCacheReceiver>(adapter, metrics_reporter);
    if (!receiver->init(config)) {
        RTP_LLM_LOG_ERROR("createMooncakeBackend: MooncakeKVCacheReceiver init failed");
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
            throw std::runtime_error("BarexRdma backend not supported in this build");
        case TransferBackend::kMooncake:
            return createMooncakeBackend(config, metrics_reporter);
        default:
            RTP_LLM_LOG_ERROR("createTransferBackend: unknown backend");
            return {};
    }
}

}  // namespace transfer
}  // namespace rtp_llm
