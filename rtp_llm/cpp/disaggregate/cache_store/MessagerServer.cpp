#include "rtp_llm/cpp/disaggregate/cache_store/MessagerServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/Interface.h"

#include "rtp_llm/cpp/utils/Logger.h"

#include "aios/network/arpc/arpc/metric/KMonitorANetServerMetricReporter.h"
#include "autil/LockFreeThreadPool.h"

namespace rtp_llm {

MessagerServer::MessagerServer(const std::shared_ptr<MemoryUtil>&                memory_util,
                               const std::shared_ptr<RequestBlockBufferStore>&   request_block_buffer_store,
                               const std::shared_ptr<CacheStoreMetricsReporter>& metrics_reporter,
                               const std::shared_ptr<arpc::TimerManager>&        timer_manager):
    memory_util_(memory_util),
    request_block_buffer_store_(request_block_buffer_store),
    metrics_reporter_(metrics_reporter),
    timer_manager_(timer_manager) {}

MessagerServer::~MessagerServer() {
    if (rpc_server_transport_) {
        rpc_server_transport_->stop();
        rpc_server_transport_->wait();
    }

    if (rpc_server_) {
        rpc_server_->Close();
        rpc_server_.reset();
    }
    rpc_server_transport_.reset();
    rpc_service_.reset();

    request_block_buffer_store_.reset();
}

bool MessagerServer::init(uint32_t listen_port, uint32_t rdma_listen_port, bool enable_metric) {
    if (!initTcpServer(listen_port, enable_metric)) {
        RTP_LLM_LOG_INFO("messager server init failed, tcp server init failed");
        return false;
    }
    return true;
}

bool MessagerServer::initTcpServer(uint32_t listen_port, bool enable_metric) {
    rpc_service_ = std::move(createCacheStoreServiceImpl(memory_util_, request_block_buffer_store_, metrics_reporter_, timer_manager_));

    if (rpc_server_transport_ == nullptr) {
        int tcp_server_io_thread_count = memory_util_->isRdmaMode() ? 1 : 3;
        rpc_server_transport_.reset(new anet::Transport(tcp_server_io_thread_count));
        if (!rpc_server_transport_ || !rpc_server_transport_->start()) {
            return false;
        }
        rpc_server_transport_->setName("ANetLayerCacheMessagerRPCServer");
    }

    rpc_server_.reset(new arpc::ANetRPCServer(rpc_server_transport_.get(), 3, 100));
    if (enable_metric) {
        arpc::KMonitorANetMetricReporterConfig metricConfig;
        metricConfig.metricLevel                 = kmonitor::NORMAL;
        metricConfig.anetConfig.enableANetMetric = true;
        metricConfig.arpcConfig.enableArpcMetric = true;
        auto metricReporter = std::make_shared<arpc::KMonitorANetServerMetricReporter>(metricConfig);
        if (!metricReporter->init(rpc_server_transport_.get())) {
            RTP_LLM_LOG_WARNING("init anet metric reporter failed");
            return false;
        }
        rpc_server_->SetMetricReporter(metricReporter);
    }

    std::shared_ptr<autil::ThreadPoolBase> rpc_worker_threadpool(
        new autil::LockFreeThreadPool(3, 100, nullptr, "messager_server_rpc_threadpool", false));
    if (!rpc_worker_threadpool->start()) {
        RTP_LLM_LOG_WARNING("messager server init failed, start rpc worker threadpool failed");
        return false;
    }
    rpc_server_->RegisterService(rpc_service_.get(), rpc_worker_threadpool);

    std::string listen_spec = "tcp:0.0.0.0:" + std::to_string(listen_port);
    if (!rpc_server_->Listen(listen_spec)) {
        RTP_LLM_LOG_WARNING("messager server init failed, listen %s failed", listen_spec.c_str());
        return false;
    }

    RTP_LLM_LOG_INFO("messager server init success, listen on %s", listen_spec.c_str());
    return true;
}

}  // namespace rtp_llm