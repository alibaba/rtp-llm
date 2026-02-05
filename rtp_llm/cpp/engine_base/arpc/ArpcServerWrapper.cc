#include "rtp_llm/cpp/engine_base/arpc/ArpcServerWrapper.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "aios/network/arpc/arpc/metric/KMonitorANetMetricReporterConfig.h"
#include "aios/network/arpc/arpc/metric/KMonitorANetServerMetricReporter.h"

namespace rtp_llm {

void ArpcServerWrapper::start() {
    RTP_LLM_LOG_INFO("start arpc server with thread=%d, queue=%d, ioThreadNum=%d", threadNum_, queueNum_, ioThreadNum_);
    arpc_server_transport_.reset(new anet::Transport(ioThreadNum_, anet::SHARE_THREAD));
    arpc_server_.reset(new arpc::ANetRPCServer(arpc_server_transport_.get(), threadNum_, queueNum_));
    RTP_LLM_CHECK_WITH_INFO(arpc_server_transport_->start(), "arpc server start public transport failed");
    arpc_server_transport_->setName("ARPC SERVER");
    std::string spec("tcp:0.0.0.0:" + std::to_string(port_));
    RTP_LLM_CHECK_WITH_INFO(arpc_server_->Listen(spec), "arpc listen on %s failed", spec.c_str());
    // set metric reporter
    arpc::KMonitorANetMetricReporterConfig metricConfig;
    metricConfig.metricLevel                 = kmonitor::FATAL;
    metricConfig.anetConfig.enableANetMetric = false;
    metricConfig.arpcConfig.enableArpcMetric = true;
    auto metricReporter                      = std::make_shared<arpc::KMonitorANetServerMetricReporter>(metricConfig);
    if (!metricReporter->init(arpc_server_transport_.get())) {
        RTP_LLM_LOG_ERROR("init anet metric reporter failed");
        return;
    }
    arpc_server_->SetMetricReporter(metricReporter);

    arpc_server_->RegisterService(service_.get());
    RTP_LLM_LOG_INFO("ARPC Server listening on %s", spec.c_str());
}

void ArpcServerWrapper::stop() {
    if (arpc_server_) {
        arpc_server_->Close();
        arpc_server_->StopPrivateTransport();
    }
    if (arpc_server_transport_) {
        RTP_LLM_CHECK_WITH_INFO(arpc_server_transport_->stop(), "transport stop failed");
        RTP_LLM_CHECK_WITH_INFO(arpc_server_transport_->wait(), "transport wait failed");
    }
    RTP_LLM_LOG_INFO("ARPC Server stopped");
}

}  // namespace rtp_llm