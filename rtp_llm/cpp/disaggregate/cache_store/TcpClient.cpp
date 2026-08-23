#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"

#include "autil/EnvUtil.h"
#include "autil/TimeUtility.h"
#include "aios/network/arpc/arpc/metric/KMonitorANetClientMetricReporter.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

TcpClient::~TcpClient() {
    stop();
}

bool TcpClient::init(int io_thread_count) {
    if (rpc_channel_transport_ == nullptr) {
        rpc_channel_transport_.reset(new anet::Transport(io_thread_count));
        if (!rpc_channel_transport_ || !rpc_channel_transport_->start()) {
            return false;
        }
        rpc_channel_transport_->setName("MessagerClientRPCChannel");
    }

    rpc_channel_manager_.reset(new arpc::ANetRPCChannelManager(rpc_channel_transport_.get()));
    {
        arpc::KMonitorANetMetricReporterConfig metricConfig;
        metricConfig.arpcConfig.enableArpcMetric = true;
        metricConfig.anetConfig.enableANetMetric = false;
        metricConfig.metricLevel                 = kmonitor::FATAL;
        auto metricReporter = std::make_shared<arpc::KMonitorANetClientMetricReporter>(metricConfig);
        if (!metricReporter->init(rpc_channel_transport_.get())) {
            RTP_LLM_LOG_ERROR("anet metric reporter init failed");
            return false;
        }
        rpc_channel_manager_->SetMetricReporter(metricReporter);
    }
    channel_idle_timeout_ms_ = autil::EnvUtil::getEnv("CACHE_STORE_TCP_CHANNEL_IDLE_TIMEOUT_MS", int64_t(60000));
    RTP_LLM_LOG_INFO("tcp client init success, io thread count %d, channel idle timeout %ld ms",
                     io_thread_count,
                     channel_idle_timeout_ms_);
    return true;
}

void TcpClient::stop() {
    if (rpc_channel_manager_) {
        rpc_channel_transport_->stop();
        rpc_channel_transport_->wait();

        rpc_channel_manager_->Close();
        rpc_channel_manager_.reset();

        rpc_channel_transport_.reset();
    }
}

std::shared_ptr<arpc::RPCChannelBase> TcpClient::getChannel(const std::string& ip, uint32_t port) {
    std::string spec = "tcp:" + ip + ":" + std::to_string(port);
    int64_t     now  = autil::TimeUtility::currentTimeInMilliSeconds();

    std::lock_guard<std::mutex> lock(channel_map_mutex_);
    auto&                       entry = channel_map_[spec];
    if (entry.channel != nullptr && !entry.channel->ChannelBroken()
        && (channel_idle_timeout_ms_ <= 0 || now - entry.last_use_ms < channel_idle_timeout_ms_)) {
        entry.last_use_ms = now;
        return entry.channel;
    }

    if (entry.channel != nullptr) {
        RTP_LLM_LOG_WARNING("tcp client drop cached channel to %s, broken %d, idle %ld ms",
                            spec.c_str(),
                            entry.channel->ChannelBroken(),
                            now - entry.last_use_ms);
    }
    auto new_channel = openChannel(spec);
    if (new_channel == nullptr || new_channel->ChannelBroken()) {
        entry.channel = nullptr;
        return nullptr;
    }

    entry.channel     = new_channel;
    entry.last_use_ms = now;
    RTP_LLM_LOG_INFO("tcp client new channel connect to %s", spec.c_str());
    return new_channel;
}

void TcpClient::invalidateChannel(const std::string& ip, uint32_t port) {
    std::string spec = "tcp:" + ip + ":" + std::to_string(port);

    std::lock_guard<std::mutex> lock(channel_map_mutex_);
    auto                        it = channel_map_.find(spec);
    if (it == channel_map_.end()) {
        return;
    }
    // erase from cache, the channel is released after in-flight references drop,
    // next getChannel() will establish a fresh connection
    channel_map_.erase(it);
    RTP_LLM_LOG_WARNING("tcp client invalidate channel %s, will reconnect on next request", spec.c_str());
}

std::shared_ptr<arpc::RPCChannelBase> TcpClient::openChannel(const std::string& spec) {
    if (!rpc_channel_manager_) {
        RTP_LLM_LOG_WARNING("tcp client open channel to %s failed, rpc channel manager is null", spec.c_str());
        return nullptr;
    }

    return std::shared_ptr<arpc::RPCChannelBase>(
        dynamic_cast<arpc::RPCChannelBase*>(rpc_channel_manager_->OpenChannel(spec, false, 1000ul)));
}

std::shared_ptr<TransferConnection>
TcpClient::getTransferConnection(const std::string& ip, uint32_t port, int device_id) {
    auto channel = getChannel(ip, port);
    if (channel == nullptr) {
        return nullptr;
    }
    return std::make_shared<TcpTransferConnection>(channel, device_id);
}

}  // namespace rtp_llm