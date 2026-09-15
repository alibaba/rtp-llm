#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpClient.h"

#include "aios/network/arpc/arpc/metric/KMonitorANetClientMetricReporter.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace transfer {

TcpClient::~TcpClient() {
    stop();
}

bool TcpClient::init(int io_thread_count, std::chrono::milliseconds idle_ttl, std::uint64_t sweep_every_n_calls) {
    channel_idle_ttl_         = idle_ttl;
    sweep_every_n_calls_      = sweep_every_n_calls;
    get_channel_call_counter_ = 0;

    if (rpc_channel_transport_ == nullptr) {
        rpc_channel_transport_.reset(new anet::Transport(io_thread_count));
        if (!rpc_channel_transport_ || !rpc_channel_transport_->start()) {
            stop();
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
            stop();
            return false;
        }
        rpc_channel_manager_->SetMetricReporter(metricReporter);
    }
    RTP_LLM_LOG_INFO("tcp client init success, io thread count %d, idle_ttl_ms %lld, sweep_every_n_calls %llu",
                     io_thread_count,
                     static_cast<long long>(idle_ttl.count()),
                     static_cast<unsigned long long>(sweep_every_n_calls));
    return true;
}

void TcpClient::stop() {
    if (rpc_channel_manager_) {
        rpc_channel_transport_->stop();
        rpc_channel_transport_->wait();

        rpc_channel_manager_->Close();
        rpc_channel_manager_.reset();

        rpc_channel_transport_.reset();
    } else if (rpc_channel_transport_) {
        rpc_channel_transport_->stop();
        rpc_channel_transport_->wait();
        rpc_channel_transport_.reset();
    }
}

void TcpClient::evictIdleExpired(std::chrono::steady_clock::time_point now) {
    if (channel_idle_ttl_ <= std::chrono::steady_clock::duration::zero()) {
        return;
    }
    for (auto it = channel_map_.begin(); it != channel_map_.end();) {
        if (now - it->second.last_used >= channel_idle_ttl_) {
            it = channel_map_.erase(it);
        } else {
            ++it;
        }
    }
}

std::shared_ptr<arpc::RPCChannelBase> TcpClient::getChannel(const std::string& ip, uint32_t port) {
    std::string spec = "tcp:" + ip + ":" + std::to_string(port);

    std::lock_guard<std::mutex> lock(channel_map_mutex_);
    const auto                  now = std::chrono::steady_clock::now();
    ++get_channel_call_counter_;

    const bool ttl_on          = channel_idle_ttl_ > std::chrono::steady_clock::duration::zero();
    bool       swept_this_call = false;
    if (ttl_on && sweep_every_n_calls_ > 0 && (get_channel_call_counter_ % sweep_every_n_calls_) == 0) {
        evictIdleExpired(now);
        swept_this_call = true;
    }

    auto it = channel_map_.find(spec);
    if (it != channel_map_.end()) {
        ChannelCacheEntry& entry   = it->second;
        const auto&        channel = entry.channel;
        if (channel != nullptr && !channel->ChannelBroken()) {
            if (ttl_on && (now - entry.last_used >= channel_idle_ttl_)) {
                channel_map_.erase(it);
            } else {
                if (ttl_on) {
                    entry.last_used = now;
                }
                return channel;
            }
        } else {
            channel_map_.erase(it);
        }
    }

    if (ttl_on && !swept_this_call) {
        evictIdleExpired(now);
    }

    auto new_channel = openChannel(spec);
    if (new_channel == nullptr || new_channel->ChannelBroken()) {
        return nullptr;
    }

    channel_map_.emplace(spec, ChannelCacheEntry{new_channel, now});
    RTP_LLM_LOG_INFO("tcp client new channel connect to %s", spec.c_str());
    return new_channel;
}

std::shared_ptr<arpc::RPCChannelBase> TcpClient::openChannel(const std::string& spec) {
    if (!rpc_channel_manager_) {
        RTP_LLM_LOG_WARNING("tcp client open channel to %s failed, rpc channel manager is null", spec.c_str());
        return nullptr;
    }

    return std::shared_ptr<arpc::RPCChannelBase>(
        dynamic_cast<arpc::RPCChannelBase*>(rpc_channel_manager_->OpenChannel(spec, false, 1000ul)));
}

}  // namespace transfer
}  // namespace rtp_llm
