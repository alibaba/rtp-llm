#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "aios/network/arpc/arpc/metric/KMonitorANetClientMetricReporter.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

TcpClient::~TcpClient() {
    stop();
}

bool TcpClient::init(int io_thread_count, size_t queue_size) {
    if (queue_size == 0) {
        RTP_LLM_LOG_ERROR("cache-store client queue capacity must be positive");
        return false;
    }
    queue_size_ = queue_size;
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
    RTP_LLM_LOG_INFO("tcp client init success, io thread count %d", io_thread_count);
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

    std::lock_guard<std::mutex> lock(channel_map_mutex_);
    auto                        channel = channel_map_[spec];
    if (channel != nullptr && !channel->ChannelBroken()) {
        return channel;
    }

    auto new_channel = openChannel(spec);
    if (new_channel == nullptr || new_channel->ChannelBroken()) {
        return nullptr;
    }

    channel_map_[spec] = new_channel;
    RTP_LLM_LOG_INFO("tcp client new channel connect to %s", spec.c_str());
    return new_channel;
}

std::shared_ptr<arpc::RPCChannelBase> TcpClient::openChannel(const std::string& spec) {
    if (!rpc_channel_manager_) {
        RTP_LLM_LOG_WARNING("tcp client open channel to %s failed, rpc channel manager is null", spec.c_str());
        return nullptr;
    }

    // CacheStore workers are shared across peers. Never block one of those
    // workers behind a saturated channel: ANet reports enqueue failure through
    // the RPC callback, and the cache-load state machine applies its bounded
    // transfer retry policy without starving healthy peers.
    size_t queue_size = queue_size_;
#if USING_ROCM
    const char* split_ssm = std::getenv("ATOM_RTP_SPLIT_SSM_STORAGE");
    if (split_ssm != nullptr && std::strcmp(split_ssm, "1") == 0) {
        // Separately published SSM and convolution layers increase the number
        // of simultaneous PD load RPCs. Keep nonblocking enqueue semantics,
        // with a bounded queue sized for high-concurrency split-layout bursts.
        // C192 can enqueue over 20K per-layer loads before P publishes them;
        // 4096 rejects valid PD requests before the server can process them.
        queue_size = std::max(queue_size, size_t{32768});
    }
#endif
    RTP_LLM_LOG_INFO("cache-store channel %s nonblocking queue capacity %zu", spec.c_str(), queue_size);
    return std::shared_ptr<arpc::RPCChannelBase>(
        dynamic_cast<arpc::RPCChannelBase*>(rpc_channel_manager_->OpenChannel(spec, false, queue_size)));
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
