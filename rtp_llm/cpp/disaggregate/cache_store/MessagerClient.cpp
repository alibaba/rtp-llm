#include "rtp_llm/cpp/disaggregate/cache_store/MessagerClient.h"

#include "rtp_llm/cpp/disaggregate/cache_store/CacheLoadServiceClosure.h"

#include "rtp_llm/cpp/utils/Logger.h"

#include "aios/network/arpc/arpc/ANetRPCController.h"
#include "aios/network/arpc/arpc/metric/KMonitorANetClientMetricReporter.h"

#include "autil/NetUtil.h"

namespace rtp_llm {

MessagerClient::MessagerClient(const std::shared_ptr<MemoryUtil>& memory_util): memory_util_(memory_util) {}

MessagerClient::~MessagerClient() {
    stopTcpClient();
}

bool MessagerClient::init(bool enable_metric) {
    if (!initTcpClient(enable_metric)) {
        RTP_LLM_LOG_WARNING("messager client init failed, tcp client init failed");
        return false;
    }
    return true;
}

bool MessagerClient::initTcpClient(bool enable_metric) {
    if (rpc_channel_transport_ == nullptr) {
        int tcp_client_io_thread_count = memory_util_->isRdmaMode() ? 1 : 3;
        rpc_channel_transport_.reset(new anet::Transport(tcp_client_io_thread_count));
        if (!rpc_channel_transport_ || !rpc_channel_transport_->start()) {
            return false;
        }
        rpc_channel_transport_->setName("MessagerClientRPCChannel");
    }

    rpc_channel_manager_.reset(new arpc::ANetRPCChannelManager(rpc_channel_transport_.get()));
    if (enable_metric) {
        arpc::KMonitorANetMetricReporterConfig metricConfig;
        metricConfig.arpcConfig.enableArpcMetric = true;
        metricConfig.anetConfig.enableANetMetric = false;
        metricConfig.metricLevel                 = kmonitor::NORMAL;
        auto metricReporter = std::make_shared<arpc::KMonitorANetClientMetricReporter>(metricConfig);
        if (!metricReporter->init(rpc_channel_transport_.get())) {
            RTP_LLM_LOG_ERROR("anet metric reporter init failed");
            return false;
        }
        rpc_channel_manager_->SetMetricReporter(metricReporter);
    }
    RTP_LLM_LOG_INFO("layer cache messager client init tcp client success");
    return true;
}

void MessagerClient::stopTcpClient() {
    if (rpc_channel_manager_) {
        rpc_channel_transport_->stop();
        rpc_channel_transport_->wait();

        rpc_channel_manager_->Close();
        rpc_channel_manager_.reset();

        rpc_channel_transport_.reset();
    }
}

void MessagerClient::load(const std::string&                                           ip,
                          uint32_t                                                     port,
                          uint32_t                                                     rdma_port,
                          const std::shared_ptr<RequestBlockBuffer>&                   request_block_buffer,
                          CacheStoreLoadDoneCallback                                   callback,
                          uint32_t                                                     timeout_ms,
                          const std::shared_ptr<CacheStoreClientLoadMetricsCollector>& collector,
                          int                                                          partition_count,
                          int                                                          partition_id) {
    auto channel = getChannel(ip, port);
    if (channel == nullptr) {
        RTP_LLM_LOG_WARNING("messager client get channel failed, ip %s", ip.c_str());
        callback(false, CacheStoreErrorCode::LoadConnectFailed);
        return;
    }

    auto request = makeLoadRequest(
        request_block_buffer, timeout_ms - 10, partition_count, partition_id);  // TODO: 10 is message transfer time
    if (request == nullptr) {
        RTP_LLM_LOG_WARNING("messager client generate load request failed");
        callback(false, CacheStoreErrorCode::LoadSendRequestFailed);
        return;
    }

    arpc::ANetRPCController* controller = new arpc::ANetRPCController();
    controller->SetExpireTime(timeout_ms);

    CacheLoadResponse*       response = new CacheLoadResponse;
    CacheLoadServiceClosure* closure  = new CacheLoadServiceClosure(
        memory_util_, request_block_buffer, controller, request, response, callback, collector);

    KvCacheStoreService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                  ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    stub.load(controller, request, response, closure);
}

CacheLoadRequest* MessagerClient::makeLoadRequest(const std::shared_ptr<RequestBlockBuffer>& request_block_buffer,
                                                  uint32_t                                   timeout_ms,
                                                  int                                        partition_count,
                                                  int                                        partition_id) {
    auto blocks = request_block_buffer->getBlocks();

    auto request = new CacheLoadRequest;
    for (auto& [key, block] : blocks) {
        auto block_msg = request->add_blocks();
        block_msg->set_key(block->key);
        block_msg->set_len(block->len);
    }
    request->set_timeout_ms(timeout_ms);
    request->set_requestid(request_block_buffer->getRequestId());
    request->set_client_ip(autil::NetUtil::getBindIp());
    request->set_request_send_start_time_us(autil::TimeUtility::currentTimeInMicroSeconds());
    request->set_partition_count(partition_count);
    request->set_partition_id(partition_id);
    return request;
}

std::shared_ptr<arpc::RPCChannelBase> MessagerClient::getChannel(const std::string& ip, uint32_t port) {
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

    RTP_LLM_LOG_INFO("new channel connect to %s", spec.c_str());
    channel_map_[spec] = new_channel;
    return new_channel;
}

std::shared_ptr<arpc::RPCChannelBase> MessagerClient::openChannel(const std::string& spec) {
    if (!rpc_channel_manager_) {
        RTP_LLM_LOG_WARNING("messager client open channel failed, rpc channel manager is null");
        return nullptr;
    }

    return std::shared_ptr<arpc::RPCChannelBase>(
        dynamic_cast<arpc::RPCChannelBase*>(rpc_channel_manager_->OpenChannel(spec, false, 1000ul)));
}

}  // namespace rtp_llm
