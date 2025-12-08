#include "rtp_llm/cpp/disaggregate/transfer/TransferClient.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"
#include "autil/NetUtil.h"
#include <memory>

namespace rtp_llm {

// TransferClosure 用于处理异步 RPC 回调
class TransferClosure: public ::google::protobuf::Closure {
public:
    TransferClosure(const std::shared_ptr<LayerCacheBuffer>&                       layer_cache_buffer,
                    const std::shared_ptr<::transfer::LayerBlockTransferRequest>&  transfer_request,
                    const std::shared_ptr<::transfer::LayerBlockTransferResponse>& transfer_response,
                    arpc::ANetRPCController*                                       controller,
                    std::function<void(bool)>                                      callback):
        layer_cache_buffer_(layer_cache_buffer),
        transfer_request_(transfer_request),
        transfer_response_(transfer_response),
        controller_(controller),
        callback_(callback) {}

    ~TransferClosure() {
        if (controller_) {
            delete controller_;
        }
    }

    void Run() override {
        bool success = false;
        if (controller_->Failed()) {
            RTP_LLM_LOG_WARNING("transfer failed, error: %s", controller_->ErrorText().c_str());
        } else if (transfer_response_->success()) {
            success = true;
        } else {
            RTP_LLM_LOG_WARNING("transfer failed, info: %s", transfer_response_->info().c_str());
        }

        if (callback_) {
            callback_(success);
        }
    }

private:
    std::shared_ptr<LayerCacheBuffer>                       layer_cache_buffer_;
    std::shared_ptr<::transfer::LayerBlockTransferRequest>  transfer_request_;
    std::shared_ptr<::transfer::LayerBlockTransferResponse> transfer_response_;
    arpc::ANetRPCController*                                controller_;
    std::function<void(bool)>                               callback_;
};

bool TransferClient::init(bool use_rdma,
                          int  tcp_io_thread_count,
                          int  rdma_io_thread_count,
                          int  rdma_worker_thread_count) {
    RTP_LLM_LOG_INFO("transfer client init start");
    if (device_ == nullptr || layer_block_convector_ == nullptr) {
        RTP_LLM_LOG_WARNING("device, layer block convector is nullptr");
        return false;
    }

    tcp_client_ = std::make_shared<transfer::TcpClient>();
    if (!tcp_client_->init(tcp_io_thread_count)) {
        RTP_LLM_LOG_WARNING("create tcp client failed");
        return false;
    }
    RTP_LLM_LOG_INFO("create tcp client success");

    if (use_rdma) {
        rdma_memory_manager_ = createRdmaMemoryManager();
        if (!rdma_memory_manager_) {
            RTP_LLM_LOG_WARNING("create rdma memory manager failed");
            return false;
        }
        rdma_listen_port_ = autil::NetUtil::randomPort();
        rdma_ip_          = autil::NetUtil::getBindIp();
        rdma_server_ =
            createRdmaServer(rdma_memory_manager_, rdma_listen_port_, rdma_io_thread_count, rdma_worker_thread_count);
        if (rdma_server_ == nullptr) {
            RTP_LLM_LOG_WARNING("create rdma server failed");
            return false;
        }
        RTP_LLM_LOG_INFO("create rdma server success, listen port: %d", rdma_listen_port_);
    }

    RTP_LLM_LOG_INFO("transfer client init success, use %s mode", use_rdma ? "rdma" : "tcp");
    return true;
}

std::shared_ptr<::transfer::LayerBlockTransferRequest>
TransferClient::makeTransferRequest(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                                    const std::string&                       unique_key,
                                    uint32_t                                 local_partition_count,
                                    uint32_t                                 local_partition_id,
                                    uint32_t                                 remote_partition_count,
                                    uint32_t                                 remote_partition_id) {
    auto transfer_request = std::make_shared<::transfer::LayerBlockTransferRequest>();
    transfer_request->set_unique_key(unique_key);
    transfer_request->set_partition_count(remote_partition_count);
    transfer_request->set_partition_id(remote_partition_id);
    transfer_request->set_server_rdma_ip(rdma_ip_);
    transfer_request->set_server_rdma_port(rdma_listen_port_);

    auto layer_id = layer_cache_buffer->getLayerId();

    auto layer_block_info = transfer_request->mutable_layer_block();
    layer_block_info->set_layer_id(layer_cache_buffer->getLayerId());

    // 遍历 layer_cache_buffer 中的所有 block_id
    for (const auto& [cache_key, block_id] : layer_cache_buffer->blockIdMap()) {
        auto cache_key_block_info = layer_block_info->add_blocks();
        cache_key_block_info->set_key(cache_key);

        auto buffers =
            layer_block_convector_->convertIndexToBuffer(layer_id, block_id, local_partition_count, local_partition_id);
        if (buffers.empty()) {
            RTP_LLM_LOG_WARNING("convert index to buffer failed, layer id: %d, block id: %d", layer_id, block_id);
            return nullptr;
        }

        for (auto& buffer : buffers) {
            auto block_buffer_info = cache_key_block_info->add_blocks();
            if (!setBlockBufferInfo(block_buffer_info, cache_key, block_id, buffer)) {
                RTP_LLM_LOG_WARNING("set block buffer info failed, layer id: %d, block id: %d, key: %lld",
                                    layer_cache_buffer->getLayerId(),
                                    block_id,
                                    cache_key);
                return nullptr;
            }
        }
    }
    // RTP_LLM_LOG_INFO("make transfer request success, unique_key: %s, layer_id: %d, request: %s",
    //                  unique_key.c_str(),
    //                  layer_cache_buffer->getLayerId(),
    //                  transfer_request->ShortDebugString().c_str());
    return transfer_request;
}

bool TransferClient::setBlockBufferInfo(::transfer::BlockBufferInfo* block_buffer_info,
                                        int64_t                      cache_key,
                                        int                          block_id,
                                        BufferPtr                    buffer) {
    block_buffer_info->set_len(buffer->size());

    // 如果存在 rdmaMemoryManager，尝试获取 remoteBuffer 并填充 rdma_info
    if (rdma_memory_manager_) {
        auto remote_buffer = rdma_memory_manager_->findMemoryMr(buffer);
        if (!remote_buffer || remote_buffer->nic_rkeys == nullptr) {
            RTP_LLM_LOG_WARNING("get remote buffer failed, cache key: %lld, block id: %d, buffer size: %lu",
                                cache_key,
                                block_id,
                                buffer->size());
            return false;
        }

        // 填充 rdma_info
        auto rdma_info = block_buffer_info->mutable_rdma_info();
        rdma_info->set_addr(static_cast<uint64_t>(remote_buffer->addr));

        // 填充 nic_rkeys
        for (const auto& [nicid, rkey] : *remote_buffer->nic_rkeys) {
            auto nic_rkey = rdma_info->add_nic_rkeys();
            nic_rkey->set_nicid(nicid);
            nic_rkey->set_rkey(rkey);
        }

        // RDMA 模式下不填充 content
        return true;
    }

    // TCP 模式：填充 content
    auto tmp_buffer = static_cast<char*>(malloc(buffer->size()));
    auto dst_buffer =
        rtp_llm::Buffer(rtp_llm::MemoryType::MEMORY_CPU, rtp_llm::DataType::TYPE_UINT8, {buffer->size()}, tmp_buffer);

    device_->noBlockCopy({dst_buffer, *buffer});
    block_buffer_info->set_content(tmp_buffer, buffer->size());
    free(tmp_buffer);
    return true;
}

void TransferClient::loadToRemote(const std::string&                                            ip,
                                  uint32_t                                                      port,
                                  const std::shared_ptr<LayerCacheBuffer>&                      layer_cache_buffer,
                                  const std::shared_ptr<::transfer::LayerBlockTransferRequest>& transfer_request,
                                  std::function<void(bool)>                                     callback,
                                  int                                                           timeout_ms) {
    // 获取 TCP channel
    auto channel = tcp_client_->getChannel(ip, port);
    if (channel == nullptr) {
        RTP_LLM_LOG_WARNING("get channel failed, ip: %s, port: %d", ip.c_str(), port);
        if (callback) {
            callback(false);
        }
        return;
    }

    auto transfer_response = std::make_shared<::transfer::LayerBlockTransferResponse>();
    auto controller        = new arpc::ANetRPCController();

    controller->SetExpireTime(timeout_ms);

    auto closure = new TransferClosure(layer_cache_buffer, transfer_request, transfer_response, controller, callback);

    ::transfer::TransferService_Stub stub((::google::protobuf::RpcChannel*)(channel.get()),
                                          ::google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
    stub.transfer(controller, transfer_request.get(), transfer_response.get(), closure);

    RTP_LLM_LOG_INFO("load to remote success, ip: %s, port: %d, unique_key: %s, layer_id: %d",
                     ip.c_str(),
                     port,
                     transfer_request->unique_key().c_str(),
                     layer_cache_buffer->getLayerId());
}

void TransferClient::transfer(const std::string&                       ip,
                              uint32_t                                 port,
                              const std::string&                       unique_key,
                              const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                              uint32_t                                 local_partition_count,
                              uint32_t                                 local_partition_id,
                              uint32_t                                 remote_partition_count,
                              uint32_t                                 remote_partition_id,
                              std::function<void(bool)>                callback,
                              int                                      timeout_ms) {
    // 构建传输请求
    auto transfer_request = makeTransferRequest(layer_cache_buffer,
                                                unique_key,
                                                local_partition_count,
                                                local_partition_id,
                                                remote_partition_count,
                                                remote_partition_id);
    if (transfer_request == nullptr) {
        RTP_LLM_LOG_WARNING("make transfer request failed, layer id: %d", layer_cache_buffer->getLayerId());
        if (callback) {
            callback(false);
        }
        return;
    }

    RTP_LLM_LOG_INFO("transfer request success, ip: %s, port: %d, unique_key: %s, layer_id: %d",
                     ip.c_str(),
                     port,
                     transfer_request->unique_key().c_str(),
                     layer_cache_buffer->getLayerId());

    // 发送到远程服务器
    loadToRemote(ip, port, layer_cache_buffer, transfer_request, callback, timeout_ms);
}

bool TransferClient::registerUserMr(const BufferPtr& buffer, uint64_t aligned_size) {
    if (rdma_memory_manager_ == nullptr) {
        return true;
    }
    return rdma_memory_manager_->regUserMr(buffer, aligned_size);
}

}  // namespace rtp_llm
