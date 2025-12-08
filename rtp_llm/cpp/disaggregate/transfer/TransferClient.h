#pragma once

#include "rtp_llm/cpp/disaggregate/transfer/TcpClient.h"
#include "rtp_llm/cpp/disaggregate/transfer/RdmaInterface.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerBlockConvertor.h"
#include "rtp_llm/cpp/disaggregate/transfer/proto/service.pb.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/Event.h"
#include "autil/NetUtil.h"
#include <memory>
#include <string>
#include <functional>

namespace rtp_llm {

/// @brief TransferClient 用于将 layer_cache_buffer 中的 buffer 内容传输到 TransferServer
/// 参考 CacheStoreServer 的使用方式
class TransferClient {
public:
    TransferClient(const std::shared_ptr<LayerBlockConvertor>& layer_block_convector, rtp_llm::DeviceBase* device):
        layer_block_convector_(layer_block_convector), device_(device) {}
    ~TransferClient() = default;

public:
    bool init(bool use_rdma, int tcp_io_thread_count, int rdma_io_thread_count, int rdma_worker_thread_count);

    virtual void transfer(const std::string&                       ip,
                          uint32_t                                 port,
                          const std::string&                       unique_key,
                          const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                          uint32_t                                 local_partition_count,
                          uint32_t                                 local_partition_id,
                          uint32_t                                 remote_partition_count,
                          uint32_t                                 remote_partition_id,
                          std::function<void(bool)>                callback,
                          int                                      timeout_ms = 1000);

    bool registerUserMr(const BufferPtr& buffer, uint64_t aligned_size);

private:
    /// @brief 构建传输请求
    /// @param layer_cache_buffer LayerCacheBuffer
    /// @param unique_key 唯一键
    /// @param local_partition_count 本地分区数量
    /// @param local_partition_id 本地分区ID
    /// @param remote_partition_count 远程分区数量
    /// @param remote_partition_id 远程分区ID
    /// @return 传输请求
    std::shared_ptr<::transfer::LayerBlockTransferRequest>
    makeTransferRequest(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                        const std::string&                       unique_key,
                        uint32_t                                 local_partition_count,
                        uint32_t                                 local_partition_id,
                        uint32_t                                 remote_partition_count,
                        uint32_t                                 remote_partition_id);

    /// @brief 设置 BlockBufferInfo
    /// @param block_buffer_info 要填充的 BlockBufferInfo
    /// @param key cache key
    /// @param buffer buffer 指针
    /// @return 是否成功
    bool setBlockBufferInfo(::transfer::BlockBufferInfo* block_buffer_info,
                            int64_t                      cache_key,
                            int                          block_id,
                            BufferPtr                    buffer);

    /// @brief 发送到远程服务器
    /// @param ip 目标服务器 IP
    /// @param port 目标服务器端口
    /// @param layer_cache_buffer LayerCacheBuffer
    /// @param transfer_request 传输请求
    /// @param callback 完成回调
    void loadToRemote(const std::string&                                            ip,
                      uint32_t                                                      port,
                      const std::shared_ptr<LayerCacheBuffer>&                      layer_cache_buffer,
                      const std::shared_ptr<::transfer::LayerBlockTransferRequest>& transfer_request,
                      std::function<void(bool)>                                     callback,
                      int                                                           timeout_ms);

    const std::shared_ptr<IRdmaMemoryManager>& getRdmaMemoryManager() const {
        return rdma_memory_manager_;
    }

private:
    std::shared_ptr<transfer::TcpClient> tcp_client_;
    std::shared_ptr<IRdmaServer>         rdma_server_;
    std::shared_ptr<LayerBlockConvertor> layer_block_convector_;
    rtp_llm::DeviceBase*                 device_;
    std::shared_ptr<IRdmaMemoryManager>  rdma_memory_manager_;

    // RDMA IP 和 port（用于填充 transfer request）
    std::string rdma_ip_;
    uint32_t    rdma_listen_port_ = 0;
};

}  // namespace rtp_llm
