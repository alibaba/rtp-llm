#pragma once

#include "rtp_llm/cpp/disaggregate/transfer/proto/service.pb.h"
#include "rtp_llm/cpp/disaggregate/transfer/RdmaInterface.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerBlockConvertor.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

namespace rtp_llm {

/// @brief 接收 TransferClient 的请求，然后查找 LayerCacheBufferStore 中是否有对应的 LayerCacheBufferPtr
/// 并根据 request 是 tcp 还是 rdma 调用相应的处理逻辑
class TransferServerService: public ::transfer::TransferService {
public:
    TransferServerService(const std::shared_ptr<LayerCacheBufferTaskStore>& layer_cache_buffer_task_store,
                          const std::shared_ptr<LayerBlockConvertor>&       layer_block_convector,
                          rtp_llm::DeviceBase*                              device,
                          const std::shared_ptr<IRdmaClient>&               rdma_client = nullptr);
    ~TransferServerService();

public:
    void transfer(::google::protobuf::RpcController*           controller,
                  const ::transfer::LayerBlockTransferRequest* request,
                  ::transfer::LayerBlockTransferResponse*      response,
                  ::google::protobuf::Closure*                 done) override;

private:
    /// @brief 基于 TCP 的 transfer 实现
    void transferViaTcp(::google::protobuf::RpcController*           controller,
                        const ::transfer::LayerBlockTransferRequest* request,
                        ::transfer::LayerBlockTransferResponse*      response,
                        ::google::protobuf::Closure*                 done,
                        const std::shared_ptr<LayerCacheBuffer>&     layer_cache_buffer,
                        const std::shared_ptr<LayerCacheBufferTask>& task);

    /// @brief 基于 RDMA 的 transfer 实现
    void transferViaRdma(::google::protobuf::RpcController*           controller,
                         const ::transfer::LayerBlockTransferRequest* request,
                         ::transfer::LayerBlockTransferResponse*      response,
                         ::google::protobuf::Closure*                 done,
                         const std::shared_ptr<LayerCacheBuffer>&     layer_cache_buffer,
                         const std::shared_ptr<LayerCacheBufferTask>& task);

private:
    std::shared_ptr<LayerCacheBufferTaskStore> layer_cache_buffer_task_store_;
    std::shared_ptr<LayerBlockConvertor>       layer_block_convector_;
    rtp_llm::DeviceBase*                       device_;
    std::shared_ptr<IRdmaClient>               rdma_client_;
};

}  // namespace rtp_llm
