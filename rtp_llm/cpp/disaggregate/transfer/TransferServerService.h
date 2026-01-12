#pragma once

#include "rtp_llm/cpp/disaggregate/transfer/proto/service.pb.h"
#include "rtp_llm/cpp/disaggregate/transfer/RdmaInterface.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerBlockConvertor.h"
#include "rtp_llm/cpp/disaggregate/transfer/TransferTaskContext.h"
#include "rtp_llm/cpp/disaggregate/transfer/CudaCopyUtil.h"
#include "autil/LoopThread.h"
#include "autil/LockFreeThreadPool.h"

namespace rtp_llm {

/// @brief 接收 TransferClient 的请求，然后查找 LayerCacheBufferStore 中是否有对应的 LayerCacheBufferPtr
/// 并根据 request 是 tcp 还是 rdma 调用相应的处理逻辑
class TransferServerService: public ::transfer::TransferService {
public:
    TransferServerService(const std::shared_ptr<LayerCacheBufferTaskStore>& layer_cache_buffer_task_store,
                          const std::shared_ptr<LayerBlockConvertor>&       layer_block_convector,
                          const std::shared_ptr<IRdmaClient>&               rdma_client      = nullptr,
                          const kmonitor::MetricsReporterPtr&               metrics_reporter = nullptr);
    ~TransferServerService();

public:
    bool init(int64_t wait_check_interval_us = 1000, int worker_thread_count = 4);

    void transfer(::google::protobuf::RpcController*           controller,
                  const ::transfer::LayerBlockTransferRequest* request,
                  ::transfer::LayerBlockTransferResponse*      response,
                  ::google::protobuf::Closure*                 done) override;

private:
    /// @brief 基于 TCP 的 transfer 实现
    void transferViaTcp(const std::shared_ptr<TransferTaskContext>& transfer_task_context);

    /// @brief 基于 RDMA 的 transfer 实现
    void transferViaRdma(const std::shared_ptr<TransferTaskContext>& transfer_task_context);

    void waitCheckProc();

private:
    std::shared_ptr<LayerCacheBufferTaskStore> layer_cache_buffer_task_store_;
    std::shared_ptr<LayerBlockConvertor>       layer_block_convector_;
    std::shared_ptr<IRdmaClient>               rdma_client_;
    kmonitor::MetricsReporterPtr               metrics_reporter_;
    std::unique_ptr<CudaCopyUtil>              cuda_copy_util_;

    std::mutex                                      wait_tasks_mutex_;
    std::list<std::shared_ptr<TransferTaskContext>> wait_tasks_;
    autil::LoopThreadPtr                            wait_check_loop_thread_;

    std::shared_ptr<autil::LockFreeThreadPool> worker_thread_pool_;
};

}  // namespace rtp_llm
