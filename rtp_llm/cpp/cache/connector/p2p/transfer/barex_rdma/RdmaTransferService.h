#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaTaskContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/proto/rdma_service.pb.h"
#include "autil/LoopThread.h"
#include "autil/LockFreeThreadPool.h"
#include <list>
#include <mutex>
#include <memory>
#include <atomic>
#include <cstdint>
#include <algorithm>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

/// @brief Decode-side RPC handler: matches incoming RDMA transfer RPCs with pre-registered recv tasks,
///        then issues RDMA Read from the Prefill-side RDMA server.
class RdmaTransferService : public ::rdma_transfer::RdmaTransferService {
public:
    RdmaTransferService(const std::shared_ptr<TransferTaskStore>& task_store,
                        const std::shared_ptr<IRdmaClient>&       rdma_client,
                        const kmonitor::MetricsReporterPtr&       metrics_reporter = nullptr);
    ~RdmaTransferService();

public:
    /// @brief 启动 wait-check 轮询线程和 worker 线程池
    /// @param wait_check_interval_us 轮询间隔(微秒)
    /// @param worker_thread_count 工作线程数
    /// @param per_connection_block_count 每个connection最多传输的block数，默认UINT64_MAX表示不限制
    bool init(int64_t wait_check_interval_us = 1000,
              int worker_thread_count = 4,
              uint64_t per_connection_block_count = UINT64_MAX);

    void transfer(::google::protobuf::RpcController*                    controller,
                  const ::rdma_transfer::RdmaLayerBlockTransferRequest* request,
                  ::rdma_transfer::RdmaLayerBlockTransferResponse*      response,
                  ::google::protobuf::Closure*                          done) override;

private:
    void transferViaRdma(const std::shared_ptr<RdmaTaskContext>& ctx);
    void waitCheckProc();

private:
    std::shared_ptr<TransferTaskStore> task_store_;
    std::shared_ptr<IRdmaClient>       rdma_client_;
    kmonitor::MetricsReporterPtr       metrics_reporter_;

    uint64_t                                    per_connection_block_count_{UINT64_MAX};

    std::mutex                                   wait_tasks_mutex_;
    std::list<std::shared_ptr<RdmaTaskContext>>  wait_tasks_;
    autil::LoopThreadPtr                         wait_check_loop_thread_;

    std::shared_ptr<autil::LockFreeThreadPool> worker_thread_pool_;
};

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
