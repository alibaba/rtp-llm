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
                        const std::shared_ptr<IRdmaClient>& rdma_client,
                        const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr,
                        const std::shared_ptr<RdmaStagingBufferPool>& staging_pool = nullptr,
                        int64_t rdma_disconnect_after_deadline_ms = 10 * 1000);
    ~RdmaTransferService();

public:
    /// @brief 启动 wait-check 轮询线程和 worker 线程池
    /// @param wait_check_interval_us 轮询间隔(微秒)
    /// @param worker_thread_count 工作线程数
    /// @param per_connection_block_count 每个connection最多传输的block数，默认UINT64_MAX表示不限制
    bool init(int64_t wait_check_interval_us = 1000,
              int worker_thread_count = 16,
              uint64_t per_connection_block_count = UINT64_MAX,
              int worker_queue_size = 100);

    void transfer(::google::protobuf::RpcController*                    controller,
                  const ::rdma_transfer::RdmaLayerBlockTransferRequest* request,
                  ::rdma_transfer::RdmaLayerBlockTransferResponse*      response,
                  ::google::protobuf::Closure*                          done) override;

private:
    void transferViaRdma(const std::shared_ptr<RdmaTaskContext>& ctx);
    void waitCheckProc();
    void transferDeadlineCheckProc();

private:
    std::shared_ptr<TransferTaskStore> task_store_;
    std::shared_ptr<IRdmaClient> rdma_client_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    std::shared_ptr<RdmaStagingBufferPool> staging_pool_;
    std::shared_ptr<tcp::CudaCopyUtil> cuda_copy_util_;
    int64_t rdma_disconnect_after_deadline_ms_{10 * 1000};

    uint64_t                                    per_connection_block_count_{UINT64_MAX};

    std::mutex                                   wait_tasks_mutex_;
    std::list<std::shared_ptr<RdmaTaskContext>>  wait_tasks_;
    autil::LoopThreadPtr                         wait_check_loop_thread_;

    struct ActiveTransfer {
        std::weak_ptr<RdmaTaskContext> ctx;
        std::shared_ptr<IRdmaConnection> connection;
        uint64_t deadline_ms;
        std::shared_ptr<std::atomic<bool>> already_run;
        std::shared_ptr<std::atomic<bool>> physical_done;
        std::shared_ptr<std::atomic<bool>> disconnect_triggered;
        std::shared_ptr<std::atomic<size_t>> completed_count;
        size_t total_batches;
        int64_t posted_at_us;
    };
    std::mutex active_transfers_mutex_;
    std::list<ActiveTransfer> active_transfers_;
    autil::LoopThreadPtr transfer_deadline_loop_thread_;

    std::shared_ptr<autil::LockFreeThreadPool> worker_thread_pool_;
};

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
