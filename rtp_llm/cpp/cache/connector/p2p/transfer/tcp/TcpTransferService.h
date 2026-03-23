#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/CudaCopyUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpTaskContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/proto/tcp_service.pb.h"
#include "autil/LoopThread.h"
#include "autil/LockFreeThreadPool.h"
#include <list>
#include <mutex>
#include <memory>

namespace rtp_llm {
namespace transfer {
namespace tcp {

/// @brief Decode-side RPC handler: matches incoming TCP transfer RPCs with pre-registered recv tasks,
///        then copies the content bytes to device memory.
class TcpTransferService: public ::tcp_transfer::TcpTransferService {
public:
    TcpTransferService(const std::shared_ptr<TransferTaskStore>& task_store,
                       const kmonitor::MetricsReporterPtr&       metrics_reporter = nullptr);
    ~TcpTransferService();

public:
    /// @brief 启动 wait-check 轮询线程和 worker 线程池
    bool init(int64_t wait_check_interval_us = 1000, int worker_thread_count = 4);

    void transfer(::google::protobuf::RpcController*                  controller,
                  const ::tcp_transfer::TcpLayerBlockTransferRequest* request,
                  ::tcp_transfer::TcpLayerBlockTransferResponse*      response,
                  ::google::protobuf::Closure*                        done) override;

    std::shared_ptr<autil::ThreadPoolBase> getWorkerThreadPool() const {
        return worker_thread_pool_;
    }

private:
    void transferViaTcp(const std::shared_ptr<TcpTaskContext>& ctx);
    void waitCheckProc();

private:
    std::shared_ptr<TransferTaskStore> task_store_;
    kmonitor::MetricsReporterPtr       metrics_reporter_;
    std::unique_ptr<CudaCopyUtil>      cuda_copy_util_;

    std::mutex                                 wait_tasks_mutex_;
    std::list<std::shared_ptr<TcpTaskContext>> wait_tasks_;
    autil::LoopThreadPtr                       wait_check_loop_thread_;

    std::shared_ptr<autil::LockFreeThreadPool> worker_thread_pool_;
};

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
