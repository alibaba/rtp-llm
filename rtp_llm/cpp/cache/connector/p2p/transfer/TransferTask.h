#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <map>
#include <vector>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace transfer {

/// @brief 单次 recv 任务（一层一个 partition），实现 IKVCacheRecvTask
class TransferTask: public IKVCacheRecvTask {
public:
    TransferTask(KeyBlockInfoMap block_infos, int64_t deadline_ms, BatchCopiedCallback on_batch_copied = {}):
        block_infos_(std::move(block_infos)),
        on_batch_copied_(std::move(on_batch_copied)),
        deadline_ms_(deadline_ms),
        start_time_us_(currentTimeUs()) {}
    ~TransferTask() override = default;

public:
    // IKVCacheRecvTask interface
    bool              done() const override;
    bool              success() const override;
    void              cancel() override;
    TransferErrorCode errorCode() const override;
    std::string       errorMessage() const override;
    void              setDoneCallback(std::function<void()> callback) override;

public:
    // 内部使用（TcpTransferService / RdmaTransferService 通知完成）
    const KeyBlockInfoMap& getBlockInfos() const {
        return block_infos_;
    }
    void    notifyDone(bool               success,
                       TransferErrorCode  error_code = TransferErrorCode::OK,
                       const std::string& error_msg  = "",
                       bool               started    = true);
    void    notifyBatchDone(const std::vector<int64_t>& keys, bool started = true);
    bool acceptsBatches() const { return static_cast<bool>(on_batch_copied_); }
    int64_t deadlineMs() const { return deadline_ms_; }

    /// Record the first logical failure while outstanding DMA still owns the buffers.
    void recordError(TransferErrorCode error_code, const std::string& error_message);

    /// Acquire one physical receive slot. Call completion once for each acquired slot.
    bool startTransfer();

    // Observe cancellation while waiting for staging, before taking a physical receive slot.
    size_t addCancelCallback(std::function<void()> callback);
    void removeCancelCallback(size_t id);

    /// @brief 强制终止任务，无论当前状态如何（仅供 P2P connector worker 超时安全网使用）。
    void forceCancel() override;

    int64_t totalCostTimeUs() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return total_cost_time_us_;
    }

private:
    void completeTransfer(TransferErrorCode           error_code,
                          const std::string&          error_msg,
                          const std::vector<int64_t>* batch_keys,
                          bool                        started);

    KeyBlockInfoMap block_infos_;
    BatchCopiedCallback on_batch_copied_;
    int64_t         deadline_ms_;
    int64_t         start_time_us_      = 0;
    int64_t         total_cost_time_us_ = 0;

    mutable std::shared_mutex mutex_;
    bool                      done_             = false;
    size_t                    in_flight_            = 0;
    bool                      completion_requested_ = false;
    bool                      cancel_requested_ = false;
    mutable TransferErrorCode error_code_       = TransferErrorCode::OK;
    mutable std::string       error_msg_;
    std::function<void()>     done_callback_;
    std::map<size_t, std::function<void()>> cancel_callbacks_;
    size_t next_cancel_callback_id_{1};
};

/// @brief 内部 task store，被 TcpKVCacheReceiver / RdmaKVCacheReceiver 私有持有
class TransferTaskStore {
public:
    TransferTaskStore()  = default;
    ~TransferTaskStore() = default;

    /// @brief 创建并注册一个新的 recv task
    std::shared_ptr<TransferTask>
    addTask(const std::string& unique_key,
            KeyBlockInfoMap block_infos,
            int64_t deadline_ms,
            BatchCopiedCallback on_batch_copied = {});

    /// @brief 按 unique_key 查询 task（不转移所有权）
    std::shared_ptr<TransferTask> getTask(const std::string& unique_key) const;
    /// @brief 按 unique_key 取出并移除 task（转移所有权）
    std::shared_ptr<TransferTask> stealTask(const std::string& unique_key);

    int64_t getTaskCount() const;
    void setTaskAddedCallback(std::function<void()> callback);

private:
    mutable std::shared_mutex                            mutex_;
    std::map<std::string, std::shared_ptr<TransferTask>> task_map_;
    std::map<std::string, int64_t> completed_batches_;
    std::multimap<int64_t, std::string> completed_expiry_;
    std::function<void()> task_added_callback_;
};

}  // namespace transfer
}  // namespace rtp_llm
