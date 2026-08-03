#pragma once

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <string>
#include <map>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace transfer {

/// @brief 单次 recv 任务（一层一个 partition），实现 IKVCacheRecvTask
class TransferTask: public IKVCacheRecvTask {
public:
    TransferTask(KeyBlockInfoMap block_infos, int64_t deadline_ms):
        block_infos_(std::move(block_infos)), deadline_ms_(deadline_ms), start_time_us_(currentTimeUs()) {}
    ~TransferTask() override = default;

public:
    // IKVCacheRecvTask interface
    bool              done() const override;
    bool              success() const override;
    void              cancel() override;
    TransferErrorCode errorCode() const override;
    std::string       errorMessage() const override;

public:
    // 内部使用（TcpTransferService / RdmaTransferService 通知完成）
    const KeyBlockInfoMap& getBlockInfos() const {
        return block_infos_;
    }
    void
    notifyDone(bool success, TransferErrorCode error_code = TransferErrorCode::OK, const std::string& error_msg = "");

    /// @brief 原子地将任务从 PENDING 迁移到 TRANSFERRING 状态。
    /// @return false 表示任务已在 PENDING 阶段被 cancel，调用方应立即报告失败。
    bool startTransfer();

    /// @brief 强制终止任务，无论当前状态如何（仅供 P2PConnectorWorker 超时安全网使用）。
    void forceCancel() override;

    int64_t totalCostTimeUs() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return total_cost_time_us_;
    }

private:
    KeyBlockInfoMap block_infos_;
    int64_t         deadline_ms_;
    int64_t         start_time_us_      = 0;
    int64_t         total_cost_time_us_ = 0;

    mutable std::shared_mutex mutex_;
    bool                      done_             = false;
    bool                      transferring_     = false;
    bool                      cancel_requested_ = false;
    TransferErrorCode         error_code_       = TransferErrorCode::OK;
    std::string               error_msg_;
};

/// @brief 内部 task store，被 TcpKVCacheReceiver / RdmaKVCacheReceiver 私有持有
class TransferTaskStore {
public:
    TransferTaskStore()  = default;
    ~TransferTaskStore() = default;

    /// @brief 创建并注册一个新的 recv task
    std::shared_ptr<TransferTask>
    addTask(const std::string& unique_key, KeyBlockInfoMap block_infos, int64_t deadline_ms);

    /// @brief 按 unique_key 查询 task（不转移所有权）
    std::shared_ptr<TransferTask> getTask(const std::string& unique_key) const;
    /// @brief 按 unique_key 取出并移除 task（转移所有权）
    std::shared_ptr<TransferTask> stealTask(const std::string& unique_key);

    int64_t getTaskCount() const;

private:
    mutable std::shared_mutex                            mutex_;
    std::map<std::string, std::shared_ptr<TransferTask>> task_map_;
};

}  // namespace transfer
}  // namespace rtp_llm
