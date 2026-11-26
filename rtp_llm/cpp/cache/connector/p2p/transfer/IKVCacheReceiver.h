#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/BlockInfo.h"

namespace rtp_llm {
namespace transfer {

/// @brief 单次 recv 操作的任务句柄（对应一层一个 partition 的接收）
class IKVCacheRecvTask {
public:
    virtual ~IKVCacheRecvTask() = default;

    /// @brief 是否已完成（成功或失败）
    virtual bool done() const = 0;

    /// @brief 是否成功（done() == true 时有效）
    virtual bool success() const = 0;

    // Zero means no successful physical completion timestamp is available.
    virtual int64_t physicalCompletionTimeUs() const { return 0; }

    /// @brief 主动取消该任务
    virtual void cancel() = 0;

    /// @brief 强制终止任务，无论当前状态如何（安全网，供超时兜底使用）
    /// 默认实现调用 cancel()，实现类可根据需要重写
    virtual void forceCancel() {
        cancel();
    }

    /// @brief 获取错误码（done() == true 时有效）
    virtual TransferErrorCode errorCode() const = 0;

    /// @brief 获取错误描述（done() == true 时有效）
    virtual std::string errorMessage() const = 0;

    /// @brief 安装任务完成回调，替换此前待执行的回调。若任务已结束则立即调用。
    virtual void setDoneCallback(std::function<void()> callback) = 0;
};

using IKVCacheRecvTaskPtr = std::shared_ptr<IKVCacheRecvTask>;

// Invoked synchronously under the receive task lock. Return true once the
// route's requested blocks are complete, or false to accept more batches.
// Do not block, throw, or call back into the task or the receiver task store.
using BatchCopiedCallback = std::function<bool(const std::vector<int64_t>&)>;

/// @brief 注册一层 KV cache 接收任务的请求参数
struct RecvRequest {
    /// Identifies the receive task and its route-wide target block table.
    std::string     unique_key;
    KeyBlockInfoMap block_info;
    int64_t         deadline_ms = 0;
    BatchCopiedCallback on_batch_copied;
};

using RecvRequestPtr = std::shared_ptr<RecvRequest>;

/// @brief KV Cache 接收接口（Decode 侧使用）
class IKVCacheReceiver {
public:
    virtual ~IKVCacheReceiver() = default;
    virtual void stopMetricsReporter() {}

    /// @brief 注册内存到 RDMA MR（建议在 recv 之前对所有 buffer 调用）
    virtual bool regMem(const BlockInfo& block_info, uint64_t aligned_size = 0) = 0;

    /// @brief 注册一次接收任务，返回任务句柄
    /// Prefill 端发起 transfer 后，该任务句柄会在数据到达后变为 done=true
    virtual IKVCacheRecvTaskPtr recv(const RecvRequest& request) = 0;

    /// @brief 从 task store 中移除并获取任务（转移所有权，供发送端确认时调用）
    virtual void stealTask(const std::string& unique_key) = 0;

    /// @brief 从 task store 中查询任务（不转移所有权）
    virtual IKVCacheRecvTaskPtr getTask(const std::string& unique_key) = 0;
};

using IKVCacheReceiverPtr = std::shared_ptr<IKVCacheReceiver>;

}  // namespace transfer
}  // namespace rtp_llm
