#pragma once

#include <cstdint>
#include <memory>
#include <string>
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
};

using IKVCacheRecvTaskPtr = std::shared_ptr<IKVCacheRecvTask>;

/// @brief 注册一层 KV cache 接收任务的请求参数
struct RecvRequest {
    /// unique_key = base_key + "_" + layer_id + "_" + remote_partition_id
    std::string     unique_key;
    KeyBlockInfoMap block_info;
    int64_t         deadline_ms = 0;
};

using RecvRequestPtr = std::shared_ptr<RecvRequest>;

/// @brief KV Cache 接收接口（Decode 侧使用）
class IKVCacheReceiver {
public:
    virtual ~IKVCacheReceiver() = default;

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
