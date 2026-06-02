#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/cache/BlockInfo.h"

namespace rtp_llm {
namespace transfer {

/// @brief 发送一层 KV cache 到远端 Decode 节点的请求参数
struct SendRequest {
    std::string ip;
    uint32_t    port = 0;
    /// unique_key = base_key + "_" + layer_id + "_" + remote_partition_id
    std::string     unique_key;
    KeyBlockInfoMap block_info;
    int64_t         deadline_ms = 0;
};

using SendRequestPtr = std::shared_ptr<SendRequest>;

class IKVCacheSender {
public:
    virtual ~IKVCacheSender() = default;

    /// @brief 注册内存到 RDMA MR（建议在 send 之前对所有 buffer 调用）
    virtual bool regMem(const BlockInfo& block_info, uint64_t aligned_size = 0) = 0;

    /// @brief 异步发送一层 KV cache 数据到指定远端节点
    /// @param request  发送请求，包含目标地址、唯一键和 block 地址信息
    /// @param callback 完成回调，(TransferErrorCode, error_msg)
    virtual void send(const SendRequest&                                         request,
                      std::function<void(TransferErrorCode, const std::string&)> callback) = 0;
};

using IKVCacheSenderPtr = std::shared_ptr<IKVCacheSender>;

}  // namespace transfer
}  // namespace rtp_llm
