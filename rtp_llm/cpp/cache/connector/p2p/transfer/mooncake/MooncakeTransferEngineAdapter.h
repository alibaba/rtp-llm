#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferBackendConfig.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"

namespace rtp_llm {
namespace transfer {
namespace mooncake {

inline constexpr uint64_t kInvalidMooncakeBatchId = std::numeric_limits<uint64_t>::max();

struct MooncakeWriteRequest {
    const void* source_addr = nullptr;
    std::string segment_name;
    uint64_t    target_addr = 0;
    uint64_t    length      = 0;
    int64_t     cache_key   = 0;
    uint32_t    block_index = 0;
};

class IMooncakeTransferEngineAdapter {
public:
    virtual ~IMooncakeTransferEngineAdapter() = default;

    virtual bool init(const MooncakeBackendConfig& config) = 0;

    virtual bool registerLocalMemory(const BlockInfo& block_info, uint64_t aligned_size) = 0;

    virtual bool openSegment(const std::string& segment_name) = 0;

    virtual uint64_t allocateBatchID(size_t request_count) = 0;

    virtual void freeBatchID(uint64_t batch_id) = 0;

    virtual bool submitTransfer(uint64_t batch_id, const std::vector<MooncakeWriteRequest>& requests) = 0;

    // 查询 batch 状态；finished=false 表示仍在传输中，调用方应继续轮询。
    virtual TransferErrorCode getTransferStatus(uint64_t batch_id, bool* finished, std::string* error_message = nullptr) = 0;
};

using IMooncakeTransferEngineAdapterPtr = std::shared_ptr<IMooncakeTransferEngineAdapter>;

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm
