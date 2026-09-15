#pragma once

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace rtp_llm {
namespace transfer {

/// @brief 单个 cache block 的传输元数据：cache key + 对应的内存地址列表（blocks）
struct KeyBlockInfo {
    int64_t                cache_key = 0;
    std::vector<BlockInfo> blocks;
};

using KeyBlockInfoPtr = std::shared_ptr<const KeyBlockInfo>;
using KeyBlockInfoMap = std::unordered_map<int64_t, KeyBlockInfoPtr>;

}  // namespace transfer
}  // namespace rtp_llm
