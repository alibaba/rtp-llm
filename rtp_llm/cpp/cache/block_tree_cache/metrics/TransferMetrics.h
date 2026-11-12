#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

#include "rtp_llm/cpp/cache/CacheTier.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"

namespace rtp_llm {

enum class CacheTransferOperation : uint8_t {
    LOAD,
    EVICT,
};

const char* cacheTransferOperationName(CacheTransferOperation operation);

struct BlockTreeTransferBytesKey {
    std::string    pool_name;
    CacheGroupType group_type{CacheGroupType::FULL};

    bool operator==(const BlockTreeTransferBytesKey& other) const {
        return pool_name == other.pool_name && group_type == other.group_type;
    }
};

struct BlockTreeTransferBytesKeyHash {
    size_t operator()(const BlockTreeTransferBytesKey& key) const {
        return std::hash<std::string>{}(key.pool_name) ^ (static_cast<size_t>(key.group_type) << 1);
    }
};

using BlockTreeTransferBytes = std::unordered_map<BlockTreeTransferBytesKey, size_t, BlockTreeTransferBytesKeyHash>;

}  // namespace rtp_llm
