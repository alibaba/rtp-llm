#pragma once

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include <string>

namespace rtp_llm {

class P2PKeyUtil {
public:
    static std::string makePartitionLayerKey(const std::string& base_key, int layer_id, int partition_id) {
        return base_key + "_" + std::to_string(layer_id) + "_" + std::to_string(partition_id);
    }

    static std::string makePartitionLayerKey(const std::string& base_key,
                                             int                layer_id,
                                             KVCacheRegionName  region_name,
                                             int                partition_id) {
        if (region_name == KVCacheRegionName::DEFAULT) {
            return makePartitionLayerKey(base_key, layer_id, partition_id);
        }
        return base_key + "_" + std::to_string(layer_id) + "_r" + std::to_string(static_cast<int>(region_name)) + "_"
               + std::to_string(partition_id);
    }
};

}  // namespace rtp_llm
