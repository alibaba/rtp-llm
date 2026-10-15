#pragma once

#include <string>
#include "rtp_llm/cpp/cache/CacheModuleId.h"

namespace rtp_llm {

/// Shared key format for P2P transfer includes module, local layer, semantic cache tag and partition.
class P2PKeyUtil {
public:
    static std::string makePartitionLayerTagKey(const std::string& base_key,
                                                CacheModuleId      module_id,
                                                int                layer_id,
                                                const std::string& cache_tag,
                                                int                partition_id) {
        return base_key + "_module" + std::to_string(module_id.value()) + "_" + std::to_string(layer_id) + "_tag"
               + cache_tag + "_" + std::to_string(partition_id);
    }
};

}  // namespace rtp_llm
