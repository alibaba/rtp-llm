#pragma once

#include <string>

namespace rtp_llm {

/// Shared key format for P2P transfer includes the semantic cache tag.
class P2PKeyUtil {
public:
    static std::string makePartitionLayerTagKey(const std::string& base_key,
                                                int                layer_id,
                                                const std::string& cache_tag,
                                                int                partition_id) {
        return base_key + "_" + std::to_string(layer_id) + "_tag" + cache_tag + "_" + std::to_string(partition_id);
    }
};

}  // namespace rtp_llm
