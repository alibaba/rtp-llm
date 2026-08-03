#pragma once

#include <string>

namespace rtp_llm {

/// Shared key format for P2P transfer: base_key + "_" + layer_id + "_" + partition_id
class P2PKeyUtil {
public:
    static std::string makePartitionLayerKey(const std::string& base_key, int layer_id, int partition_id) {
        return base_key + "_" + std::to_string(layer_id) + "_" + std::to_string(partition_id);
    }
};

}  // namespace rtp_llm
