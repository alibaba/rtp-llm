#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rtp_llm {
namespace transfer {
namespace mooncake {

struct MooncakeRemoteBlockDescriptor {
    int64_t  cache_key   = 0;
    uint32_t block_index = 0;
    uint64_t target_addr = 0;
    uint64_t len         = 0;
};

struct MooncakeRemoteDescriptor {
    std::string                          segment_name;
    std::vector<MooncakeRemoteBlockDescriptor> blocks;
};

}  // namespace mooncake
}  // namespace transfer
}  // namespace rtp_llm
