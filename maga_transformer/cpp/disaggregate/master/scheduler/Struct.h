#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace rtp_llm {
namespace rtp_llm_master {

struct RequestInfo {
    std::vector<int64_t> hash_block_info;
    std::vector<int64_t> token_ids;
    int                  input_length;
};

struct MachineInfo {
    std::string ip;
    int         port;
};

struct WorkerStatus {
    std::string ip;
};

}  // namespace rtp_llm_master
}  // namespace rtp_llm