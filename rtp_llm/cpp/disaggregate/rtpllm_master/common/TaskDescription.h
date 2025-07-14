#pragma once
#include <vector>
#include <string>
#include <cstdint>

namespace rtp_llm {
namespace rtp_llm_master {

struct TaskDescription {
public:
    std::string          task_id;
    int                  prefix_length;
    int                  input_length;
    std::vector<int>     token_ids;
    std::vector<int64_t> hash_block_ids;
};

inline TaskDescription createDummyTask() {
    return TaskDescription({"", 0, 1, {}, {}});
}

}  // namespace rtp_llm_master

}  // namespace rtp_llm