#pragma once

#include <vector>

namespace rtp_llm {

struct ToolCallLoopCheckResult {
    bool hit                 = false;
    int  repeat_count        = 0;
    int  current_span_tokens = 0;
    int  marker_index        = -1;
};

ToolCallLoopCheckResult checkToolCallLoop(const std::vector<int>&              input_ids,
                                          const std::vector<int>&              output_ids,
                                          const std::vector<std::vector<int>>& marker_begin_ids,
                                          const std::vector<std::vector<int>>& marker_end_ids,
                                          int                                  repeat_threshold = 5,
                                          int                                  max_span_tokens  = 16384);

}  // namespace rtp_llm
