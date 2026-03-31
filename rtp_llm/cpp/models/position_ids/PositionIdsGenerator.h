#pragma once

#include <algorithm>
#include <cstring>
#include <random>
#include <optional>
#include <vector>
#include <torch/python.h>

namespace rtp_llm {

typedef enum {
    DEFAULT   = 0,
    MMWITHTAG = 1,  // chatglm4v
    MROPE     = 2   // qwen2vl
} PositionIdsStyle;

// assume position ids in position encoding (word embedding) and rotary embedding is same
// TODO: not same -> implement different interface here and BatchStreamProcessor
class PositionIdsGenerator {
public:
    static torch::Tensor generatePositionIds(int32_t                      input_len,
                                             PositionIdsStyle             style   = PositionIdsStyle::DEFAULT,
                                             std::optional<torch::Tensor> mm_locs = std::nullopt,
                                             std::optional<std::vector<torch::Tensor>> mm_position_ids = std::nullopt);

    static void generateNextPositionId(int32_t*         now_pos,
                                       int32_t          now_len,
                                       PositionIdsStyle style           = PositionIdsStyle::DEFAULT,
                                       torch::Tensor    context_pos_ids = torch::Tensor());
};

}  // namespace rtp_llm
