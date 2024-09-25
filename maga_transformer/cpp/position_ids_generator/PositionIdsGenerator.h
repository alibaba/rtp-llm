#pragma once

#include <algorithm>
#include <cstring>
#include <random>
#include <optional>
#include <vector>
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/DeviceBase.h"

namespace ft = fastertransformer;

namespace rtp_llm {

typedef enum
{
    DEFAULT   = 0,
    MMWITHTAG = 1, // chatglm4v
    MROPE     = 2  // qwen2vl
} PositionIdsStyle;

// assume position ids in position encoding (word embedding) and rotary embedding is same
// TODO: not same -> implement different interface here and BatchStreamProcessor
class PositionIdsGenerator {
public:
    static ft::BufferPtr generatePositionIds(ft::DeviceBase* device, 
                                             int32_t input_len, 
                                             PositionIdsStyle style = PositionIdsStyle::DEFAULT, 
                                             std::optional<ft::BufferPtr> mm_locs = std::nullopt,
                                             std::optional<std::vector<ft::BufferPtr>> mm_position_ids = std::nullopt);

    static void generateNextPositionId(int32_t* now_pos,
                                     int32_t  now_len,
                                     PositionIdsStyle style = PositionIdsStyle::DEFAULT,
                                     ft::BufferPtr context_pos_ids = nullptr);
};

} // namespace rtp_llm