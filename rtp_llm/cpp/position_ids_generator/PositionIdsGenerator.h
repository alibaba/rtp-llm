#pragma once

#include <algorithm>
#include <cstring>
#include <random>
#include <optional>
#include <vector>
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"

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
    static rtp_llm::BufferPtr
    generatePositionIds(rtp_llm::DeviceBase*                           device,
                        int32_t                                        input_len,
                        PositionIdsStyle                               style           = PositionIdsStyle::DEFAULT,
                        std::optional<rtp_llm::BufferPtr>              mm_locs         = std::nullopt,
                        std::optional<std::vector<rtp_llm::BufferPtr>> mm_position_ids = std::nullopt);

    static void generateNextPositionId(int32_t*           now_pos,
                                       int32_t            now_len,
                                       PositionIdsStyle   style           = PositionIdsStyle::DEFAULT,
                                       rtp_llm::BufferPtr context_pos_ids = nullptr);
};

}  // namespace rtp_llm