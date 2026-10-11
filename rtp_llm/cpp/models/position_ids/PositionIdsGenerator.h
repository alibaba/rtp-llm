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

    // Write `token_num * position_id_len_factor` int32 entries into `combo_dst`,
    // the flat combo position-ids buffer consumed by the model input gatherers.
    // Multi-axis rope styles (interleaved MRoPE) read position ids with a
    // stride of `position_id_len_factor` per token, so the buffer must carry
    // that many entries per token: factor 1 copies the arange slice directly,
    // higher factors repeat each position on every axis (the text-only MRoPE
    // (t, t, t) layout; image positions arrive through the multimodal path).
    // Both engines' gatherers must go through this single layout so the
    // buffer geometry cannot drift apart again.
    static void writeComboPositionIds(int32_t*       combo_dst,
                                      const int32_t* position_arange,
                                      int32_t        token_num,
                                      int            position_id_len_factor);
};

}  // namespace rtp_llm
