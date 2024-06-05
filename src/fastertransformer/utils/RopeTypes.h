#pragma once

#include "src/fastertransformer/utils/assert_utils.h"
#include <string>

namespace fastertransformer {

enum class RopeType {
    NOROPE = 0,
    NTKScale = 3,
    Base = 1,
};

struct RopeConfig {
    RopeType embedding_style;
    size_t embedding_dim;
    size_t embedding_base;
    float rotary_embedding_scale = 1.0;
    int dynamic_embedding_max_pos = 0;
    float base_scale = 1.0f;
    bool use_logn_attn = false;
    int logn_seq_len = 2048;
};

} // namespace fastertransformer
