#pragma once

#include "src/fastertransformer/utils/assert_utils.h"
#include <string>

namespace fastertransformer {

enum class RopeType {
    Base = 1,
    NTKScale = 3,
    NOROPE = 5,
};

struct RopeConfig {
    RopeType embedding_style;
    size_t embedding_dim;
    size_t embedding_base;
    float dynamic_embedding_scale;
    int dynamic_embedding_max_pos;
    float position_embeddings_scale;
    float base_scale;
};

} // namespace fastertransformer
