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
    float dynamic_embedding_scale = 0.0;
    int dynamic_embedding_max_pos = 0;
    float position_embeddings_scale = 1.0f;
    float base_scale = 1.0f;
};

} // namespace fastertransformer
