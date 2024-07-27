#pragma once

#include <string>

namespace fastertransformer {

enum class RopeType {
    No = 0,
    Base = 1,
    Glm2 = 2,
    DynamicNTK = 3,
    QwenDynamicNTK = 4,
    Yarn = 5,
    Llama3 = 6,
};

// low_freq_factor, high_freq_factor for llama3
// beta_slow, beta_fast for yarn

struct RopeConfig {
    RopeType style = RopeType::No;
    int dim = 0;
    int base = 10000;

    float scale = 1.0;
    float factor1 = 0;
    float factor2 = 0;
    int max_pos = 0;
    float extrapolation_factor = 1.0;
};

} // namespace fastertransformer
