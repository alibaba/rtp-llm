#pragma once

#include <string>
#include "rtp_llm/cpp/utils/utils.h"
namespace rtp_llm {

enum class RopeStyle {
    No             = 0,
    Base           = 1,
    Glm2           = 2,
    DynamicNTK     = 3,
    QwenDynamicNTK = 4,
    Yarn           = 5,
    Llama3         = 6,
    Mrope          = 7,
};

// low_freq_factor, high_freq_factor for llama3
// beta_slow, beta_fast for yarn

struct RopeConfig {
    RopeStyle style = RopeStyle::No;
    int       dim   = 0;
    int       base  = 10000;

    float scale                = 1.0;
    float factor1              = 1.0;  // beta_slow
    float factor2              = 1.0;  // beta_fast
    int   max_pos              = 0;    // original_max_position_embeddings
    float extrapolation_factor = 1.0;
    float mscale               = 1.0;
    int   offset               = 0;
    int   index_factor         = 1;
    int   mrope_dim1           = 0;
    int   mrope_dim2           = 0;
    int   mrope_dim3           = 0;

    std::string DebugRopeConfigStr() const;
};

#define FT_ROPE_SWITCH(COND, CONST_NAME, ...)                                                                          \
    [&] {                                                                                                              \
        switch (COND) {                                                                                                \
            FT_SWITCH_ONE_CASE(CONST_NAME, RopeStyle::No, __VA_ARGS__)                                                 \
            FT_SWITCH_ONE_CASE(CONST_NAME, RopeStyle::Base, __VA_ARGS__)                                               \
            FT_SWITCH_ONE_CASE(CONST_NAME, RopeStyle::Glm2, __VA_ARGS__)                                               \
            FT_SWITCH_ONE_CASE(CONST_NAME, RopeStyle::DynamicNTK, __VA_ARGS__)                                         \
            FT_SWITCH_ONE_CASE(CONST_NAME, RopeStyle::QwenDynamicNTK, __VA_ARGS__)                                     \
            FT_SWITCH_ONE_CASE(CONST_NAME, RopeStyle::Yarn, __VA_ARGS__)                                               \
            FT_SWITCH_ONE_CASE(CONST_NAME, RopeStyle::Llama3, __VA_ARGS__)                                             \
            FT_SWITCH_ONE_CASE(CONST_NAME, RopeStyle::Mrope, __VA_ARGS__)                                              \
        }                                                                                                              \
    }()

#define FT_ROPE_WITH_CACHE_SWITCH(COND, CONST_NAME, ...)                                                               \
    [&] {                                                                                                              \
        switch (COND) {                                                                                                \
            FT_SWITCH_ONE_CASE(CONST_NAME, RopeStyle::Base, __VA_ARGS__)                                               \
            FT_SWITCH_ONE_CASE(CONST_NAME, RopeStyle::Yarn, __VA_ARGS__)                                               \
        }                                                                                                              \
    }()

}  // namespace rtp_llm
