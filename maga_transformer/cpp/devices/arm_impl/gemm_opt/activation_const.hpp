/*!
 * Copyright (c) Alibaba, Inc. and its affiliates.
 * @file    activation_const.hpp
 */
#pragma once
#include <cstdint>

// clang-format off
typedef struct activation_const_s {
    const uint32_t exp_const[12] = {
        0x3f7ffffb, // [0.0],  p1 = 0.999999701f
        0x3efffee3, // [0.1],  p2 = 0.499991506f
        0x3e2aad40, // [0.2],  p3 = 0.166676521f
        0x3d2b9d0d, // [0.3],  p4 = 0.0418978221f
        0x3c07cfce, // [1.0],  p5 = 0.00828929059f
        0x0000007e, // [1.1],  exponent_bias - 1
        0x3fb8aa3b, // [1.2],  inv_ln2,       1 / ln(2) = 0x1.715476p+0f
        0xbf317200, // [1.3],  neg_ln2_hi,    -ln(2) from bits  -1 to -19: -0x1.62e400p-1f
        0xb5bfbe8e, // [2.0],  neg_ln2_lo,    -ln(2) from bits -20 to -42: -0x1.7f7d1cp-20f
        0x42b17218, // [2.1],  max_input,     88.72283935546875
        0xc2aeac50, // [2.2],  min_input,     -87.3365478515625
        0x7f800000, // [2.3],  inf,           std::numeric_limits<float>::infinity()
    };

    const uint32_t erf_const[8] = {
        0x3ea7ba05, // [0.0] approx const, 0.32759109139442444
        0x3e827906, // [0.1] p0, 0.2548295855522156
        0xbe91a98e, // [0.2] p1, -0.2844967246055603
        0x3fb5f0e3, // [0.3] p2, 1.421413779258728
        0xbfba00e3, // [1.0] p3, -1.453152060508728
        0x3f87dc22, // [1.1] p4, 1.0614054203033447
        0,          // [1.2]
        0           // [1.3]
    };

    const float gelu_tanh_const[2] = {
        0.044715, 
        0.7978845608028654 // sqrt(2/pi)
    };

const float inv_sqrt = 0.7071067690849304; // 1/sqrt(2)
} activation_const_t;
// clang-format on
