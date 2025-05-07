#include <cstdlib>

namespace rtp_llm {

// Kernel index 0 2
void int4_dequant_gemm_256x128x128x128_32_32x32_2x2_16x16x1_4x64x1_32_1x32x1x8_8_intrawave_v3(
    const ckGemmParam& params);

// Kernel index 1
void int4_dequant_gemm_256x128x128x64_32_32x32_2x2_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v4(const ckGemmParam& params);

// Kernel index 3
void int4_dequant_gemm_256x128x128x64_32_32x32_2x2_8x32x1_2x128x1_16_1x32x1x8_8_intrawave_v4(const ckGemmParam& params);

// Kernel index 4
void int4_dequant_gemm_256x128x128x64_32_32x32_2x2_8x32x1_2x128x1_16_1x32x1x8_8_intrawave_v3(const ckGemmParam& params);

// Kernel index 5
void int4_dequant_gemm_128x32x16x128_16_16x16_1x1_8x16x1_8x16x1_16_1x16x1x8_2_intrawave_v3(const ckGemmParam& params);

// Kernel index 6 14 15
void int4_dequant_gemm_64x16x16x128_16_16x16_1x1_16x4x1_8x8x1_16_1x16x1x4_4_intrawave_v3(const ckGemmParam& params);

// Kernel index 7
void int4_dequant_gemm_64x16x16x128_16_16x16_1x1_8x8x1_8x8x1_16_1x16x1x4_4_intrawave_v3(const ckGemmParam& params);

// Kernel index 8
void int4_dequant_gemm_128x16x32x128_32_16x16_1x1_8x16x1_4x32x1_32_1x16x1x8_4_intrawave_v3(const ckGemmParam& params);

// Kernel index 9
void int4_dequant_gemm_128x128x32x128_32_32x32_2x1_16x8x1_4x32x1_32_1x16x1x8_4_intrawave_v3(const ckGemmParam& params);

// Kernel index 10
void int4_dequant_gemm_128x128x16x128_16_16x16_4x1_16x8x1_8x16x1_16_1x16x1x8_2_intrawave_v3(const ckGemmParam& params);

// Kernel index 11
void int4_dequant_gemm_128x64x32x128_32_32x32_1x1_16x8x1_4x32x1_32_1x16x1x8_4_intrawave_v3(const ckGemmParam& params);

// Kernel index 12
void int4_dequant_gemm_128x64x16x128_16_16x16_2x1_16x8x1_8x16x1_16_1x16x1x8_2_intrawave_v3(const ckGemmParam& params);

// Kernel index 13
void int4_dequant_gemm_128x32x16x128_16_16x16_1x1_16x8x1_8x16x1_16_1x16x1x8_2_intrawave_v3(const ckGemmParam& params);

// Kernel index 16
void int4_dequant_gemm_128x16x32x128_32_16x16_1x1_16x8x1_4x32x1_32_1x16x1x8_4_intrawave_v3(const ckGemmParam& params);

// Kernel index 17
void int4_dequant_gemm_128x16x64x128_32_16x16_1x2_16x8x1_4x32x1_32_1x16x1x8_4_intrawave_v3(const ckGemmParam& params);

// Kernel index 18
void int4_dequant_gemm_128x32x64x128_32_32x32_1x1_16x8x1_4x32x1_32_1x16x1x8_8_intrawave_v3(const ckGemmParam& params);

// Kernel index 19
void int4_dequant_gemm_128x16x128x128_32_16x16_1x4_16x8x1_4x32x1_32_1x16x1x8_4_intrawave_v3(const ckGemmParam& params);

// Kernel index 20
void int4_dequant_gemm_128x32x128x128_32_32x32_1x2_16x8x1_4x32x1_32_1x16x1x8_8_intrawave_v3(const ckGemmParam& params);

// Kernel index 21
void int4_dequant_gemm_256x16x256x128_32_16x16_1x4_16x8x1_4x32x1_32_1x16x1x16_4_intrawave_v3(const ckGemmParam& params);

// Kernel index 22
void int4_dequant_gemm_256x32x256x128_32_32x32_1x2_16x16x1_4x64x1_32_1x16x1x16_8_intrawave_v3(
    const ckGemmParam& params);

// Kernel index 23
void int4_dequant_gemm_128x64x32x128_32_32x32_1x1_16x8x1_4x32x1_32_1x16x1x8_4_intrawave_v4(const ckGemmParam& params);

// Kernel index 24
void int4_dequant_gemm_128x64x16x128_16_16x16_2x1_16x8x1_8x16x1_16_1x16x1x8_2_intrawave_v4(const ckGemmParam& params);

// Kernel index 25
void int4_dequant_gemm_128x32x16x128_16_16x16_1x1_16x8x1_8x16x1_16_1x16x1x8_2_intrawave_v4(const ckGemmParam& params);

// Kernel index 26 27
void int4_dequant_gemm_64x16x16x128_16_16x16_1x1_16x4x1_8x8x1_16_1x16x1x4_4_intrawave_v4(const ckGemmParam& params);

// Kernel index 28
void int4_dequant_gemm_128x16x32x128_32_16x16_1x1_16x8x1_4x32x1_32_1x16x1x8_4_intrawave_v4(const ckGemmParam& params);

// Kernel index 29
void int4_dequant_gemm_128x16x64x128_32_16x16_1x2_16x8x1_4x32x1_32_1x16x1x8_4_intrawave_v4(const ckGemmParam& params);

// Kernel index 30
void int4_dequant_gemm_128x32x64x128_32_32x32_1x1_16x8x1_4x32x1_32_1x16x1x8_8_intrawave_v4(const ckGemmParam& params);

// Kernel index 31
void int4_dequant_gemm_128x16x128x128_32_16x16_1x4_16x8x1_4x32x1_32_1x16x1x8_4_intrawave_v4(const ckGemmParam& params);

// kernel index 32
void int4_dequant_gemm_128x32x128x128_32_32x32_1x2_16x8x1_4x32x1_32_1x16x1x8_8_intrawave_v4(const ckGemmParam& params);

// kernel index 33
void int4_dequant_gemm_256x16x256x128_32_16x16_1x4_16x8x1_4x32x1_32_1x16x1x16_4_intrawave_v4(const ckGemmParam& params);

// kernel index 34
void int4_dequant_gemm_256x32x256x128_32_32x32_1x2_16x16x1_4x64x1_32_1x16x1x16_8_intrawave_v4(
    const ckGemmParam& params);

// kernel index 35
void int4_dequant_gemm_256x128x128x64_32_32x32_2x2_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3(const ckGemmParam& params);

// kernel index 36
void int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3(const ckGemmParam& params);

// kernel index 37
void int4_dequant_gemm_256x16x64x256_32_16x16_1x1_32x8x1_8x32x1_32_1x16x1x8_8_intrawave_v3(const ckGemmParam& params);

}  // namespace rtp_llm