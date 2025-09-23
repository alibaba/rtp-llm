#include "torch/extension.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_lut.h"

namespace rtp_llm {

using namespace tensorrt_llm::kernels::cutlass_kernels;

#define CONVERT(s)                                                                                                     \
    if (tile_config == #s) {                                                                                           \
        return {tensorrt_llm::cutlass_extensions::s, split_k, stage};                                                  \
    }

CutlassGemmConfig getConfigFromStr(std::string tile_config, int split_k, int stage) {
    CONVERT(CutlassTileConfig::CtaShape128x128x8_WarpShape64x64x8)
    CONVERT(CutlassTileConfig::CtaShape16x64x256_WarpShape16x16x256)
    CONVERT(CutlassTileConfig::CtaShape32x64x256_WarpShape16x32x256)
    CONVERT(CutlassTileConfig::CtaShape32x32x256_WarpShape16x16x256)

    CONVERT(CutlassTileConfig::CtaShape32x128x128_WarpShape32x32x128)
    CONVERT(CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64)
    CONVERT(CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64)

    CONVERT(CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64)

    CONVERT(CutlassTileConfig::CtaShape64x128x64_WarpShape32x64x64)
    CONVERT(CutlassTileConfig::CtaShape64x64x128_WarpShape32x64x64)
    CONVERT(CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64)

    CONVERT(CutlassTileConfig::CtaShape128x64x64_WarpShape64x32x64)
    CONVERT(CutlassTileConfig::CtaShape128x128x64_WarpShape64x32x64)
    CONVERT(CutlassTileConfig::CtaShape128x128x64_WarpShape64x64x64)
    CONVERT(CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64)
    CONVERT(CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64)
    CONVERT(CutlassTileConfig::CtaShape128x256x128_WarpShape64x64x128)

    CONVERT(CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64)
    CONVERT(CutlassTileConfig::CtaShape256x128x128_WarpShape64x64x128)
    RTP_LLM_FAIL("undefined gemm config %s", tile_config.c_str());
}

#undef CONVERT

void insertFp16Int8GemmConfig(
    int64_t m, int64_t n, int64_t k, std::string tile_config, int64_t split_k, int64_t stage) {
    auto gemm_config = getConfigFromStr(tile_config, split_k, stage);
    GemmConfigMap::registerEntryForFp16Int8Lut(m, n, k, gemm_config);
}

void insertFp16Int4GemmConfig(
    int64_t m, int64_t n, int64_t k, std::string tile_config, int64_t split_k, int64_t stage) {
    auto gemm_config = getConfigFromStr(tile_config, split_k, stage);
    GemmConfigMap::registerEntryForFp16Int4Lut(m, n, k, gemm_config);
}

void insertW8A8GemmConfig(int64_t m, int64_t n, int64_t k, std::string tile_config, int64_t split_k, int64_t stage) {
    auto gemm_config = getConfigFromStr(tile_config, split_k, stage);
    GemmConfigMap::registerEntryForW8A8Lut(m, n, k, gemm_config);
}

static auto insert_fp16_int8_gemm_config =
    torch::RegisterOperators("rtp_llm::insert_fp16_int8_gemm_config", &insertFp16Int8GemmConfig);
static auto insert_fp16_int4_gemm_config =
    torch::RegisterOperators("rtp_llm::insert_fp16_int4_gemm_config", &insertFp16Int4GemmConfig);
static auto insert_w8a8_gemm_config =
    torch::RegisterOperators("rtp_llm::insert_w8a8_gemm_config", &insertW8A8GemmConfig);

}