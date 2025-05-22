#include "rocmCKGemmWrapper.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_b_scale.hpp"
#include "ck/library/utility/device_memory.hpp"

#include "int4_gemm_kernels/int4_dequant_kernel_manifest.h"

namespace rtp_llm {

using int4QuantKernel = std::function<void(const ckGemmParam&)>;

// Define a custom hash function for std::tuple<int, int, int>
struct IntTupleHash {
    size_t operator()(const std::tuple<int, int, int>& t) const {
        auto hash1 = std::hash<int>{}(std::get<0>(t));
        auto hash2 = std::hash<int>{}(std::get<1>(t));
        auto hash3 = std::hash<int>{}(std::get<2>(t));
        return hash1 ^ hash2 ^ hash3;
    }
};

// For certain high priority shapes, we directly map to the best kernel rather
// than use heuristics.
static const std::unordered_map<std::tuple<int, int, int>, int4QuantKernel, IntTupleHash> int4Gemm_lookup_dispatch = {
    // Need to fill this table after tuning.

    {{1, 29696, 8192}, int4_dequant_gemm_256x16x64x256_32_16x16_1x1_32x8x1_8x32x1_32_1x16x1x8_8_intrawave_v3},
    {{16, 29696, 8192}, int4_dequant_gemm_256x16x64x256_32_16x16_1x1_32x8x1_8x32x1_32_1x16x1x8_8_intrawave_v3},
    {{32, 29696, 8192}, int4_dequant_gemm_128x32x64x128_32_32x32_1x1_16x8x1_4x32x1_32_1x16x1x8_8_intrawave_v3},
    {{48, 29696, 8192}, int4_dequant_gemm_256x16x64x256_32_16x16_1x1_32x8x1_8x32x1_32_1x16x1x8_8_intrawave_v3},
    {{64, 29696, 8192}, int4_dequant_gemm_128x32x128x128_32_32x32_1x2_16x8x1_4x32x1_32_1x16x1x8_8_intrawave_v3},
    {{80, 29696, 8192}, int4_dequant_gemm_256x128x128x128_32_32x32_2x2_16x16x1_4x64x1_32_1x32x1x8_8_intrawave_v3},
    {{96, 29696, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{112, 29696, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{128, 29696, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{256, 29696, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{512, 29696, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{1024, 29696, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{2048, 29696, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{4096, 29696, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{8192, 29696, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},

    {{1, 8192, 29696}, int4_dequant_gemm_128x16x32x128_32_16x16_1x1_8x16x1_4x32x1_32_1x16x1x8_4_intrawave_v3},
    {{16, 8192, 29696}, int4_dequant_gemm_256x16x64x256_32_16x16_1x1_32x8x1_8x32x1_32_1x16x1x8_8_intrawave_v3},
    {{32, 8192, 29696}, int4_dequant_gemm_128x32x64x128_32_32x32_1x1_16x8x1_4x32x1_32_1x16x1x8_8_intrawave_v3},
    {{48, 8192, 29696}, int4_dequant_gemm_256x128x128x128_32_32x32_2x2_16x16x1_4x64x1_32_1x32x1x8_8_intrawave_v3},
    {{64, 8192, 29696}, int4_dequant_gemm_256x128x128x128_32_32x32_2x2_16x16x1_4x64x1_32_1x32x1x8_8_intrawave_v3},
    {{80, 8192, 29696}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{96, 8192, 29696}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{112, 8192, 29696}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{128, 8192, 29696}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{256, 8192, 29696}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{512, 8192, 29696}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{1024, 8192, 29696}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{2048, 8192, 29696}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{4096, 8192, 29696}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{8192, 8192, 29696}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},

    {{1, 10240, 8192}, int4_dequant_gemm_256x16x64x256_32_16x16_1x1_32x8x1_8x32x1_32_1x16x1x8_8_intrawave_v3},
    {{16, 10240, 8192}, int4_dequant_gemm_256x16x64x256_32_16x16_1x1_32x8x1_8x32x1_32_1x16x1x8_8_intrawave_v3},
    {{32, 10240, 8192}, int4_dequant_gemm_128x32x128x128_32_32x32_1x2_16x8x1_4x32x1_32_1x16x1x8_8_intrawave_v3},
    {{48, 10240, 8192}, int4_dequant_gemm_256x128x128x128_32_32x32_2x2_16x16x1_4x64x1_32_1x32x1x8_8_intrawave_v3},
    {{64, 10240, 8192}, int4_dequant_gemm_128x32x64x128_32_32x32_1x1_16x8x1_4x32x1_32_1x16x1x8_8_intrawave_v3},
    {{80, 10240, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{96, 10240, 8192}, int4_dequant_gemm_256x128x128x128_32_32x32_2x2_16x16x1_4x64x1_32_1x32x1x8_8_intrawave_v3},
    {{112, 10240, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{128, 10240, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{256, 10240, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{512, 10240, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{1024, 10240, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{2048, 10240, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{4096, 10240, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{8192, 10240, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},

    {{1, 8192, 8192}, int4_dequant_gemm_256x16x64x256_32_16x16_1x1_32x8x1_8x32x1_32_1x16x1x8_8_intrawave_v3},
    {{16, 8192, 8192}, int4_dequant_gemm_256x16x64x256_32_16x16_1x1_32x8x1_8x32x1_32_1x16x1x8_8_intrawave_v3},
    {{32, 8192, 8192}, int4_dequant_gemm_128x32x64x128_32_32x32_1x1_16x8x1_4x32x1_32_1x16x1x8_8_intrawave_v4},
    {{48, 8192, 8192}, int4_dequant_gemm_256x128x128x128_32_32x32_2x2_16x16x1_4x64x1_32_1x32x1x8_8_intrawave_v3},
    {{64, 8192, 8192}, int4_dequant_gemm_256x128x128x128_32_32x32_2x2_16x16x1_4x64x1_32_1x32x1x8_8_intrawave_v3},
    {{80, 8192, 8192}, int4_dequant_gemm_256x128x128x128_32_32x32_2x2_16x16x1_4x64x1_32_1x32x1x8_8_intrawave_v3},
    {{96, 8192, 8192}, int4_dequant_gemm_256x128x128x128_32_32x32_2x2_16x16x1_4x64x1_32_1x32x1x8_8_intrawave_v3},
    {{112, 8192, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{128, 8192, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{256, 8192, 8192}, int4_dequant_gemm_256x128x128x128_32_32x32_2x2_16x16x1_4x64x1_32_1x32x1x8_8_intrawave_v3},
    {{512, 8192, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{1024, 8192, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{2048, 8192, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{4096, 8192, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3},
    {{8192, 8192, 8192}, int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3}

};

int4QuantKernel int4Gemm_heuristic_dispatch(int M, int N, int K) {
    // Apply shape heuristics to find a suitable kernel implementation.

    //   if (M < 64 && N < 2048 && K < 2048) {
    // Kernel that generally works well on small shapes.
    return int4_dequant_gemm_256x128x128x64_32_32x32_4x1_8x32x1_2x128x1_32_1x32x1x8_8_intrawave_v3;
}

int4QuantKernel int4Gemm_dispatch(int M, int N, int K) {
    // For a given shape, either find the best kernel via lookup or heuristic.
    // For many small M shapes, we bucket them to the next largest kernel.
    // This is fine since kernels are padded anyway.
    int padded_m = M;
    if (M == 1) {
        padded_m = 1;
    } else if (M <= 16) {
        padded_m = 16;
    } else if (M <= 32) {
        padded_m = 32;
    } else if (M <= 48) {
        padded_m = 48;
    } else if (M <= 64) {
        padded_m = 64;
    } else if (M <= 80) {
        padded_m = 80;
    } else if (M <= 96) {
        padded_m = 96;
    } else if (M <= 112) {
        padded_m = 112;
    } else if (M <= 128) {
        padded_m = 128;
    }
    // First check if this shape is available in the direct lookup.
    auto it = int4Gemm_lookup_dispatch.find({padded_m, N, K});
    // If we found an optimal kernel, use it.
    if (it != int4Gemm_lookup_dispatch.end()) {
        return it->second;
    }
    // Otherwise, use heuristics.
    return int4Gemm_heuristic_dispatch(M, N, K);
}

template<typename ADtype, typename BDtype>
void CKGemmImpl(const ckGemmParam& params) {
    // Call HipblasLT GEMM instead.
    return;
}

template<typename ADtype, typename BDtype>
void CKGemmQINT4X2Impl(const ckGemmParam& params) {
    auto M = static_cast<int>(params.M);
    auto N = static_cast<int>(params.N);
    auto K = static_cast<int>(params.K);

    int4QuantKernel int4Gemm_caller = int4Gemm_dispatch(M, N, K);
    int4Gemm_caller(params);
}

void rocmCKGemmWrapper::runCKGemm(const ckGemmParam& ckParams, DataType ADtype, DataType BDtype) {
    if (BDtype == TYPE_INT8) {
        // implemented here for fusion dequantize GEMM
        // BDtype is the weight
    } else if (ADtype == DataType::TYPE_FP16 && BDtype == DataType::TYPE_QINT4X2) {
        CKGemmQINT4X2Impl<ck::half_t, ck::pk_i4_t>(ckParams);
    } else if (ADtype == DataType::TYPE_FP16 && BDtype == DataType::TYPE_FP16) {
        CKGemmImpl<ck::half_t, ck::half_t>(ckParams);
    } else {
        RTP_LLM_LOG_ERROR("input A type: %d and B type: %d are not supported by CK lib", ADtype, BDtype);
    }
};

}  // namespace rtp_llm
