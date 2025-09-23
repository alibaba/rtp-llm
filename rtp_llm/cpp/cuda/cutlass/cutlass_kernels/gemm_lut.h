#pragma once
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/gemm_lut_utils.h"
#include "cutlass/numeric_types.h"
#include <type_traits>

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

class GemmConfigMap{
    public:
    static GemmLut fp16_int4_lut;
    static GemmLut fp16_int8_lut;
    static GemmLut w8a8_lut;
    static GemmLut moe_fp16_int4_lut;
    static GemmLut moe_fp16_int8_lut;

    static void registerEntryForFp16Int8Lut(int m, int n, int k, CutlassGemmConfig config);
    static void registerEntryForFp16Int4Lut(int m, int n, int k, CutlassGemmConfig config);
    static void registerEntryForW8A8Lut(int m, int n, int k, CutlassGemmConfig config);
    static void registerEntryForMoeFp16Int8Lut(int m, int n, int k, CutlassGemmConfig config);
    static void registerEntryForMoeFp16Int4Lut(int m, int n, int k, CutlassGemmConfig config);
};

template <typename T, typename WeightType, bool is_moe=false>
GemmLut* get_gemm_lut()
{
    if constexpr (is_moe)
    {
        if ((std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value)
            && std::is_same<WeightType, uint8_t>::value)
        {
            return &GemmConfigMap::moe_fp16_int8_lut;
        }
        else if ((std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value)
            && std::is_same<WeightType, cutlass::uint4b_t>::value)
        {
            return &GemmConfigMap::moe_fp16_int4_lut;
        }
    }
    else
    {
        if ((std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value)
            && std::is_same<WeightType, uint8_t>::value)
        {
            return &GemmConfigMap::fp16_int8_lut;
        }
        else if ((std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value)
            && std::is_same<WeightType, cutlass::uint4b_t>::value)
        {
            return &GemmConfigMap::fp16_int4_lut;
        }
        else if (std::is_same<T, uint8_t>::value)
        {
            return &GemmConfigMap::w8a8_lut;
        }
    }

    return nullptr;
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
