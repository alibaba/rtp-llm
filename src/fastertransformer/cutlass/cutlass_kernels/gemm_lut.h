#pragma once
#include "src/fastertransformer/cutlass/cutlass_kernels/gemm_lut_utils.h"
#include "cutlass/numeric_types.h"

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

class GemmConfigMap{
    public:
    static GemmLut fp16_int4_lut;
    static GemmLut fp16_int8_lut;

    static void registerEntryForFp16Int8Lut(int m, int n, int k, CutlassGemmConfig config);
    static void registerEntryForFp16Int4Lut(int m, int n, int k, CutlassGemmConfig config);
};

template<typename T, typename WeightType>
GemmLut* get_gemm_lut() {
    if ((std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value)&& std::is_same<WeightType, uint8_t>::value) {
        return &GemmConfigMap::fp16_int8_lut;
    } 
    else if((std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value) && std::is_same<WeightType, cutlass::uint4b_t>::value) {
        return &GemmConfigMap::fp16_int4_lut;
    }else{
        return nullptr;
    } 
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
