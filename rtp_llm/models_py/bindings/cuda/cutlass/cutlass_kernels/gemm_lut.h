#pragma once
#include "rtp_llm/models_py/bindings/cuda/cutlass/cutlass_kernels/gemm_lut_utils.h"
#include "cutlass/numeric_types.h"
#include <type_traits>

namespace tensorrt_llm {
namespace kernels {
namespace cutlass_kernels {

class GemmConfigMap {
public:
    static GemmLut moe_fp16_int4_lut;
    static GemmLut moe_fp16_int8_lut;
};

template<typename T, typename WeightType, bool is_moe = false>
GemmLut* get_gemm_lut() {
    if constexpr (is_moe) {
        if ((std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value)
            && std::is_same<WeightType, uint8_t>::value) {
            return &GemmConfigMap::moe_fp16_int8_lut;
        } else if ((std::is_same<T, half>::value || std::is_same<T, __nv_bfloat16>::value)
                   && std::is_same<WeightType, cutlass::uint4b_t>::value) {
            return &GemmConfigMap::moe_fp16_int4_lut;
        }
    }

    return nullptr;
}

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace tensorrt_llm
