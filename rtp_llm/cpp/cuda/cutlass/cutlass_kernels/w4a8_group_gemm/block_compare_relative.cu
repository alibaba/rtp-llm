#include "cute/numeric/integral_constant.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/w4a8_group_gemm/w4a8_group_gemm.h"

#define DISPATCH_TYPE(tt, ct)                                                                                          \
    else if (a.dtype() == tt) {                                                                                        \
        passed = cutlass::reference::device::BlockCompareRelativelyEqual(static_cast<ct*>(a.data_ptr()),               \
                                                                         static_cast<ct*>(b.data_ptr()),               \
                                                                         a.numel(),                                    \
                                                                         static_cast<ct>(epsilon),                     \
                                                                         static_cast<ct>(nonzero_floor));              \
    }

bool rtp_llm::run_block_compare_relative(const torch::Tensor& a,
                                         const torch::Tensor& b,
                                         const float          epsilon,
                                         const float          nonzero_floor) {
    TORCH_CHECK(a.dtype() == torch::kBFloat16 || a.dtype() == torch::kFloat16 || a.dtype() == torch::kFloat8_e4m3fn,
                "Output must be of type kBFloat16, kFloat16 or kFloat8_e4m3fn.");
    TORCH_CHECK(a.dtype() == b.dtype(), "The types of a and b must be consistent.");

    bool passed = false;

    if (false) {}
    DISPATCH_TYPE(torch::kBFloat16, cutlass::bfloat16_t)
    DISPATCH_TYPE(torch::kFloat16, cutlass::half_t)
    DISPATCH_TYPE(torch::kFloat8_e4m3fn, cutlass::float_e4m3_t)
    else {
        TORCH_CHECK(false, "Invalid output type (must be int8 or kFloat8_e4m3fn)");
    }

    return passed;
}
#undef DISPATCH_TYPE
