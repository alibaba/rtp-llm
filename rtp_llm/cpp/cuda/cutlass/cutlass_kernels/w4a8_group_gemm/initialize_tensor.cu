#include "cute/numeric/integral_constant.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/w4a8_group_gemm/w4a8_group_gemm.h"

#define DISPATCH_TYPE(tt, ct)                                                                                          \
    else if (output.dtype() == tt) {                                                                                   \
        cutlass::DeviceAllocation<ct> block(block_size);                                                               \
        initialize_tensor(block, min, max, seed);                                                                      \
        cutlass::device_memory::copy_device_to_device(static_cast<ct*>(output.data_ptr()), block.get(), block_size);   \
    }

template<class Element>
static bool initialize_tensor(cutlass::DeviceAllocation<Element>& block,
                              std::optional<float>                min,
                              std::optional<float>                max,
                              const uint64_t                      seed) {
    double scope_max, scope_min;
    int    bits_input  = cutlass::sizeof_bits<Element>::value;
    int    bits_output = cutlass::sizeof_bits<Element>::value;

    if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
    } else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
    } else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
    } else {
        scope_max = 8;
        scope_min = -8;
    }

    if (min.has_value()) {
        scope_min = *min;
    }

    if (max.has_value()) {
        scope_max = *max;
    }

    cutlass::reference::device::BlockFillRandomUniform(
        block.get(), block.size(), seed, Element(scope_max), Element(scope_min));

    return true;
}

void rtp_llm::run_initialize_tensor(torch::Tensor&       output,
                                    std::optional<float> min,
                                    std::optional<float> max,
                                    const int            seed) {
    TORCH_CHECK(output.dtype() == torch::kInt8 || output.dtype() == torch::kFloat8_e4m3fn
                    || output.dtype() == torch::kBFloat16 || output.dtype() == torch::kFloat16,
                "Output must be of type int8, kFloat8_e4m3fn, kBFloat16 or kFloat16.");

    auto block_size = output.dtype() == torch::kInt8 ?
                          output.numel() * 8 / cutlass::sizeof_bits<cutlass::int4b_t>::value :
                          output.numel();

    if (false) {}
    DISPATCH_TYPE(torch::kInt8, cutlass::int4b_t)
    DISPATCH_TYPE(torch::kFloat8_e4m3fn, cutlass::float_e4m3_t)
    DISPATCH_TYPE(torch::kBFloat16, cutlass::bfloat16_t)
    DISPATCH_TYPE(torch::kFloat16, cutlass::half_t)
    else {
        TORCH_CHECK(false, "Invalid output type (must be int8 or kFloat8_e4m3fn)");
    }
}
#undef DISPATCH_TYPE
