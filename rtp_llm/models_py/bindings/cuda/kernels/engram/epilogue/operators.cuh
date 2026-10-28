#pragma once

#include <rtp_llm/models_py/bindings/cuda/kernels/engram/common/math.cuh>
#include <rtp_llm/models_py/bindings/cuda/kernels/engram/common/types.cuh>
#include <rtp_llm/models_py/bindings/cuda/kernels/engram/ptx/ld_st.cuh>
#include <rtp_llm/models_py/bindings/cuda/kernels/engram/ptx/utils.cuh>

namespace rtp_llm::engram::epilogue::operators {

// Identity also provides shared parameters, lifecycle hooks and default BF16 conversion.
// NOTES: the operators do not compose with each other
struct Identity {
    using Arguments = EpilogueOperatorArgs;
    const EpilogueOperatorArgs& args;

    static constexpr bool kHasRowStore            = false;
    static constexpr bool kIsFP8CastWithClamp     = false;
    static constexpr bool kWithStochasticRounding = false;

    CUTLASS_DEVICE explicit Identity(const EpilogueOperatorArgs& args): args(args) {}

    CUTLASS_DEVICE void init(const uint32_t&, const uint32_t&) {}

    CUTLASS_DEVICE void finalize() const {}

    template<uint32_t kNumValues>
    CUTLASS_DEVICE void apply_values(uint32_t (&)[kNumValues]) const {}

    template<bool kSwapAB>
    CUTLASS_DEVICE void
    cast_bf16(const uint32_t (&values)[8], uint32_t (&packed)[4], const uint32_t&, const uint32_t&) const {
#pragma unroll
        for (uint32_t pair_idx = 0; pair_idx < 4; ++pair_idx)
            packed[pair_idx] = math::cast_into_bf16_and_pack(values[pair_idx * 2], values[pair_idx * 2 + 1]);
    }
};

}  // namespace rtp_llm::engram::epilogue::operators
