#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"

#include "cutlass_extensions/arch/mma.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template<typename arch>
struct GroupGemmArchTraits {
};

template<>
struct GroupGemmArchTraits<cutlass::arch::Sm70> {
    static constexpr int Stages = 2;

    using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

};

template<>
struct GroupGemmArchTraits<cutlass::arch::Sm75> {
    static constexpr int Stages = 2;

    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

};

template<>
struct GroupGemmArchTraits<cutlass::arch::Sm80> {
    static constexpr int Stages = 4;

    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass