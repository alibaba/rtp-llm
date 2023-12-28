
#include "src/fastertransformer/cutlass/cutlass_kernels/weightOnlyBatchedGemv/kernel.h"

namespace fastertransformer {
namespace kernels {

template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm75,  WeightOnlyPerChannel,
    IdentityActivation, false, false, 1, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm75,  WeightOnlyPerChannel,
    IdentityActivation, false, false, 2, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm75,  WeightOnlyPerChannel,
    IdentityActivation, false, false, 3, 256>;
template struct WeightOnlyBatchedGemvKernelLauncher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm75,  WeightOnlyPerChannel,
    IdentityActivation, false, false, 4, 256>;


template struct WeightOnlyBatchedGemvKernelSm70Launcher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm70, WeightOnlyPerChannel,
    IdentityActivation, false, false, 1, 64>;
template struct WeightOnlyBatchedGemvKernelSm70Launcher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm70, WeightOnlyPerChannel,
    IdentityActivation, false, false, 1, 128>;
template struct WeightOnlyBatchedGemvKernelSm70Launcher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm70, WeightOnlyPerChannel,
    IdentityActivation, false, false, 1, 256>;
template struct WeightOnlyBatchedGemvKernelSm70Launcher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm70, WeightOnlyPerChannel,
    IdentityActivation, false, false, 2, 64>;
template struct WeightOnlyBatchedGemvKernelSm70Launcher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm70, WeightOnlyPerChannel,
    IdentityActivation, false, false, 2, 128>;
template struct WeightOnlyBatchedGemvKernelSm70Launcher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm70, WeightOnlyPerChannel,
    IdentityActivation, false, false, 2, 256>;
template struct WeightOnlyBatchedGemvKernelSm70Launcher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm70, WeightOnlyPerChannel,
    IdentityActivation, false, false, 3, 64>;
template struct WeightOnlyBatchedGemvKernelSm70Launcher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm70, WeightOnlyPerChannel,
    IdentityActivation, false, false, 3, 128>;
template struct WeightOnlyBatchedGemvKernelSm70Launcher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm70, WeightOnlyPerChannel,
    IdentityActivation, false, false, 3, 256>;
template struct WeightOnlyBatchedGemvKernelSm70Launcher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm70, WeightOnlyPerChannel,
    IdentityActivation, false, false, 4, 64>;
template struct WeightOnlyBatchedGemvKernelSm70Launcher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm70, WeightOnlyPerChannel,
    IdentityActivation, false, false, 4, 128>;
template struct WeightOnlyBatchedGemvKernelSm70Launcher<WeightOnlyQuantType::Int8b, cutlass::arch::Sm70, WeightOnlyPerChannel,
    IdentityActivation, false, false, 4, 256>;
}
}