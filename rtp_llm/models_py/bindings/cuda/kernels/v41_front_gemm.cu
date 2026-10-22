#include "rtp_llm/models_py/bindings/cuda/kernels/v41_front_gemm.h"

// Device entry points are provided by a build-time patch to the pinned wheel.
// The usual DeepGEMM global entry points are unchanged in other translation units.
#define DG_GEMM_DEVICE_ENTRY
#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>
#include <deep_gemm/impls/sm100_bf16_gemm.cuh>
#undef DG_GEMM_DEVICE_ENTRY
#include <algorithm>
#include <stdexcept>
#include <string>

namespace rtp_llm {
namespace {
using namespace deep_gemm;

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void check(CUresult error) {
    if (error != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(error, &message);
        throw std::runtime_error(message ? message : "CUDA tensor map error");
    }
}

template<unsigned BM, unsigned ST>
using FP8Storage = layout::SM100FP8FP4GemmSharedStorage<
    ST, 2, 2, BM / 2, 128, 128, 16, 128, 128, 128, 1,
    cutlass::float_e4m3_t, cutlass::float_e4m3_t, cutlass::bfloat16_t>;
using BF16Storage = layout::SM100BF16GemmSharedStorage<
    8, 2, 2, 32, 16, 128, 32, 32, cutlass::bfloat16_t>;

template<unsigned BM, unsigned ST>
__global__ __launch_bounds__(256, 1) void frontGemm(
    unsigned m, const __grid_constant__ EpilogueOperatorArgs ep,
    const __grid_constant__ cute::TmaDescriptor a,
    const __grid_constant__ cute::TmaDescriptor b,
    const __grid_constant__ cute::TmaDescriptor sa,
    const __grid_constant__ cute::TmaDescriptor sb,
    const __grid_constant__ cute::TmaDescriptor out,
    const __grid_constant__ cute::TmaDescriptor x,
    const __grid_constant__ cute::TmaDescriptor head,
    const __grid_constant__ cute::TmaDescriptor head_out) {
    // 112 FP8 CTAs and 36 BF16 CTAs, in independent two-CTA clusters.
    // Each head tile owns its entire K reduction; no split-K scratch/reduction.
    if (blockIdx.x < 112) {
        sm100_fp8_fp4_gemm_1d1d_impl<cute::UMMA::Major::K, cute::UMMA::Major::K,
            32,32,32,0,1792,5120,BM,128,128,1,128,128,128,ST,2,
            128,128,2,true,112,true,true,GemmType::Normal,false,
            cutlass::float_e4m3_t,cutlass::float_e4m3_t,cutlass::bfloat16_t,
            epilogue::operators::Identity>(nullptr,m,1792,5120,ep,a,b,sa,sb,out);
    } else {
        sm100_bf16_gemm_impl<cute::UMMA::Major::K, cute::UMMA::Major::K,
            0,32,5120,32,32,64,1,128,128,64,16,128,128,
            2,false,36,64,false,true,GemmType::Normal,false,
            cutlass::bfloat16_t,epilogue::operators::Identity,100,112>(
                nullptr,m,32,5120,ep,x,head,head_out);
    }
}

cute::TmaDescriptor tensorMap(void* ptr, unsigned rows, unsigned cols,
    unsigned bytes, unsigned box_rows, unsigned box_cols, unsigned swizzle) {
    cute::TmaDescriptor out{};
    cuuint64_t dims[] = {cols, rows}, strides[] = {uint64_t(cols) * bytes};
    cuuint32_t box[] = {box_cols, box_rows}, elements[] = {1, 1};
    auto dtype = bytes == 1 ? CU_TENSOR_MAP_DATA_TYPE_UINT8 :
                 bytes == 2 ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 : CU_TENSOR_MAP_DATA_TYPE_INT32;
    auto sw = swizzle == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
              swizzle == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_NONE;
    check(cuTensorMapEncodeTiled(&out, dtype, 2, ptr, dims, strides, box, elements,
        CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return out;
}

template<unsigned BM, unsigned ST>
void launch(void* a, void* sa, void* b, void* sb, void* x, void* head,
            void* out, void* head_out, unsigned m, cudaStream_t stream) {
    constexpr unsigned smem = std::max(sizeof(FP8Storage<BM,ST>), sizeof(BF16Storage));
    // An idempotent host attribute update; no module loading, allocation or sync.
    check(cudaFuncSetAttribute(frontGemm<BM,ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim = {2,1,1};
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(148);
    config.blockDim = dim3(256);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;
    // Runtime copies the maps by value into kernel parameters, including during
    // graph capture. Device helpers take references to this parameter storage.
    check(cudaLaunchKernelEx(&config, frontGemm<BM,ST>, m, EpilogueOperatorArgs{},
        tensorMap(a,m,5120,1,BM/2,128,128), tensorMap(b,1792,5120,1,128,128,128),
        tensorMap(sa,40,m,4,1,128,0), tensorMap(sb,40,1792,4,1,128,0),
        tensorMap(out,m,1792,2,16,64,128), tensorMap(x,m,5120,2,32,64,128),
        tensorMap(head,32,5120,2,16,64,128), tensorMap(head_out,m,32,2,32,32,64)));
}
}  // namespace

void invokeV41FrontGemm(void* a, void* sa, void* b, void* sb, void* x, void* head,
                        void* out, void* head_out, unsigned m, cudaStream_t stream) {
    switch (m) {
        case 128: return launch<16,11>(a,sa,b,sb,x,head,out,head_out,m,stream);
        case 256: return launch<32,11>(a,sa,b,sb,x,head,out,head_out,m,stream);
        case 384: return launch<48,10>(a,sa,b,sb,x,head,out,head_out,m,stream);
        case 512: return launch<64,10>(a,sa,b,sb,x,head,out,head_out,m,stream);
        default: throw std::invalid_argument("V41 front GEMM requires M=128/256/384/512");
    }
}
}  // namespace rtp_llm
