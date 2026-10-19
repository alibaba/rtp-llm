#include "rtp_llm/models_py/bindings/cuda/kernels/v41_qb_gemm.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/v41_qb_device.cuh"
#include <stdexcept>

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


using Storage = layout::SM100FP8FP4GemmSharedStorage<
    5,2,2,128,128,128,128,64,128,128,1,
    cutlass::float_e4m3_t,cutlass::float_e4m3_t,cutlass::bfloat16_t>;

__global__ __launch_bounds__(256,1) void qbGemm(
    unsigned m, const __grid_constant__ EpilogueOperatorArgs ep,
    const __grid_constant__ QBArgs qb,
    const __grid_constant__ cute::TmaDescriptor a,
    const __grid_constant__ cute::TmaDescriptor b,
    const __grid_constant__ cute::TmaDescriptor sa,
    const __grid_constant__ cute::TmaDescriptor sb,
    const __grid_constant__ cute::TmaDescriptor c) {
    v41_qb_gemm_impl<cute::UMMA::Major::K,cute::UMMA::Major::K,32,32,32,0,36864,1280,
        128,128,128,1,128,128,128,5,2,128,128,1,false,148,false,true,GemmType::Normal,false,
        cutlass::float_e4m3_t,cutlass::float_e4m3_t,cutlass::bfloat16_t,
        epilogue::operators::Identity>(nullptr,m,36864,1280,ep,a,b,sa,sb,c,qb);
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


}  // namespace

void invokeV41QBGemm(void* a, void* sa, void* b, void* sb, void* out,
                     void* payload, void* sf, void* scaled_weights, void* weights,
                     void* frequencies, void* positions, bool positions_i64,
                     unsigned m, unsigned max_position, cudaStream_t stream) {
    check(cudaFuncSetAttribute(qbGemm, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage)));
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(148);
    config.blockDim = dim3(256);
    config.dynamicSmemBytes = sizeof(Storage);
    config.stream = stream;
    QBArgs args{static_cast<unsigned char*>(payload),static_cast<unsigned char*>(sf),
        static_cast<__nv_bfloat16*>(scaled_weights),static_cast<const __nv_bfloat16*>(weights),
        static_cast<const float*>(frequencies),positions,positions_i64,m,max_position};
    // Maps are copied by value; no module/pointer cache or host synchronization.
    check(cudaLaunchKernelEx(&config, qbGemm, m, EpilogueOperatorArgs{}, args,
        tensorMap(a,m,1280,1,128,128,128), tensorMap(b,36864,1280,1,128,128,128),
        tensorMap(sa,10,m,4,1,128,0), tensorMap(sb,10,36864,4,1,128,0),
        tensorMap(out,m,36864,2,128,64,128)));
}
}  // namespace rtp_llm
