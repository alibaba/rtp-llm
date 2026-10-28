#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#include <set>
#include "rtp_llm/models_py/bindings/cuda/kernels/engram_transition.h"

namespace rtp_llm::engram {
inline void check(
    const at::Tensor& t, const at::Tensor& owner, at::ScalarType dtype, at::IntArrayRef sizes, bool contiguous = true) {
    TORCH_CHECK(t.is_cuda() && t.device() == owner.device(), "Engram tensors must share a CUDA device");
    TORCH_CHECK(t.scalar_type() == dtype && t.sizes() == sizes, "Invalid Engram tensor dtype/shape");
    TORCH_CHECK(!contiguous || t.is_contiguous(), "Engram tensor must be contiguous");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "Engram tensor must be 16-byte aligned");
}
inline int rows(const at::Tensor& t) {
    TORCH_CHECK(t.is_cuda() && t.dim() >= 2 && t.size(0) >= 1 && t.size(0) <= 512,
                "Engram transition supports 1..512 CUDA rows");
    // Use PyTorch's CUDA runtime properties, matching Python dispatch.
    const auto* prop = at::cuda::getDeviceProperties(t.get_device());
    TORCH_CHECK(prop->major == 10 && prop->minor == 3 && prop->multiProcessorCount == 148,
                "Engram transition requires SM103 with 148 SMs");
    return t.size(0);
}
inline void checkSf(const at::Tensor& sf, const at::Tensor& owner, int m, int words) {
    check(sf, owner, at::kInt, {m, words}, false);
    TORCH_CHECK(sf.stride(0) == 1 && sf.stride(1) >= (m + 3) / 4 * 4,
                "Engram packed scales require TMA-aligned column-major storage");
    TORCH_CHECK(sf.stride(1) % 4 == 0, "Engram scale stride must be a multiple of four");
}
inline CUtensorMap tensorMap(const at::Tensor&   t,
                             CUtensorMapDataType type,
                             uint64_t            inner,
                             uint64_t            outer,
                             uint32_t            box_inner,
                             uint32_t            box_outer,
                             uint64_t            stride,
                             int                 swizzle      = 0,
                             uint64_t            depth        = 0,
                             uint64_t            depth_stride = 0) {
    CUtensorMap      map;
    const cuuint64_t dims[]            = {inner, outer, depth};
    const cuuint64_t strides[]         = {stride * t.element_size(), depth_stride * t.element_size()};
    const cuuint32_t box[]             = {swizzle ? uint32_t(swizzle / t.element_size()) : box_inner, box_outer, 1};
    const cuuint32_t element_strides[] = {1, 1, 1};
    const auto       status = cuTensorMapEncodeTiled(&map,
                                                     type,
                                                     depth ? 3 : 2,
                                                     t.data_ptr(),
                                                     dims,
                                                     strides,
                                                     box,
                                                     element_strides,
                                                     CU_TENSOR_MAP_INTERLEAVE_NONE,
                                                     swizzle ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_NONE,
                                                     CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                                     CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    TORCH_CHECK(status == CUDA_SUCCESS, "Engram TMA descriptor failed: ", int(status));
    return map;
}
template<class Kernel, class... Args>
inline void launch(Kernel kernel, int threads, int smem, int cluster, dim3 grid, Args... args) {
    // Attributes are device-specific. Warmup sets each specialization once;
    // graph capture/replay never initializes workspaces or changes attributes.
    static std::mutex                            mutex;
    static std::set<std::pair<int, const void*>> initialized;
    int                                          device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (initialized.emplace(device, reinterpret_cast<const void*>(kernel)).second) {
            C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        }
    }
    cudaLaunchAttribute attrs[2]{};
    attrs[0].id                                         = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    attrs[1].id                                         = cudaLaunchAttributeClusterDimension;
    attrs[1].val.clusterDim                             = {static_cast<unsigned>(cluster), 1, 1};
    cudaLaunchConfig_t config{};
    config.gridDim          = grid;
    config.blockDim         = dim3(threads, 1, 1);
    config.dynamicSmemBytes = smem;
    config.stream           = at::cuda::getCurrentCUDAStream();
    config.attrs            = attrs;
    config.numAttrs         = cluster > 1 ? 2 : 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}
}  // namespace rtp_llm::engram
