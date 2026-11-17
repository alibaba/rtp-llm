#pragma once

#include <cuda.h>

#include <deep_jit/utils/exception.hpp>
#include <deep_jit/utils/lazy.hpp>

DJ_STATIC_ASSERT(CUDA_VERSION >= 12040, "DeepJIT requires CUDA 12.4 or newer");

namespace deep_jit::cuda::driver {

DJ_DECL_LAZY_DL_HANDLE(get_cuda_handle, "libcuda.so.1");

DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuGetErrorName);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuGetErrorString);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuDeviceGet);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuDeviceGetAttribute);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuFuncSetAttribute);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLaunchKernelEx);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuStreamBatchMemOp);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuTensorMapEncodeTiled);
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLibraryLoadFromFile)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLibraryUnload)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLibraryGetKernelCount)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuLibraryEnumerateKernels)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuKernelGetFunction)

// VMM operations
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemAddressFree)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemAddressReserve)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemCreate)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemExportToShareableHandle)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemGetAccess)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemGetAllocationGranularity)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemGetAllocationPropertiesFromHandle)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemImportFromShareableHandle)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemMap)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemMapArrayAsync)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemRelease)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemRetainAllocationHandle)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemSetAccess)
DJ_DECL_LAZY_DL_FUNCTION(get_cuda_handle, cuMemUnmap)

inline void check_cuda_driver(const CUresult error, const char* expression) {
    if (error == CUDA_SUCCESS)
        return;

    const char* error_name = "unknown";
    const char* error_description = "unknown";
    lazy_cuGetErrorName(error, &error_name);
    lazy_cuGetErrorString(error, &error_description);
    DJ_PANIC("{} failed with CUDA error {} ({}): {}",
             expression,
             static_cast<int>(error),
             error_name,
             error_description);
}

#ifndef DJ_CUDA_DRIVER_CHECK
#define DJ_CUDA_DRIVER_CHECK(expr) ::deep_jit::cuda::driver::check_cuda_driver((expr), #expr)
#endif

}  // namespace deep_jit::cuda::driver
