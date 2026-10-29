#pragma once

#include <cuda_runtime.h>

#if __has_include(<cuda_profiler_api.h>)
#include <cuda_profiler_api.h>
#else
// CUDA 13 removed cuda_profiler_api.h while libcudart still exports these APIs.
extern "C" cudaError_t CUDARTAPI cudaProfilerStart(void);
extern "C" cudaError_t CUDARTAPI cudaProfilerStop(void);
#endif
