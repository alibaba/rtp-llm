#pragma once

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#endif

namespace rtp_llm {

#if USING_CUDA
using DeviceError         = cudaError_t;
using DeviceStream        = cudaStream_t;
using DeviceCaptureStatus = cudaStreamCaptureStatus;
#elif USING_ROCM
using DeviceError         = hipError_t;
using DeviceStream        = hipStream_t;
using DeviceCaptureStatus = hipStreamCaptureStatus;
#else
using DeviceStream = void*;
#endif

}  // namespace rtp_llm
