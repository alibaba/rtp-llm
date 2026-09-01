#pragma once

#include "rtp_llm/cpp/runtime/DeviceTypes.h"

namespace rtp_llm {

#if USING_CUDA

inline constexpr auto kDeviceSuccess           = cudaSuccess;
inline constexpr auto kDeviceCaptureStatusNone = cudaStreamCaptureStatusNone;

#elif USING_ROCM

inline constexpr auto kDeviceSuccess           = hipSuccess;
inline constexpr auto kDeviceCaptureStatusNone = hipStreamCaptureStatusNone;

#endif

#if USING_CUDA || USING_ROCM

void checkDeviceError(DeviceError error, const char* file, int line);
void checkDeviceErrorInDebug(DeviceStream stream, const char* file, int line);

#endif

}  // namespace rtp_llm

#define RTP_LLM_DEVICE_CHECK(value) rtp_llm::checkDeviceError((value), __FILE__, __LINE__)
#define RTP_LLM_DEVICE_CHECK_DEBUG(stream) rtp_llm::checkDeviceErrorInDebug((stream), __FILE__, __LINE__)
