#include "rtp_llm/cpp/runtime/DeviceError.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

#if USING_CUDA || USING_ROCM

namespace {

#if USING_CUDA
const char* deviceErrorString(DeviceError error) {
    return cudaGetErrorString(error);
}

DeviceError deviceStreamIsCapturing(DeviceStream stream, DeviceCaptureStatus* status) {
    return cudaStreamIsCapturing(stream, status);
}

DeviceError deviceSynchronize() {
    return cudaDeviceSynchronize();
}

DeviceError deviceGetLastError() {
    return cudaGetLastError();
}
#else
const char* deviceErrorString(DeviceError error) {
    return hipGetErrorString(error);
}

DeviceError deviceStreamIsCapturing(DeviceStream stream, DeviceCaptureStatus* status) {
    return hipStreamIsCapturing(stream, status);
}

DeviceError deviceSynchronize() {
    return hipDeviceSynchronize();
}

DeviceError deviceGetLastError() {
    return hipGetLastError();
}
#endif

}  // namespace

void checkDeviceError(DeviceError error, const char* file, int line) {
    if (error != kDeviceSuccess) {
        throwRuntimeError(file, line, fmtstr("device runtime error: %s", deviceErrorString(error)));
    }
}

void checkDeviceErrorInDebug(DeviceStream stream, const char* file, int line) {
    if (!Logger::getEngineLogger().isDebugMode()) {
        return;
    }

    DeviceCaptureStatus capture_status;
    checkDeviceError(deviceStreamIsCapturing(stream, &capture_status), file, line);
    if (capture_status == kDeviceCaptureStatusNone) {
        checkDeviceError(deviceSynchronize(), file, line);
    }
    checkDeviceError(deviceGetLastError(), file, line);
    RTP_LLM_LOG_DEBUG("device debug synchronization completed at %s:%d", file, line);
}

#endif

}  // namespace rtp_llm
