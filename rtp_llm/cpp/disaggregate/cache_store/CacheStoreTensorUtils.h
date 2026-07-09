#pragma once

#include <torch/torch.h>
#include <c10/core/DeviceGuard.h>

#include "rtp_llm/models_py/bindings/core/ExecOps.h"

namespace rtp_llm {

inline torch::TensorOptions cacheStoreByteTensorOptions(bool gpu_mem) {
    auto options = torch::TensorOptions().dtype(torch::kUInt8);
    if (!gpu_mem) {
        return options.device(torch::kCPU);
    }

    const int device_id = isRuntimeInitialized() ? static_cast<int>(getDeviceId()) : 0;
    return options.device(torch::Device(torch::kCUDA, device_id));
}

inline torch::Tensor cacheStoreByteTensorFromBlob(void* data, int64_t len, bool gpu_mem) {
    return torch::from_blob(data, {len}, cacheStoreByteTensorOptions(gpu_mem));
}

inline void
cacheStoreCopyByteTensor(void* dst, int64_t dst_len, bool dst_gpu, void* src, int64_t src_len, bool src_gpu) {
    const bool needs_gpu = dst_gpu || src_gpu;
    if (needs_gpu) {
        const int        device_id = isRuntimeInitialized() ? static_cast<int>(getDeviceId()) : 0;
        c10::DeviceGuard guard(c10::Device(c10::kCUDA, static_cast<c10::DeviceIndex>(device_id)));
        cudaPreRun(device_id);
        execNoBlockCopy(
            {cacheStoreByteTensorFromBlob(dst, dst_len, dst_gpu), cacheStoreByteTensorFromBlob(src, src_len, src_gpu)});
        return;
    }
    execNoBlockCopy(
        {cacheStoreByteTensorFromBlob(dst, dst_len, false), cacheStoreByteTensorFromBlob(src, src_len, false)});
}

}  // namespace rtp_llm
