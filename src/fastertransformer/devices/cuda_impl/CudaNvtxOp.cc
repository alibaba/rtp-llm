#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "nvToolsExt.h"

namespace fastertransformer {
    void CudaDevice::perfRangePush(const std::string& name) const {
        if (!enable_device_perf_) {
            return;
        }
        nvtxRangePushA(name.c_str());        
    }

    void CudaDevice::perfRangePop() const {
        if (!enable_device_perf_) {
            return;
        }
        nvtxRangePop();
    }
} // namespace fastertransformer