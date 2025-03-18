#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "nvToolsExt.h"

namespace fastertransformer {
    void CudaDevice::perfRangePush(const std::string& name) const {
        nvtxRangePushA(name.c_str());        
    }

    void CudaDevice::perfRangePop() const {
        nvtxRangePop();
    }
} // namespace fastertransformer