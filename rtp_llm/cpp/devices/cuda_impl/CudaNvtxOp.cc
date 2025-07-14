#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "nvToolsExt.h"

namespace rtp_llm {
void CudaDevice::perfRangePush(const std::string& name) const {
    nvtxRangePushA(name.c_str());
}

void CudaDevice::perfRangePop() const {
    nvtxRangePop();
}
}  // namespace rtp_llm