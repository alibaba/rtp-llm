#include <cuda.h>
#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#if CUDA_VERSION >= 12090
#include "nvtx3/nvToolsExt.h"
#else
#include "nvToolsExt.h"
#endif

namespace rtp_llm {
void CudaDevice::perfRangePush(const std::string& name) const {
    nvtxRangePushA(name.c_str());
}

void CudaDevice::perfRangePop() const {
    nvtxRangePop();
}
}  // namespace rtp_llm
