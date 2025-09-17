#include "rtp_llm/cpp/cuda/cutlass/cutlass_kernels/fp8_group_gemm/fp8_group_gemm.h"

namespace rtp_llm {

int32_t get_sm_version_num() {
    int32_t major_capability, minor_capability;
    cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor, 0);
    int32_t version_num = major_capability * 10 + minor_capability;
    return version_num;
}

}  // namespace rtp_llm