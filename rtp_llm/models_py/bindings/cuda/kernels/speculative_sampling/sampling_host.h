#pragma once

#include <cuda_runtime.h>

namespace rtp_llm {

struct RejectionSamplingParams;
struct RejectionSamplingLaunchConfig;

cudaError_t dispatchRejectionSampling(const RejectionSamplingParams&       params,
                                      const RejectionSamplingLaunchConfig& config,
                                      cudaStream_t                         stream);

}  // namespace rtp_llm
