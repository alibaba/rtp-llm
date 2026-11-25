#pragma once

#if USING_CUDA
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>

namespace rtp_llm {

// The caller must order reads of tensor on producer. The returned tensor can
// be read by CPU code immediately, with no dependency on the destination stream.
torch::Tensor copyCudaTensorToPinnedHost(const torch::Tensor& tensor, const c10::cuda::CUDAStream& producer);

}  // namespace rtp_llm
#endif
