#include "rtp_llm/cpp/devices/cuda_impl/CudaDevice.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/kernels/mla_kernels/mla_merge_transpose_kernel.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include "rtp_llm/cpp/core/BufferHelper.h"

namespace rtp_llm {
BufferPtr CudaDevice::mlaQKVGemm(const AttentionLayerParams& params) {
    throw std::runtime_error("mlaQKVGemm is not implemented, please use flashinfer or flashmla instead");
}
}  // namespace rtp_llm