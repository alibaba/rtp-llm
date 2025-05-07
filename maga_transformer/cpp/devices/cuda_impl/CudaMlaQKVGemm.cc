#include "maga_transformer/cpp/devices/cuda_impl/CudaDevice.h"
#include "maga_transformer/cpp/cuda/Dispatch.h"
#include "maga_transformer/cpp/kernels/mla_kernels/mla_merge_transpose_kernel.h"
#include "maga_transformer/cpp/devices/utils/DebugUtils.h"
#include "maga_transformer/cpp/core/BufferHelper.h"

namespace rtp_llm {
BufferPtr CudaDevice::mlaQKVGemm(const AttentionLayerParams& params) {
    throw std::runtime_error("mlaQKVGemm is not implemented, please use flashinfer or flashmla instead");
}
}  // namespace rtp_llm