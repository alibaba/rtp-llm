#include "src/fastertransformer/devices/cuda_impl/CudaDevice.h"
#include "src/fastertransformer/cuda/Dispatch.h"
#include "src/fastertransformer/kernels/mla_kernels/mla_merge_transpose_kernel.h"
#include "src/fastertransformer/devices/utils/DebugUtils.h"
#include "src/fastertransformer/core/BufferHelper.h"

namespace fastertransformer {
BufferPtr CudaDevice::mlaQKVGemm(const AttentionLayerParams& params) {
    throw std::runtime_error("mlaQKVGemm is not implemented, please use flashinfer or flashmla instead");
}
}  // namespace fastertransformer