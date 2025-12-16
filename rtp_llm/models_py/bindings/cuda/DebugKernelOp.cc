#include "rtp_llm/models_py/bindings/cuda/DebugKernelOp.h"
#include "rtp_llm/cpp/core/Dispatch.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"

namespace rtp_llm {

void debugKernel(const torch::Tensor& data,
                 int64_t              start_row,
                 int64_t              start_col,
                 int64_t              m,
                 int64_t              n,
                 int64_t              row_len,
                 int64_t              info_id) {
    // Validate input tensor
    RTP_LLM_CHECK_WITH_INFO(data.is_cuda(), "Input tensor must be on CUDA device");
    RTP_LLM_CHECK_WITH_INFO(data.is_contiguous(), "Input tensor must be contiguous");

    // Get CUDA stream
    auto stream = c10::cuda::getCurrentCUDAStream(data.get_device());

    // Dispatch based on data type
    DISPATCH_CUDA_FUNCTION_DATA_TYPE(torchDTypeToDataType(data.dtype()),
                                     invoke_debug_kernel2,
                                     data.data_ptr(),
                                     static_cast<int>(start_row),
                                     static_cast<int>(start_col),
                                     static_cast<int>(m),
                                     static_cast<int>(n),
                                     static_cast<int>(row_len),
                                     static_cast<int>(info_id),
                                     stream);
}

}  // namespace rtp_llm