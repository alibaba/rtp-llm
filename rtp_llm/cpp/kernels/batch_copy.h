#pragma once

namespace rtp_llm {
namespace kernels {

struct BatchCopyConfig {
    size_t uniform_size;
    bool   is_fully_aligned;
};

BatchCopyConfig getBatchCopyConfig(const size_t* bytes_host, size_t batch_size);

void invokeBatchCopy(void* const*           dst,
                     void const* const*     src,
                     size_t*                bytes,
                     size_t                 batch_size,
                     const BatchCopyConfig& config,
#if USING_CUDA
                     cudaStream_t           stream);
#elif USING_ROCM
                     hipStream_t            stream);
#endif

}  // namespace kernels
}  // namespace rtp_llm